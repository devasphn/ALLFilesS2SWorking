import asyncio
import json
import logging
import warnings
import time
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from aiohttp import web, WSMsgType
from aiortc import (RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer, MediaStreamTrack)
import av

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voice-agent")

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
<title>Voice Agent</title>
</head>
<body style="font-family:sans-serif;text-align:center;margin-top:40px;">
<h1>🎙️ Real-Time Speech-to-Speech Agent</h1>
<div id="status">Loading models... Please wait</div>
<button id="startBtn" onclick="start()" disabled>Start</button>
<button id="stopBtn" onclick="stop()" disabled>Stop</button>
<audio id="audioOut" autoplay></audio>
<script>
let pc, ws, stream;
const statusDiv = document.getElementById('status');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
function updateStatus(msg) { statusDiv.textContent = msg; }
async function start() {
  startBtn.disabled = true; stopBtn.disabled = false;
  updateStatus("Requesting microphone...");
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  stream.getTracks().forEach(t=>pc.addTrack(t,stream));
  pc.ontrack = e => { document.getElementById('audioOut').srcObject = e.streams[0]; updateStatus("🤖 AI responding..."); };
  pc.onicecandidate = e => { if(e.candidate && ws) ws.send(JSON.stringify({type:'ice-candidate',candidate:e.candidate.toJSON()})); };
  ws = new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+'/ws');
  ws.onopen = async () => { let offer=await pc.createOffer(); await pc.setLocalDescription(offer); ws.send(JSON.stringify(offer)); };
  ws.onmessage = async m => { let d=JSON.parse(m.data);
    if(d.type=='answer') await pc.setRemoteDescription(new RTCSessionDescription(d));
    else if(d.type=='ice-candidate') await pc.addIceCandidate(new RTCIceCandidate(d.candidate)); };
  ws.onclose = stop;
  updateStatus("Speak now!");
}
function stop() {
  startBtn.disabled = false; stopBtn.disabled = true;
  updateStatus("Stopped - Click Start to reconnect");
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(pc) pc.close();
  if(ws) ws.close();
}
function checkReady() {
  fetch('/health').then(r => r.json()).then(d=>{
    if(d.ready){startBtn.disabled=false;updateStatus("Ready - Click Start!");}
    else{startBtn.disabled=true;updateStatus("Loading models... Please wait"); setTimeout(checkReady,3000);}
  }).catch(()=>{setTimeout(checkReady,3000);});
}
checkReady();
</script>
</body>
</html>
"""

# Global state
executor = ThreadPoolExecutor(max_workers=2)
models_ready = False
vad = None
uv_pipe = None
tts_model = None

# Background – load models and mark as ready
async def preload_models():
    global vad, uv_pipe, tts_model, models_ready
    try:
        logger.info("Loading Silero VAD...")
        from silero_vad import load_silero_vad
        vad = load_silero_vad()
        logger.info("Silero VAD loaded.")

        logger.info("Loading Ultravox...")
        from transformers import pipeline
        uv_pipe = pipeline(
            "automatic-speech-recognition",
            "fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="float16"
        )
        logger.info("Ultravox loaded.")

        logger.info("Loading Chatterbox TTS...")
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("Chatterbox TTS loaded.")

        models_ready = True
        logger.info("✅ Models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type="text/html")

async def health_handler(request):
    return web.json_response({"ready": models_ready})

# A minimal dummy audio handler (replace with full pipeline when testing ready)
class DummyTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self): super().__init__(); self.pts = 0
    async def recv(self):
        samples = np.zeros(960, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis,:], format="s16", layout="mono")
        frame.pts = self.pts
        frame.sample_rate = 48000
        self.pts += 960
        await asyncio.sleep(0.02)
        return frame

async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30); await ws.prepare(request)
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            # TODO: Replace DummyTrack() with Processor(track,...) when integrating main pipeline
            dummy = DummyTrack()
            pc.addTrack(dummy)
    async for msg in ws:
        try:
            data = json.loads(msg.data)
            if data.get("type") == "offer":
                await pc.setRemoteDescription(RTCSessionDescription(**data))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
            elif data.get("type") == "ice-candidate":
                await pc.addIceCandidate(RTCIceCandidate(**data["candidate"]))
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    await pc.close()
    return ws

async def main():
    # Start web server immediately so /health is always available
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 7860))
    await web.TCPSite(runner, "0.0.0.0", port).start()
    logger.info(f"🌎 Web server started on port {port}")
    logger.info("🚦 /health endpoint now up")
    # Now background-load models
    asyncio.create_task(preload_models())
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
