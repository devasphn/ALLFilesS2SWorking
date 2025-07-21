import torch
import asyncio
import json
import logging
import numpy as np
import warnings
import time
import librosa
from concurrent.futures import ThreadPoolExecutor
import av

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Globals
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=2)
pcs = set()

# HTML Client
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Working Voice Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; text-align: center;
        }
        .container { max-width: 800px; margin: 0 auto; }
        button { 
            padding: 20px 40px; font-size: 18px; margin: 20px; border: none; border-radius: 10px; 
            cursor: pointer; font-weight: bold;
        }
        .start { background: #28a745; color: white; }
        .stop { background: #dc3545; color: white; }
        .status { 
            padding: 20px; margin: 20px 0; border-radius: 10px; font-size: 18px; font-weight: bold;
        }
        .connected { background: #28a745; }
        .disconnected { background: #dc3545; }
        .processing { background: #ffc107; color: black; }
        .conversation { 
            background: #333; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: left;
            max-height: 400px; overflow-y: auto;
        }
        .message { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .user { background: #007bff; }
        .ai { background: #28a745; }
        audio { width: 100%; margin: 20px 0; }
        .debug { 
            background: #222; padding: 15px; border-radius: 5px; margin: 20px 0; 
            font-family: monospace; font-size: 12px; text-align: left; max-height: 200px; overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Working Voice Assistant</h1>
        
        <button id="startBtn" class="start" onclick="start()">Start Talking</button>
        <button id="stopBtn" class="stop" onclick="stop()" disabled>Stop</button>
        
        <div id="status" class="status disconnected">Click Start to begin</div>
        
        <audio id="audioPlayer" controls autoplay></audio>
        
        <div id="conversation" class="conversation">Conversation will appear here...</div>
        
        <div id="debug" class="debug">Ready...</div>
    </div>

    <script>
        let pc, ws, localStream;
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const audioPlayer = document.getElementById('audioPlayer');
        const conversationDiv = document.getElementById('conversation');
        const debugDiv = document.getElementById('debug');

        function log(msg) {
            console.log(msg);
            debugDiv.innerHTML += new Date().toLocaleTimeString() + ': ' + msg + '<br>';
            debugDiv.scrollTop = debugDiv.scrollHeight;
        }

        function updateStatus(msg, className) {
            statusDiv.textContent = msg;
            statusDiv.className = 'status ' + className;
            log(msg);
        }

        function addMessage(text, isUser) {
            if (conversationDiv.innerHTML.includes('Conversation will appear here')) {
                conversationDiv.innerHTML = '';
            }
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'ai');
            div.innerHTML = '<strong>' + (isUser ? 'You' : 'AI') + ':</strong> ' + text;
            conversationDiv.appendChild(div);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        async function start() {
            startBtn.disabled = true;
            updateStatus('Getting microphone...', 'processing');
            
            try {
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: { 
                        echoCancellation: true, 
                        noiseSuppression: true,
                        sampleRate: 16000
                    }
                });
                log('✅ Microphone ready');

                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });

                localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

                pc.ontrack = event => {
                    log('📻 Audio track received');
                    if (event.streams[0]) {
                        audioPlayer.srcObject = event.streams[0];
                        audioPlayer.onplay = () => {
                            updateStatus('🔊 AI Speaking', 'processing');
                        };
                        audioPlayer.onended = () => {
                            updateStatus('🎤 Listening', 'connected');
                        };
                    }
                };

                pc.onconnectionstatechange = () => {
                    log('Connection: ' + pc.connectionState);
                    if (pc.connectionState === 'connected') {
                        updateStatus('🎤 Listening', 'connected');
                        stopBtn.disabled = false;
                    } else if (pc.connectionState === 'failed') {
                        stop();
                    }
                };

                pc.onicecandidate = event => {
                    if (event.candidate && ws?.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ice-candidate',
                            candidate: event.candidate.toJSON()
                        }));
                    }
                };

                ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');

                ws.onopen = async () => {
                    log('🌐 Connected to server');
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    ws.send(JSON.stringify(offer));
                };

                ws.onmessage = async event => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'answer') {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (data.type === 'user_speech') {
                        addMessage(data.text, true);
                    } else if (data.type === 'ai_response') {
                        addMessage(data.text, false);
                    } else if (data.type === 'processing') {
                        updateStatus('🧠 Processing your speech...', 'processing');
                    }
                };

                ws.onclose = () => {
                    log('Connection lost');
                    stop();
                };

            } catch (err) {
                log('❌ Error: ' + err.message);
                updateStatus('Error: ' + err.message, 'disconnected');
                stop();
            }
        }

        function stop() {
            if (ws) { ws.close(); ws = null; }
            if (pc) { pc.close(); pc = null; }
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            if (audioPlayer.srcObject) audioPlayer.srcObject = null;
            
            updateStatus('Disconnected', 'disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    </script>
</body>
</html>
"""

# FIXED: ICE Candidate Parser
def parse_ice_candidate(candidate_str):
    """Parse ICE candidate string into components"""
    try:
        # Remove "candidate:" prefix if present
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str[10:]
        
        parts = candidate_str.split()
        if len(parts) < 8:
            return None
            
        return {
            'foundation': parts[0],
            'component': int(parts[1]),
            'protocol': parts[2],
            'priority': int(parts[3]),
            'ip': parts[4],
            'port': int(parts[5]),
            'type': parts[7],  # parts[6] is "typ"
            'relatedAddress': None,
            'relatedPort': None
        }
    except Exception as e:
        logger.error(f"Failed to parse ICE candidate: {e}")
        return None

# Simple audio track
class SimpleAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_buffer = []
        self._position = 0
        self._timestamp = 0
        
    async def recv(self):
        # Generate 20ms of audio at 48kHz
        samples_per_frame = 960
        
        if self._position < len(self._audio_buffer):
            # Get audio data
            end_pos = min(self._position + samples_per_frame, len(self._audio_buffer))
            frame_data = np.array(self._audio_buffer[self._position:end_pos], dtype=np.int16)
            self._position += len(frame_data)
        else:
            # Silence
            frame_data = np.zeros(samples_per_frame, dtype=np.int16)
        
        # Pad if needed
        if len(frame_data) < samples_per_frame:
            padding = np.zeros(samples_per_frame - len(frame_data), dtype=np.int16)
            frame_data = np.concatenate([frame_data, padding])
        
        # Create frame
        frame = av.AudioFrame.from_ndarray(
            frame_data.reshape(1, -1), format="s16", layout="mono"
        )
        frame.pts = self._timestamp
        frame.sample_rate = 48000
        self._timestamp += samples_per_frame
        
        return frame
    
    def add_audio(self, audio_data):
        """Add audio data (float32 array at 48kHz)"""
        if len(audio_data) > 0:
            # Convert to int16
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            self._audio_buffer = audio_int16.tolist()
            self._position = 0
            logger.info(f"🔊 Added {len(audio_data)} samples, {len(audio_data)/48000:.1f}s")

# Simple speech processor
class SimpleSpeechProcessor:
    def __init__(self, output_track):
        self.output_track = output_track
        self.is_processing = False
        self.ws = None
        
    def set_websocket(self, ws):
        self.ws = ws
    
    async def process_audio(self, audio_data):
        if self.is_processing:
            return
            
        self.is_processing = True
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Processing {len(audio_data)} samples")
            
            # Send processing signal
            if self.ws:
                await self.ws.send_json({'type': 'processing'})
            
            # Run in executor
            loop = asyncio.get_running_loop()
            
            # Speech to text
            text = await loop.run_in_executor(executor, self._run_stt, audio_data)
            if not text:
                return
                
            stt_time = time.time() - start_time
            logger.info(f"💬 STT ({stt_time*1000:.0f}ms): {text}")
            
            # Send user speech
            if self.ws:
                await self.ws.send_json({'type': 'user_speech', 'text': text})
            
            # Generate TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(executor, self._run_tts, text)
            
            if len(audio_output) > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                logger.info(f"⚡ TTS ({tts_time*1000:.0f}ms), Total ({total_time*1000:.0f}ms)")
                
                # Send AI response
                if self.ws:
                    await self.ws.send_json({'type': 'ai_response', 'text': text})
                
                # Add to audio track
                self.output_track.add_audio(audio_output)
                
                # Wait for playback
                duration = len(audio_output) / 48000
                await asyncio.sleep(duration + 1)
                
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
        finally:
            self.is_processing = False
    
    def _run_stt(self, audio_data):
        """Speech to text"""
        try:
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_data,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=20)
                
                if isinstance(result, list) and result:
                    text = result[0].get('generated_text', '') if isinstance(result[0], dict) else str(result[0])
                else:
                    text = str(result)
                
                return text.strip()[:100]
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""
    
    def _run_tts(self, text):
        """Text to speech - OPTIMIZED"""
        try:
            if len(text) < 2:
                return np.array([])
            
            # Limit text length for speed
            text = text[:50]
            
            with torch.inference_mode():
                # Generate audio
                wav = tts_model.generate(text)
                if torch.is_tensor(wav):
                    wav = wav.cpu().numpy()
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                
                # Normalize
                if np.max(np.abs(wav_48k)) > 0:
                    wav_48k = wav_48k / np.max(np.abs(wav_48k)) * 0.7
                
                return wav_48k
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return np.array([])

# Simple audio buffer with VAD
class SimpleVAD:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000
        self.min_length = int(1.0 * self.sample_rate)  # 1 second minimum
        self.max_length = int(3.0 * self.sample_rate)  # 3 second maximum
        self.last_process = 0
        self.cooldown = 2.0  # 2 second cooldown
        
    def add_audio(self, audio_data):
        # Convert to 16kHz float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Simple preprocessing
        audio_data = np.clip(audio_data.flatten(), -1, 1)
        self.buffer.extend(audio_data)
        
        # Keep buffer manageable
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[-self.max_length:]
    
    def should_process(self):
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_process < self.cooldown:
            return False, None
            
        # Check length
        if len(self.buffer) < self.min_length:
            return False, None
            
        # Simple energy check
        audio_array = np.array(self.buffer[-self.min_length:])
        energy = np.sqrt(np.mean(audio_array**2))
        
        if energy > 0.01:  # Speech detected
            self.last_process = current_time
            result = audio_array.copy()
            self.buffer = []  # Clear buffer
            return True, result
            
        return False, None

def load_models():
    """Load models with optimized settings"""
    global uv_pipe, tts_model
    
    try:
        logger.info("Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("✅ Ultravox loaded")
        
        logger.info("Loading TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info("✅ TTS loaded")
        
        # Quick warmup
        dummy_audio = np.random.randn(8000).astype(np.float32) * 0.01
        uv_pipe({'audio': dummy_audio, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=5)
        tts_model.generate("test")
        logger.info("✅ Models ready")
        
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    logger.info("WebSocket connected")
    
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302")
    ]))
    pcs.add(pc)
    
    # Create audio track and processor
    audio_track = SimpleAudioTrack()
    processor = SimpleSpeechProcessor(audio_track)
    processor.set_websocket(ws)
    vad = SimpleVAD()
    
    @pc.on("track")
    def on_track(track):
        logger.info("Track received")
        if track.kind == "audio":
            pc.addTrack(audio_track)
            asyncio.create_task(process_incoming_audio(track, processor, vad))
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                if data["type"] == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=data["sdp"], type=data["type"]
                    ))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    })
                    
                elif data["type"] == "ice-candidate":
                    candidate_data = data["candidate"]
                    candidate_str = candidate_data.get("candidate", "")
                    
                    # FIXED: Parse candidate properly
                    parsed = parse_ice_candidate(candidate_str)
                    if parsed:
                        try:
                            await pc.addIceCandidate(RTCIceCandidate(
                                component=parsed["component"],
                                foundation=parsed["foundation"],
                                ip=parsed["ip"],
                                port=parsed["port"],
                                priority=parsed["priority"],
                                protocol=parsed["protocol"],
                                type=parsed["type"],
                                sdpMid=candidate_data.get("sdpMid"),
                                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                            ))
                        except Exception as e:
                            logger.error(f"Failed to add ICE candidate: {e}")
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        pcs.discard(pc)
        await pc.close()
    
    return ws

async def process_incoming_audio(track, processor, vad):
    """Process incoming audio frames"""
    try:
        while True:
            frame = await track.recv()
            audio_data = frame.to_ndarray().flatten()
            
            # Resample to 16kHz if needed
            if frame.sample_rate != 16000:
                audio_float = audio_data.astype(np.float32) / 32768.0 if audio_data.dtype == np.int16 else audio_data.astype(np.float32)
                audio_resampled = librosa.resample(audio_float, orig_sr=frame.sample_rate, target_sr=16000)
            else:
                audio_resampled = audio_data.astype(np.float32) / 32768.0 if audio_data.dtype == np.int16 else audio_data.astype(np.float32)
            
            vad.add_audio(audio_resampled)
            
            should_process, audio_to_process = vad.should_process()
            if should_process and audio_to_process is not None:
                await processor.process_audio(audio_to_process)
                
    except Exception as e:
        logger.error(f"Audio processing error: {e}")

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def main():
    if not load_models():
        return
    
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("\n🚀 FIXED VOICE ASSISTANT")
    print("📡 http://0.0.0.0:7860")
    print("🔧 ICE Candidate Issue Fixed")
    print("🎤 Should connect properly now")
    print("🛑 Press Ctrl+C to stop\n")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
