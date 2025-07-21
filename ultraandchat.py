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
import collections

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# Minimal setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)  # Reduce noise
logger = logging.getLogger(__name__)

# Globals
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=2)
pcs = set()

# Minimal HTML with better error handling
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant - Final</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #1a1a1a; color: white; text-align: center; }
        button { padding: 20px 40px; font-size: 18px; margin: 20px; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; }
        .start { background: #28a745; color: white; }
        .stop { background: #dc3545; color: white; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; font-size: 18px; font-weight: bold; }
        .connected { background: #28a745; }
        .disconnected { background: #dc3545; }
        .processing { background: #ffc107; color: black; }
        .conversation { background: #333; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: left; max-height: 300px; overflow-y: auto; }
        .message { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .user { background: #007bff; }
        .ai { background: #28a745; }
        audio { width: 100%; margin: 20px 0; background: #333; }
        .debug { background: #222; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace; font-size: 12px; text-align: left; max-height: 150px; overflow-y: auto; }
    </style>
</head>
<body>
    <h1>🎙️ Voice Assistant</h1>
    
    <button id="startBtn" class="start" onclick="start()">Start</button>
    <button id="stopBtn" class="stop" onclick="stop()" disabled>Stop</button>
    
    <div id="status" class="status disconnected">Ready</div>
    
    <audio id="audioPlayer" controls autoplay></audio>
    
    <div id="conversation" class="conversation">Conversation will appear here...</div>
    
    <div id="debug" class="debug">Debug info...</div>

    <script>
        let pc, ws, localStream, connectionTimeout;
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const audioPlayer = document.getElementById('audioPlayer');
        const conversationDiv = document.getElementById('conversation');
        const debugDiv = document.getElementById('debug');

        function log(msg) {
            const time = new Date().toLocaleTimeString();
            debugDiv.innerHTML += time + ': ' + msg + '<br>';
            debugDiv.scrollTop = debugDiv.scrollHeight;
            console.log(msg);
        }

        function updateStatus(msg, className) {
            statusDiv.textContent = msg;
            statusDiv.className = 'status ' + className;
            log('Status: ' + msg);
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
            
            // Connection timeout
            connectionTimeout = setTimeout(() => {
                log('Connection timeout - retrying...');
                stop();
                setTimeout(start, 2000);
            }, 15000);
            
            try {
                // Get microphone
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 48000,
                        channelCount: 1
                    }
                });
                log('✅ Microphone ready');

                // Create peer connection with multiple STUN servers
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' },
                        { urls: 'stun:stun.cloudflare.com:3478' },
                        { urls: 'stun:stun.relay.metered.ca:80' }
                    ],
                    iceCandidatePoolSize: 10
                });

                // Add tracks
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                    log('Added track: ' + track.kind);
                });

                // Handle remote audio
                pc.ontrack = event => {
                    log('🎵 Remote audio received');
                    if (event.streams[0]) {
                        audioPlayer.srcObject = event.streams[0];
                        
                        audioPlayer.onloadeddata = () => {
                            log('Audio data loaded');
                        };
                        
                        audioPlayer.onplay = () => {
                            updateStatus('🔊 AI Speaking', 'processing');
                            log('Audio playing');
                        };
                        
                        audioPlayer.onended = () => {
                            updateStatus('🎤 Listening', 'connected');
                            log('Audio ended');
                        };
                        
                        audioPlayer.onerror = (e) => {
                            log('Audio error: ' + (e.target.error?.message || 'unknown'));
                        };
                    }
                };

                // Connection monitoring
                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    log('Connection: ' + state);
                    
                    if (state === 'connected') {
                        clearTimeout(connectionTimeout);
                        updateStatus('🎤 Listening', 'connected');
                        stopBtn.disabled = false;
                        log('🎉 Connected successfully!');
                    } else if (state === 'failed') {
                        log('❌ Connection failed');
                        stop();
                        setTimeout(start, 3000);
                    } else if (state === 'closed') {
                        log('Connection closed');
                        stop();
                    }
                };

                pc.oniceconnectionstatechange = () => {
                    log('ICE: ' + pc.iceConnectionState);
                };

                pc.onicecandidate = event => {
                    if (event.candidate) {
                        log('ICE candidate: ' + event.candidate.type);
                        if (ws?.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate.toJSON()
                            }));
                        }
                    }
                };

                // WebSocket with retry
                const wsUrl = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws';
                log('Connecting to: ' + wsUrl);
                ws = new WebSocket(wsUrl);

                ws.onopen = async () => {
                    log('🌐 WebSocket connected');
                    try {
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        ws.send(JSON.stringify(offer));
                        log('📤 Offer sent');
                    } catch (err) {
                        log('❌ Offer failed: ' + err.message);
                        stop();
                    }
                };

                ws.onmessage = async event => {
                    try {
                        const data = JSON.parse(event.data);
                        log('📥 ' + data.type);
                        
                        if (data.type === 'answer') {
                            await pc.setRemoteDescription(new RTCSessionDescription(data));
                            log('✅ Answer processed');
                        } else if (data.type === 'user_speech') {
                            addMessage(data.text, true);
                        } else if (data.type === 'ai_response') {
                            addMessage(data.text, false);
                        } else if (data.type === 'processing') {
                            updateStatus('🧠 Processing...', 'processing');
                        }
                    } catch (err) {
                        log('Message error: ' + err.message);
                    }
                };

                ws.onclose = () => {
                    log('WebSocket closed');
                    if (startBtn.disabled) {
                        setTimeout(() => {
                            if (startBtn.disabled) start();
                        }, 5000);
                    }
                };

                ws.onerror = () => {
                    log('WebSocket error');
                };

            } catch (err) {
                clearTimeout(connectionTimeout);
                log('❌ Error: ' + err.message);
                updateStatus('Error: ' + err.message, 'disconnected');
                stop();
            }
        }

        function stop() {
            clearTimeout(connectionTimeout);
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (pc) {
                pc.close();
                pc = null;
            }
            
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            if (audioPlayer.srcObject) {
                audioPlayer.srcObject = null;
            }
            
            updateStatus('Disconnected', 'disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        // Auto-retry on page focus
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && startBtn.disabled && pc?.connectionState !== 'connected') {
                log('Page visible, checking connection...');
                setTimeout(() => {
                    if (startBtn.disabled && pc?.connectionState !== 'connected') {
                        stop();
                        setTimeout(start, 1000);
                    }
                }, 2000);
            }
        });

        log('Interface ready');
    </script>
</body>
</html>
"""

# Robust ICE candidate parser
def parse_ice_candidate(candidate_str):
    try:
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
            'type': parts[7]
        }
    except:
        return None

# Working audio track
class WorkingAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_samples = []
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_size = 960  # 20ms at 48kHz
        self._lock = asyncio.Lock()
        
    async def recv(self):
        async with self._lock:
            # Create frame data
            if self._position < len(self._audio_samples):
                # Get samples for this frame
                end_pos = min(self._position + self._frame_size, len(self._audio_samples))
                frame_samples = self._audio_samples[self._position:end_pos]
                self._position += len(frame_samples)
                
                # Pad to frame size
                if len(frame_samples) < self._frame_size:
                    padding = [0] * (self._frame_size - len(frame_samples))
                    frame_samples.extend(padding)
            else:
                # Silence
                frame_samples = [0] * self._frame_size
            
            # Convert to numpy array
            frame_data = np.array(frame_samples, dtype=np.int16)
        
        # Create AV frame
        frame = av.AudioFrame.from_ndarray(
            frame_data.reshape(1, -1), format="s16", layout="mono"
        )
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_size
        
        return frame
    
    async def set_audio(self, audio_float32):
        """Set new audio data"""
        async with self._lock:
            if len(audio_float32) > 0:
                # Convert to int16 samples
                audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
                self._audio_samples = audio_int16.tolist()
                self._position = 0
                print(f"🔊 Audio set: {len(audio_int16)} samples, {len(audio_int16)/48000:.1f}s")
            else:
                self._audio_samples = []
                self._position = 0

# Audio processor
class AudioProcessor:
    def __init__(self, output_track):
        self.output_track = output_track
        self.audio_buffer = collections.deque(maxlen=48000 * 3)  # 3 seconds max
        self.is_processing = False
        self.ws = None
        self.last_process_time = 0
        
    def set_websocket(self, ws):
        self.ws = ws
    
    def add_audio_frame(self, audio_data, sample_rate):
        """Add incoming audio frame"""
        # Convert to float32
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Resample to 16kHz for processing
        if sample_rate != 16000:
            audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
        
        # Add to buffer
        self.audio_buffer.extend(audio_float.flatten())
        
        # Check if should process
        current_time = time.time()
        if (len(self.audio_buffer) >= 16000 and  # At least 1 second
            current_time - self.last_process_time > 2.0 and  # 2 second cooldown
            not self.is_processing):
            
            # Check for speech (simple energy check)
            recent_audio = np.array(list(self.audio_buffer)[-16000:])  # Last 1 second
            energy = np.sqrt(np.mean(recent_audio ** 2))
            
            if energy > 0.01:  # Speech detected
                self.last_process_time = current_time
                # Get audio to process (last 2-3 seconds)
                process_length = min(len(self.audio_buffer), 48000)  # Max 3 seconds
                audio_to_process = np.array(list(self.audio_buffer)[-process_length:])
                self.audio_buffer.clear()
                
                # Process async
                asyncio.create_task(self._process_speech(audio_to_process))
    
    async def _process_speech(self, audio_array):
        if self.is_processing:
            return
            
        self.is_processing = True
        start_time = time.time()
        
        try:
            print(f"🎯 Processing {len(audio_array)} samples ({len(audio_array)/16000:.1f}s)")
            
            # Send processing signal
            if self.ws:
                await self.ws.send_json({'type': 'processing'})
            
            # Run STT
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(executor, self._run_stt, audio_array)
            
            if not text:
                print("⚠️ No text generated")
                return
                
            stt_time = time.time() - start_time
            print(f"💬 STT ({stt_time*1000:.0f}ms): '{text}'")
            
            # Send user speech
            if self.ws:
                await self.ws.send_json({'type': 'user_speech', 'text': text})
            
            # Generate TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(executor, self._run_tts, text)
            
            if len(audio_output) > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                print(f"⚡ TTS ({tts_time*1000:.0f}ms), Total ({total_time*1000:.0f}ms)")
                
                # Send AI response
                if self.ws:
                    await self.ws.send_json({'type': 'ai_response', 'text': text})
                
                # Set audio for playback
                await self.output_track.set_audio(audio_output)
                
                # Wait for playback
                duration = len(audio_output) / 48000
                await asyncio.sleep(duration + 1.0)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
        finally:
            self.is_processing = False
    
    def _run_stt(self, audio_array):
        """Speech to text using Ultravox"""
        try:
            with torch.inference_mode():
                # Ultravox expects: {'audio': numpy_array, 'turns': [], 'sampling_rate': 16000}
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=25, do_sample=False, temperature=0.1)
                
                # Extract text
                if isinstance(result, list) and result:
                    item = result[0]
                    text = item.get('generated_text', '') if isinstance(item, dict) else str(item)
                else:
                    text = str(result)
                
                # Clean text
                text = text.strip()
                if text.startswith(("It seems like", "I think you", "You appear to")):
                    # Extract quoted content
                    if '"' in text:
                        parts = text.split('"')
                        if len(parts) >= 3:
                            text = parts[1].strip()
                
                return text[:80]  # Limit length
                
        except Exception as e:
            print(f"STT error: {e}")
            return ""
    
    def _run_tts(self, text):
        """Text to speech using ChatterboxTTS"""
        try:
            if len(text) < 2:
                return np.array([])
            
            # Limit text for speed
            text = text[:60]
            
            with torch.inference_mode():
                # ChatterboxTTS generates at 24kHz
                wav = tts_model.generate(text)
                if torch.is_tensor(wav):
                    wav = wav.cpu().numpy()
                
                wav = wav.flatten().astype(np.float32)
                print(f"🔊 TTS generated: {len(wav)} samples at 24kHz ({len(wav)/24000:.1f}s)")
                
                # Resample to 48kHz for WebRTC
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                
                # Normalize
                if np.max(np.abs(wav_48k)) > 0:
                    wav_48k = wav_48k / np.max(np.abs(wav_48k)) * 0.8
                
                print(f"🔊 Resampled to 48kHz: {len(wav_48k)} samples ({len(wav_48k)/48000:.1f}s)")
                return wav_48k
                
        except Exception as e:
            print(f"TTS error: {e}")
            return np.array([])

def load_models():
    """Load models with proper error handling"""
    global uv_pipe, tts_model
    
    try:
        print("Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ Ultravox loaded")
        
        print("Loading ChatterboxTTS...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ ChatterboxTTS loaded")
        
        # Test both models
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
        uv_pipe({'audio': dummy_audio, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=3)
        tts_model.generate("test")
        print("✅ Models tested successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30, timeout=60)
    await ws.prepare(request)
    
    print("WebSocket connected")
    
    # Create peer connection with robust configuration
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        RTCIceServer(urls="stun:stun.cloudflare.com:3478")
    ]))
    pcs.add(pc)
    
    # Create audio components
    audio_track = WorkingAudioTrack()
    processor = AudioProcessor(audio_track)
    processor.set_websocket(ws)
    
    @pc.on("track")
    def on_track(track):
        print(f"Track received: {track.kind}")
        if track.kind == "audio":
            pc.addTrack(audio_track)
            asyncio.create_task(handle_audio_track(track, processor))
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        print(f"Connection: {state}")
        if state in ["failed", "closed"]:
            pcs.discard(pc)
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data["type"] == "offer":
                        print("Processing offer...")
                        await pc.setRemoteDescription(RTCSessionDescription(
                            sdp=data["sdp"], type=data["type"]
                        ))
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await ws.send_json({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        })
                        print("Answer sent")
                        
                    elif data["type"] == "ice-candidate":
                        candidate_info = data["candidate"]
                        candidate_str = candidate_info.get("candidate", "")
                        
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
                                    sdpMid=candidate_info.get("sdpMid"),
                                    sdpMLineIndex=candidate_info.get("sdpMLineIndex")
                                ))
                            except Exception as e:
                                print(f"ICE candidate error: {e}")
                                
                except Exception as e:
                    print(f"Message processing error: {e}")
                    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket closing")
        pcs.discard(pc)
        await pc.close()
    
    return ws

async def handle_audio_track(track, processor):
    """Handle incoming audio from WebRTC"""
    try:
        while True:
            frame = await track.recv()
            audio_data = frame.to_ndarray().flatten()
            processor.add_audio_frame(audio_data, frame.sample_rate)
    except Exception as e:
        print(f"Audio track error: {e}")

async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def main():
    if not load_models():
        print("❌ Failed to load models")
        return
    
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("\n🚀 VOICE ASSISTANT - FINAL VERSION")
    print("📡 http://0.0.0.0:7860")
    print("🔧 Robust error handling")
    print("🔊 Complete audio pipeline")
    print("🎤 Speech detection optimized")
    print("🛑 Press Ctrl+C to stop\n")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    asyncio.run(main())
