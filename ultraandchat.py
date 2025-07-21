import torch
import asyncio
import json
import logging
import numpy as np
import fractions
import warnings
import collections
import time
import librosa
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Enhanced Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for optimized event loop")
except ImportError:
    print("⚠️ uvloop not found, using default event loop")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
for logger_name in ['aioice.ice', 'aiortc.rtcpeerconnection', 'av.audio.resampler']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_worker")
pcs = set()

# --- Complete HTML Client ---
HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 UltraFast Voice Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; display: flex; align-items: center; justify-content: center; min-height: 100vh;
        }
        .container { 
            background: rgba(255,255,255,0.1); 
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
            padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            text-align: center; max-width: 700px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
        }
        h1 { margin-bottom: 30px; font-weight: 300; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        .controls { margin: 30px 0; }
        button { 
            background: linear-gradient(45deg, #00c851, #007e33);
            color: white; border: none; padding: 18px 36px; font-size: 18px; font-weight: 600;
            border-radius: 50px; cursor: pointer; margin: 10px; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-transform: uppercase; letter-spacing: 1px;
        }
        button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
        button:active { transform: translateY(-1px); }
        button:disabled { 
            background: linear-gradient(45deg, #6c757d, #495057); cursor: not-allowed; 
            transform: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .stop-btn { background: linear-gradient(45deg, #dc3545, #c82333); }
        .stop-btn:hover { background: linear-gradient(45deg, #c82333, #a71e2a); }
        
        .status { 
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.1em;
            transition: all 0.5s ease; border: 2px solid transparent;
        }
        .status.connected { 
            background: linear-gradient(45deg, #28a745, #20c997); 
            box-shadow: 0 0 20px rgba(40, 167, 69, 0.4);
        }
        .status.disconnected { 
            background: linear-gradient(45deg, #dc3545, #fd7e14);
            box-shadow: 0 0 20px rgba(220, 53, 69, 0.4);
        }
        .status.connecting { 
            background: linear-gradient(45deg, #ffc107, #fd7e14);
            box-shadow: 0 0 20px rgba(255, 193, 7, 0.4);
            animation: pulse 2s infinite;
        }
        .status.speaking { 
            background: linear-gradient(45deg, #007bff, #6610f2);
            box-shadow: 0 0 30px rgba(0, 123, 255, 0.6);
            animation: speaking 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes speaking {
            0%, 100% { transform: scale(1); box-shadow: 0 0 30px rgba(0, 123, 255, 0.6); }
            50% { transform: scale(1.03); box-shadow: 0 0 40px rgba(0, 123, 255, 0.8); }
        }
        
        .conversation { 
            margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.3); 
            border-radius: 10px; text-align: left; max-height: 300px; overflow-y: auto;
        }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user-msg { background: rgba(0, 123, 255, 0.3); margin-left: 20px; }
        .ai-msg { background: rgba(40, 167, 69, 0.3); margin-right: 20px; }
        .debug { 
            margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); 
            border-radius: 8px; font-family: 'Courier New', monospace; font-size: 11px;
            max-height: 150px; overflow-y: auto;
        }
        .latency { 
            margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.2); 
            border-radius: 6px; font-weight: bold; font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 UltraFast Voice AI</h1>
        <div class="controls">
            <button id="startBtn" onclick="start()">🎙️ Start Talking</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>⏹️ Stop</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <div id="latency" class="latency" style="display:none;">Response Time: <span id="latencyValue">0ms</span></div>
        <div id="conversation" class="conversation"></div>
        <div id="debug" class="debug">Debug logs will appear here...</div>
        <audio id="remoteAudio" autoplay playsinline controls></audio>
    </div>

    <script>
        let pc, ws, localStream, startTime;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const latencyDiv = document.getElementById('latency');
        const latencyValue = document.getElementById('latencyValue');
        const debugDiv = document.getElementById('debug');
        const conversationDiv = document.getElementById('conversation');

        function log(message) {
            console.log(message);
            debugDiv.innerHTML += new Date().toLocaleTimeString() + ': ' + message + '<br>';
            debugDiv.scrollTop = debugDiv.scrollHeight;
        }

        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-msg' : 'ai-msg'}`;
            messageDiv.textContent = `${isUser ? '👤 You' : '🤖 AI'}: ${text}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        function updateStatus(message, className) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${className}`;
            log(`Status: ${message}`);
        }

        function updateLatency(ms) {
            latencyValue.textContent = `${ms}ms`;
            latencyDiv.style.display = 'block';
        }

        async function start() {
            startBtn.disabled = true;
            updateStatus('🔄 Initializing...', 'connecting');
            debugDiv.innerHTML = '';
            conversationDiv.innerHTML = '';
            
            try {
                // Request microphone with optimal settings
                log('Requesting microphone access...');
                const constraints = {
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: { ideal: 48000 },
                        channelCount: 1
                    }
                };

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                log(`✅ Microphone access granted`);
                
                const audioTrack = localStream.getAudioTracks()[0];
                log(`Audio settings: ${JSON.stringify(audioTrack.getSettings())}`);

                // Create peer connection
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });

                log('RTCPeerConnection created');

                // Add local stream
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });

                // Handle remote audio with better error handling
                pc.ontrack = event => {
                    log(`🎵 Remote track received`);
                    if (event.streams[0]) {
                        remoteAudio.srcObject = event.streams[0];
                        
                        remoteAudio.oncanplay = () => {
                            remoteAudio.play().catch(err => {
                                log(`Play failed: ${err.message}`);
                            });
                        };

                        remoteAudio.onplaying = () => {
                            if (startTime) {
                                const latency = Date.now() - startTime;
                                updateLatency(latency);
                            }
                            updateStatus('🤖 AI Speaking...', 'speaking');
                        };
                        
                        remoteAudio.onended = () => {
                            if (pc && pc.connectionState === 'connected') {
                                updateStatus('🎙️ Listening...', 'connected');
                            }
                        };
                    }
                };

                // Handle ICE candidates
                pc.onicecandidate = event => {
                    if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ice-candidate',
                            candidate: event.candidate.toJSON()
                        }));
                    }
                };

                // Handle connection state changes
                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    log(`🔗 Connection: ${state}`);
                    
                    if (state === 'connected') {
                        updateStatus('🎙️ Listening...', 'connected');
                        stopBtn.disabled = false;
                    } else if (['failed', 'closed', 'disconnected'].includes(state)) {
                        stop();
                    }
                };

                // WebSocket connection
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws`);

                ws.onopen = async () => {
                    log('🌐 WebSocket connected');
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    ws.send(JSON.stringify(offer));
                };

                ws.onmessage = async event => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'answer') {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (data.type === 'speech_start') {
                        startTime = Date.now();
                        updateStatus('🧠 Processing...', 'connecting');
                    } else if (data.type === 'user_speech') {
                        addMessage(data.text, true);
                    } else if (data.type === 'ai_response') {
                        addMessage(data.text, false);
                    }
                };

                ws.onclose = () => {
                    if (pc && !['closed', 'failed'].includes(pc.connectionState)) {
                        stop();
                    }
                };

                ws.onerror = () => {
                    if (pc && !['closed', 'failed'].includes(pc.connectionState)) {
                        stop();
                    }
                };

            } catch (err) {
                log(`❌ Error: ${err.message}`);
                updateStatus(`❌ Error: ${err.message}`, 'disconnected');
                stop();
            }
        }

        function stop() {
            log('🛑 Stopping...');
            
            if (ws) {
                ws.onclose = ws.onerror = null;
                if (ws.readyState !== WebSocket.CLOSED) ws.close();
                ws = null;
            }
            
            if (pc) {
                pc.onconnectionstatechange = pc.onicecandidate = pc.ontrack = null;
                if (pc.connectionState !== 'closed') pc.close();
                pc = null;
            }
            
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            if (remoteAudio.srcObject) {
                remoteAudio.srcObject = null;
            }
            
            updateStatus('🔌 Disconnected', 'disconnected');
            latencyDiv.style.display = 'none';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
        }

        window.addEventListener('beforeunload', stop);
    </script>
</body>
</html>
"""

# --- ICE Candidate Parser ---
def parse_ice_candidate(candidate_string: str) -> dict:
    try:
        if candidate_string.startswith("candidate:"):
            candidate_string = candidate_string[10:]
        
        parts = candidate_string.split()
        if len(parts) < 8:
            return {}
        
        candidate_info = {
            'foundation': parts[0],
            'component': int(parts[1]),
            'protocol': parts[2],
            'priority': int(parts[3]),
            'ip': parts[4],
            'port': int(parts[5]),
            'type': parts[7]
        }
        
        # Parse additional attributes
        i = 8
        while i < len(parts) - 1:
            if parts[i] == "raddr":
                candidate_info['relatedAddress'] = parts[i + 1]
                i += 2
            elif parts[i] == "rport":
                candidate_info['relatedPort'] = int(parts[i + 1])
                i += 2
            elif parts[i] == "tcptype":
                candidate_info['tcpType'] = parts[i + 1]
                i += 2
            else:
                i += 1
                
        return candidate_info
        
    except Exception as e:
        logger.debug(f"ICE parse error: {e}")
        return {}

# --- Enhanced VAD System ---
class EnhancedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)  # Moderate sensitivity
        self.silero_model = None
        self.load_silero()
        
    def load_silero(self):
        try:
            logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', force_reload=False
            )
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.error(f"❌ Silero VAD error: {e}")
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if len(audio) == 0:
            return False
            
        # Energy threshold
        energy = np.mean(audio ** 2)
        if energy < 0.0005:
            return False
            
        # WebRTC VAD
        try:
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            
            # Check 20ms chunks
            chunk_size = 320
            speech_count = 0
            total_chunks = 0
            
            for i in range(0, len(audio_int16) - chunk_size + 1, chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    if self.webrtc_vad.is_speech(chunk.tobytes(), 16000):
                        speech_count += 1
                    total_chunks += 1
            
            if total_chunks > 0:
                speech_ratio = speech_count / total_chunks
                return speech_ratio > 0.3  # At least 30% speech
                
        except Exception as e:
            logger.debug(f"WebRTC VAD error: {e}")
            
        return energy > 0.01

# --- Audio Buffer with Better Processing ---
class SmartAudioBuffer:
    def __init__(self):
        self.sample_rate = 16000
        self.max_duration = 4.0  # seconds
        self.max_samples = int(self.max_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_speech_duration = 0.8  # Longer minimum for better quality
        self.min_samples = int(self.min_speech_duration * self.sample_rate)
        self.last_process_time = 0
        self.cooldown = 0.5
        
    def add_audio(self, audio_data: np.ndarray):
        # Better normalization
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Remove DC offset and normalize
        audio_data = audio_data - np.mean(audio_data)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / max(np.max(np.abs(audio_data)), 0.01)
        
        self.buffer.extend(audio_data.flatten())
    
    def should_process(self, vad: EnhancedVAD) -> Tuple[bool, Optional[np.ndarray]]:
        current_time = time.time()
        
        if current_time - self.last_process_time < self.cooldown:
            return False, None
            
        if len(self.buffer) < self.min_samples:
            return False, None
            
        audio_array = np.array(list(self.buffer), dtype=np.float32)
        
        # Better energy check
        if np.max(np.abs(audio_array)) < 0.01:
            return False, None
            
        if vad.detect_speech(audio_array, self.sample_rate):
            self.last_process_time = current_time
            return True, audio_array
            
        return False, None
    
    def reset(self):
        self.buffer.clear()

# --- Improved Audio Track ---
class ImprovedAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_queue = asyncio.Queue(maxsize=20)
        self._current_audio = None
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_size = 960  # 20ms
        
    async def recv(self):
        # Try to get new audio if current is finished
        if self._current_audio is None or self._position >= len(self._current_audio):
            try:
                self._current_audio = await asyncio.wait_for(self._audio_queue.get(), timeout=0.02)
                self._position = 0
            except asyncio.TimeoutError:
                self._current_audio = None
        
        # Fill frame
        frame_data = np.zeros(self._frame_size, dtype=np.int16)
        
        if self._current_audio is not None:
            remaining = len(self._current_audio) - self._position
            copy_size = min(self._frame_size, remaining)
            frame_data[:copy_size] = self._current_audio[self._position:self._position + copy_size]
            self._position += copy_size
        
        # Create audio frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame_data]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_size
        
        return audio_frame
    
    async def add_audio(self, audio_data: np.ndarray):
        if len(audio_data) > 0:
            # Convert to int16 and queue entire audio at once
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            try:
                await asyncio.wait_for(self._audio_queue.put(audio_int16), timeout=0.1)
            except asyncio.TimeoutError:
                logger.warning("Audio queue full")

# --- Enhanced Audio Processor ---
class EnhancedAudioProcessor:
    def __init__(self, output_track, executor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = SmartAudioBuffer()
        self.vad = EnhancedVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None
        self._stop_event = asyncio.Event()
        
    def set_websocket(self, ws):
        self.ws = ws
        
    def add_track(self, track):
        self.input_track = track
        
    async def start(self):
        if not self.task:
            logger.info("🎵 Starting audio processor")
            self.task = asyncio.create_task(self._audio_loop())
            
    async def stop(self):
        if self.task:
            logger.info("🛑 Stopping audio processor")
            self._stop_event.set()
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
    
    async def _audio_loop(self):
        try:
            while not self._stop_event.is_set():
                # Don't process while AI is speaking
                if self.is_processing:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except mediastreams.MediaStreamError:
                    logger.info("Audio stream ended")
                    break
                except Exception as e:
                    logger.error(f"Frame receive error: {e}")
                    break
                
                # Process audio
                try:
                    audio_data = frame.to_ndarray().flatten()
                    
                    # Convert to float32 and resample if needed
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    if frame.sample_rate != 16000:
                        audio_float = librosa.resample(
                            audio_float, orig_sr=frame.sample_rate, target_sr=16000
                        )
                    
                    self.buffer.add_audio(audio_float)
                    
                    # Check for speech
                    should_process, audio_array = self.buffer.should_process(self.vad)
                    if should_process:
                        logger.info(f"🎯 Speech detected: {len(audio_array)/16000:.2f}s")
                        self.buffer.reset()
                        asyncio.create_task(self._process_speech(audio_array))
                        
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")
                    continue
                    
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Audio processor stopped")
    
    def _run_inference(self, audio_array: np.ndarray) -> str:
        try:
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=30, do_sample=False, temperature=0.1)
                
                # Extract text
                if isinstance(result, list) and result:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        return item['generated_text'].strip()
                    elif isinstance(item, str):
                        return item.strip()
                elif isinstance(result, str):
                    return result.strip()
                    
                return ""
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ""
    
    def _run_tts(self, text: str) -> np.ndarray:
        try:
            if not text.strip():
                return np.array([], dtype=np.float32)
                
            with torch.inference_mode():
                wav = tts_model.generate(text)
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                return wav_48k
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return np.array([], dtype=np.float32)
    
    async def _process_speech(self, audio_array: np.ndarray):
        if self.is_processing:
            return
            
        start_time = time.time()
        self.is_processing = True
        
        try:
            # Signal processing start
            if self.ws and not self.ws.closed:
                await self.ws.send_json({'type': 'speech_start'})
            
            # Run STT
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self.executor, self._run_inference, audio_array)
            
            if not text:
                return
                
            logger.info(f"💬 User: '{text}'")
            stt_time = time.time() - start_time
            
            # Send user speech to client
            if self.ws and not self.ws.closed:
                await self.ws.send_json({'type': 'user_speech', 'text': text})
            
            # Run TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(self.executor, self._run_tts, text)
            
            if audio_output.size > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                
                logger.info(f"⚡ STT: {stt_time*1000:.0f}ms, TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
                
                # Send AI response to client
                if self.ws and not self.ws.closed:
                    await self.ws.send_json({'type': 'ai_response', 'text': text})
                
                # Queue complete audio
                await self.output_track.add_audio(audio_output)
                
                # Wait for audio to finish
                duration = len(audio_output) / 48000
                await asyncio.sleep(duration + 0.5)
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
        finally:
            self.is_processing = False

# --- Model Loading ---
def initialize_models() -> bool:
    global uv_pipe, tts_model
    
    try:
        logger.info("📥 Loading Ultravox...")
        uv_pipe = pipeline(
            "automatic-speech-recognition",
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("✅ Ultravox loaded")
        
        logger.info("📥 Loading TTS...")
        tts_model = ChatterboxTTS.from_pretrained()
        logger.info("✅ TTS loaded")
        
        # Warmup
        logger.info("🔥 Warming up...")
        dummy = np.random.randn(8000).astype(np.float32) * 0.01
        uv_pipe({'audio': dummy, 'sampling_rate': 16000, 'turns': []}, max_new_tokens=5)
        tts_model.generate("test")
        
        logger.info("🎉 Models ready!")
        return True
        
    except Exception as e:
        logger.error(f"Model init error: {e}", exc_info=True)
        return False

# --- WebSocket Handler ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    
    logger.info("🌐 WebSocket connected")
    
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302")
    ]))
    pcs.add(pc)
    processor = None
    
    @pc.on("track")
    def on_track(track):
        nonlocal processor
        if track.kind == "audio":
            logger.info("🎧 Audio track received")
            response_track = ImprovedAudioTrack()
            pc.addTrack(response_track)
            
            processor = EnhancedAudioProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"🔗 Connection: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if pc in pcs:
                pcs.remove(pc)
            if processor:
                await processor.stop()
    
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
                    candidate_data = data.get("candidate", {})
                    candidate_str = candidate_data.get("candidate", "")
                    parsed = parse_ice_candidate(candidate_str)
                    
                    if parsed:
                        candidate = RTCIceCandidate(
                            component=parsed["component"],
                            foundation=parsed["foundation"],
                            ip=parsed["ip"],
                            port=parsed["port"],
                            priority=parsed["priority"],
                            protocol=parsed["protocol"],
                            type=parsed["type"],
                            sdpMid=candidate_data.get("sdpMid"),
                            sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                            relatedAddress=parsed.get("relatedAddress"),
                            relatedPort=parsed.get("relatedPort"),
                            tcpType=parsed.get("tcpType")
                        )
                        await pc.addIceCandidate(candidate)
                        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket closing")
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

# --- HTTP Handlers ---
async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def health_handler(request):
    return web.json_response({
        "status": "ok",
        "models": uv_pipe is not None and tts_model is not None,
        "connections": len(pcs)
    })

# --- App Setup ---
async def on_shutdown(app):
    logger.info("Shutting down...")
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    executor.shutdown(wait=True)

async def main():
    if not initialize_models():
        return
        
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("🚀 Enhanced Speech-to-Speech Server Started!")
    print("📡 http://0.0.0.0:7860")
    print("🎯 Fixed audio quality and TTS streaming!")
    print("\n🛑 Press Ctrl+C to stop")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
