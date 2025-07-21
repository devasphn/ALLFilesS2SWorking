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
import io

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub
import torchaudio

# --- Optimized Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for optimized event loop")
except ImportError:
    print("⚠️ uvloop not found, using default event loop")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence noisy loggers
for logger_name in ['aioice', 'aiortc', 'av']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# --- Global Variables ---
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_worker")
pcs = set()

# --- Fixed HTML Client ---
HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 Fixed Voice Assistant</title>
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
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            text-align: center; max-width: 800px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
        }
        h1 { margin-bottom: 30px; font-weight: 300; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        .controls { margin: 30px 0; }
        button { 
            background: linear-gradient(45deg, #00c851, #007e33);
            color: white; border: none; padding: 20px 40px; font-size: 18px; font-weight: 600;
            border-radius: 50px; cursor: pointer; margin: 15px; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-transform: uppercase; letter-spacing: 1px;
        }
        button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
        button:disabled { 
            background: linear-gradient(45deg, #6c757d, #495057); cursor: not-allowed; 
            transform: none; opacity: 0.6;
        }
        .stop-btn { background: linear-gradient(45deg, #dc3545, #c82333); }
        
        .status { 
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.2em;
            transition: all 0.5s ease;
        }
        .status.connected { background: linear-gradient(45deg, #28a745, #20c997); }
        .status.disconnected { background: linear-gradient(45deg, #dc3545, #fd7e14); }
        .status.connecting { background: linear-gradient(45deg, #ffc107, #fd7e14); animation: pulse 2s infinite; }
        .status.processing { background: linear-gradient(45deg, #17a2b8, #007bff); animation: pulse 1s infinite; }
        .status.speaking { background: linear-gradient(45deg, #6f42c1, #007bff); animation: speaking 1.5s infinite; }
        
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        @keyframes speaking { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        
        .conversation { 
            margin-top: 25px; padding: 25px; background: rgba(0,0,0,0.3); 
            border-radius: 15px; text-align: left; max-height: 400px; overflow-y: auto;
        }
        .message { margin: 20px 0; padding: 20px; border-radius: 12px; line-height: 1.5; }
        .user-msg { background: rgba(0, 123, 255, 0.3); margin-left: 30px; }
        .ai-msg { background: rgba(40, 167, 69, 0.3); margin-right: 30px; }
        .message strong { display: block; margin-bottom: 8px; font-size: 1.1em; }
        
        .metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 20px; margin: 25px 0;
        }
        .metric { 
            padding: 20px; background: rgba(0,0,0,0.2); border-radius: 12px; text-align: center;
        }
        .metric-value { font-size: 2em; font-weight: bold; color: #00ff88; margin-bottom: 5px; }
        .metric-label { font-size: 0.9em; opacity: 0.8; }
        
        .debug { 
            margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.2); 
            border-radius: 12px; font-family: 'Courier New', monospace; font-size: 12px;
            max-height: 200px; overflow-y: auto; text-align: left;
        }
        
        .audio-controls {
            margin: 20px 0; padding: 20px; background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        .audio-controls audio {
            width: 100%; height: 40px;
        }
        
        .tip {
            margin-top: 20px; padding: 15px; background: rgba(0,123,255,0.2);
            border-radius: 10px; font-size: 0.9em; border-left: 4px solid #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Fixed Voice Assistant</h1>
        
        <div class="controls">
            <button id="startBtn" onclick="start()">🎙️ Start Conversation</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>⏹️ Stop</button>
        </div>
        
        <div id="status" class="status disconnected">🔌 Ready to Connect</div>
        
        <div class="metrics">
            <div class="metric">
                <div id="latencyValue" class="metric-value">0ms</div>
                <div class="metric-label">Response Time</div>
            </div>
            <div class="metric">
                <div id="connectionValue" class="metric-value">Offline</div>
                <div class="metric-label">Connection</div>
            </div>
        </div>
        
        <div class="audio-controls">
            <div style="margin-bottom: 10px; font-weight: bold;">🔊 AI Response Audio:</div>
            <audio id="remoteAudio" controls style="width: 100%;"></audio>
        </div>
        
        <div id="conversation" class="conversation">
            <div style="text-align: center; opacity: 0.7; font-style: italic;">
                Conversation will appear here...
            </div>
        </div>
        
        <div class="tip">
            💡 <strong>Tips:</strong> Speak clearly, wait for the AI to finish responding, and ensure your microphone is working properly.
        </div>
        
        <div id="debug" class="debug">Debug info will appear here...</div>
    </div>

    <script>
        let pc, ws, localStream, startTime, isConnected = false;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const debugDiv = document.getElementById('debug');
        const conversationDiv = document.getElementById('conversation');
        const latencyValue = document.getElementById('latencyValue');
        const connectionValue = document.getElementById('connectionValue');

        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            console.log(message);
            debugDiv.innerHTML += `${timestamp}: ${message}<br>`;
            debugDiv.scrollTop = debugDiv.scrollHeight;
        }

        function addMessage(text, isUser = false) {
            // Clear placeholder
            if (conversationDiv.innerHTML.includes('Conversation will appear here')) {
                conversationDiv.innerHTML = '';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-msg' : 'ai-msg'}`;
            messageDiv.innerHTML = `<strong>${isUser ? '👤 You said:' : '🤖 AI responds:'}</strong>${text}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        function updateStatus(message, className) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${className}`;
            log(`Status: ${message}`);
        }

        function updateMetrics(latency, connection) {
            if (latency !== undefined) latencyValue.textContent = `${latency}ms`;
            if (connection !== undefined) connectionValue.textContent = connection;
        }

        async function start() {
            if (isConnected) return;
            
            startBtn.disabled = true;
            updateStatus('🔄 Initializing microphone...', 'connecting');
            debugDiv.innerHTML = '';
            
            try {
                log('Requesting microphone access...');
                
                // High-quality audio constraints
                const constraints = {
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: { ideal: 48000, min: 16000 },
                        channelCount: 1,
                        volume: 1.0
                    }
                };

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                const track = localStream.getAudioTracks()[0];
                const settings = track.getSettings();
                log(`✅ Microphone: ${settings.sampleRate}Hz, Echo cancellation: ${settings.echoCancellation}`);

                updateStatus('🔄 Establishing connection...', 'connecting');
                
                // Create peer connection
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ]
                });

                // Add local audio track
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });

                // Handle remote audio
                pc.ontrack = event => {
                    log('🎵 Received remote audio track');
                    if (event.streams[0]) {
                        remoteAudio.srcObject = event.streams[0];
                        
                        remoteAudio.onloadeddata = () => {
                            log('📊 Audio data loaded, duration: ' + remoteAudio.duration + 's');
                        };
                        
                        remoteAudio.oncanplay = () => {
                            log('🔊 Audio ready to play');
                        };

                        remoteAudio.onplay = () => {
                            log('▶️ Audio playback started');
                            if (startTime) {
                                const latency = Date.now() - startTime;
                                updateMetrics(latency);
                                log(`⚡ Total latency: ${latency}ms`);
                            }
                            updateStatus('🤖 AI is speaking...', 'speaking');
                        };
                        
                        remoteAudio.onended = () => {
                            log('🔇 Audio playback ended');
                            if (isConnected) {
                                updateStatus('🎙️ Listening for your voice...', 'connected');
                            }
                        };
                        
                        remoteAudio.onerror = (e) => {
                            log(`❌ Audio error: ${e.target.error?.message || 'Unknown'}`);
                        };
                    }
                };

                // Handle connection state
                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    log(`🔗 Connection: ${state}`);
                    
                    if (state === 'connected') {
                        isConnected = true;
                        updateStatus('🎙️ Listening for your voice...', 'connected');
                        updateMetrics(undefined, 'Connected');
                        stopBtn.disabled = false;
                    } else if (state === 'failed' || state === 'disconnected') {
                        stop();
                    }
                };

                // ICE candidates
                pc.onicecandidate = event => {
                    if (event.candidate && ws?.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ice-candidate',
                            candidate: event.candidate.toJSON()
                        }));
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
                        log('✅ WebRTC handshake complete');
                    } else if (data.type === 'speech_detected') {
                        startTime = Date.now();
                        updateStatus('🧠 Processing your speech...', 'processing');
                    } else if (data.type === 'user_speech') {
                        addMessage(data.text, true);
                    } else if (data.type === 'ai_response') {
                        addMessage(data.text, false);
                    }
                };

                ws.onclose = () => {
                    log('🔌 WebSocket closed');
                    stop();
                };

                ws.onerror = () => {
                    log('❌ WebSocket error');
                    stop();
                };

            } catch (err) {
                log(`❌ Error: ${err.message}`);
                updateStatus(`❌ Error: ${err.message}`, 'disconnected');
                stop();
            }
        }

        function stop() {
            log('🛑 Stopping...');
            isConnected = false;
            
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
            
            if (remoteAudio.srcObject) {
                remoteAudio.srcObject = null;
            }
            
            updateStatus('🔌 Disconnected', 'disconnected');
            updateMetrics(0, 'Offline');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
        }

        log('🚀 Interface ready');
    </script>
</body>
</html>
"""

# --- Simplified ICE Candidate Parser ---
def parse_ice_candidate(candidate_str: str) -> dict:
    """Simple, reliable ICE candidate parser"""
    try:
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str[10:]
        
        parts = candidate_str.split()
        if len(parts) < 8:
            return {}
        
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
        return {}

# --- Improved VAD ---
class SimpleVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)
        self.silero_model = None
        self._load_silero()
        
    def _load_silero(self):
        try:
            logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD ready")
        except Exception as e:
            logger.error(f"❌ Silero VAD failed: {e}")
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Simple, reliable speech detection"""
        if len(audio) == 0:
            return False
            
        # Energy check
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            return False
            
        # Resample if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        # Use Silero if available, otherwise energy-based
        if self.silero_model:
            try:
                timestamps = self.get_speech_timestamps(
                    torch.from_numpy(audio), self.silero_model,
                    sampling_rate=16000, threshold=0.5
                )
                return len(timestamps) > 0
            except:
                pass
                
        return rms > 0.02

# --- Clean Audio Buffer ---
class CleanAudioBuffer:
    def __init__(self):
        self.sample_rate = 16000
        self.buffer_duration = 3.0  # 3 second buffer
        self.max_samples = int(self.buffer_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_duration = 0.8  # Minimum 0.8 seconds for processing
        self.min_samples = int(self.min_duration * self.sample_rate)
        self.last_process = 0
        self.cooldown = 1.5  # 1.5 second cooldown
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio with minimal processing"""
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Simple preprocessing
        audio_data = audio_data.flatten()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        self.buffer.extend(audio_data)
    
    def should_process(self, vad: SimpleVAD) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if we should process the audio"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_process < self.cooldown:
            return False, None
            
        # Length check
        if len(self.buffer) < self.min_samples:
            return False, None
            
        audio_array = np.array(list(self.buffer), dtype=np.float32)
        
        # VAD check
        if vad.detect_speech(audio_array):
            self.last_process = current_time
            return True, audio_array
            
        return False, None
    
    def reset(self):
        self.buffer.clear()

# --- Complete Audio Track (No Chunking) ---
class CompleteAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._complete_audio = None
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_samples = 960  # 20ms
        self._audio_ready = asyncio.Event()
        
    async def recv(self):
        """Send complete audio without fragmentation"""
        frame_data = np.zeros(self._frame_samples, dtype=np.int16)
        
        if self._complete_audio is not None and self._position < len(self._complete_audio):
            end_pos = min(self._position + self._frame_samples, len(self._complete_audio))
            chunk_size = end_pos - self._position
            frame_data[:chunk_size] = self._complete_audio[self._position:end_pos]
            self._position += chunk_size
        
        # Create frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame_data]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_samples
        
        return audio_frame
    
    async def set_complete_audio(self, audio_data: np.ndarray):
        """Set complete audio for playback"""
        if len(audio_data) > 0:
            # Convert to int16
            self._complete_audio = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            self._position = 0
            logger.info(f"🔊 Queued complete audio: {len(self._complete_audio)} samples, {len(self._complete_audio)/48000:.2f}s")
        else:
            self._complete_audio = None

# --- Simplified Audio Processor ---
class FixedAudioProcessor:
    def __init__(self, output_track, executor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = CleanAudioBuffer()
        self.vad = SimpleVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None
        self._stop_event = asyncio.Event()
        
    def set_websocket(self, ws):
        self.ws = ws
        
    def add_track(self, track):
        self.input_track = track
        logger.info("✅ Audio track added")
        
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
        """Simplified audio processing loop"""
        try:
            while not self._stop_event.is_set():
                if self.is_processing:
                    await asyncio.sleep(0.2)
                    continue
                
                try:
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                except:
                    break
                
                try:
                    # Simple audio extraction
                    audio_data = frame.to_ndarray().flatten()
                    
                    # Convert to float
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    # Resample if needed
                    if frame.sample_rate != 16000:
                        audio_float = librosa.resample(audio_float, orig_sr=frame.sample_rate, target_sr=16000)
                    
                    self.buffer.add_audio(audio_float)
                    
                    # Check for speech
                    should_process, audio_array = self.buffer.should_process(self.vad)
                    if should_process:
                        duration = len(audio_array) / 16000
                        logger.info(f"🎯 Processing speech: {duration:.2f}s")
                        self.buffer.reset()
                        asyncio.create_task(self._process_speech(audio_array))
                        
                except Exception as e:
                    logger.error(f"❌ Audio processing error: {e}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("🔚 Audio processor stopped")
    
    def _run_ultravox(self, audio_array: np.ndarray) -> str:
        """Clean Ultravox inference"""
        try:
            # Ensure audio quality
            if len(audio_array) < 8000:  # Less than 0.5 seconds
                return ""
                
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=30, do_sample=False, temperature=0.0)  # More deterministic
                
                # Extract clean text
                text = ""
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        text = item['generated_text']
                    elif isinstance(item, str):
                        text = item
                elif isinstance(result, str):
                    text = result
                
                # Clean up the text
                text = text.strip()
                
                # Remove common Ultravox artifacts
                if text.startswith("It seems like"):
                    # Try to extract the actual content
                    if '"' in text:
                        parts = text.split('"')
                        if len(parts) >= 3:
                            text = parts[1]
                
                return text[:100]  # Limit length
                
        except Exception as e:
            logger.error(f"❌ Ultravox error: {e}")
            return ""
    
    def _run_tts(self, text: str) -> np.ndarray:
        """Clean TTS generation"""
        try:
            if not text.strip() or len(text) < 2:
                return np.array([])
                
            # Limit text length to prevent long generations
            text = text[:200]
                
            with torch.inference_mode():
                wav = tts_model.generate(text)
                
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                elif torch.is_tensor(wav):
                    wav = wav.numpy()
                
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                
                # Normalize
                max_val = np.max(np.abs(wav_48k))
                if max_val > 0:
                    wav_48k = wav_48k / max_val * 0.8
                
                return wav_48k
                
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            return np.array([])
    
    async def _process_speech(self, audio_array: np.ndarray):
        """Clean speech processing"""
        if self.is_processing:
            return
            
        start_time = time.time()
        self.is_processing = True
        
        try:
            # Signal detection
            if self.ws and not self.ws.closed:
                await self.ws.send_json({'type': 'speech_detected'})
            
            # Run Ultravox
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(self.executor, self._run_ultravox, audio_array)
            
            if not user_text:
                logger.warning("⚠️ No text generated")
                return
                
            stt_time = time.time() - start_time
            logger.info(f"💬 User said: '{user_text}'")
            
            # Send to client
            if self.ws and not self.ws.closed:
                await self.ws.send_json({'type': 'user_speech', 'text': user_text})
            
            # Generate TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(self.executor, self._run_tts, user_text)
            
            if len(audio_output) > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                
                logger.info(f"⚡ STT: {stt_time*1000:.0f}ms, TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
                
                # Send AI response
                if self.ws and not self.ws.closed:
                    await self.ws.send_json({'type': 'ai_response', 'text': user_text})
                
                # Set complete audio (not chunks)
                await self.output_track.set_complete_audio(audio_output)
                
                # Wait for playback to complete
                duration = len(audio_output) / 48000
                logger.info(f"🔊 Playing {duration:.1f}s of audio")
                await asyncio.sleep(duration + 1.0)  # Extra buffer
            
        except Exception as e:
            logger.error(f"❌ Speech processing failed: {e}")
        finally:
            self.is_processing = False
            logger.info("✅ Speech processing complete")

# --- Model Initialization ---
def initialize_models() -> bool:
    """Initialize models with proper configuration"""
    global uv_pipe, tts_model
    
    try:
        logger.info("📥 Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("✅ Ultravox loaded")
        
        logger.info("📥 Loading TTS...")
        tts_model = ChatterboxTTS.from_pretrained()
        logger.info("✅ TTS loaded")
        
        # Simple warmup
        logger.info("🔥 Warming up...")
        dummy = np.random.randn(16000).astype(np.float32) * 0.001
        uv_pipe({'audio': dummy, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=5)
        tts_model.generate("test")
        logger.info("✅ Models ready")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

# --- WebSocket Handler ---
async def websocket_handler(request):
    """Simplified WebSocket handler"""
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
            response_track = CompleteAudioTrack()
            pc.addTrack(response_track)
            
            processor = FixedAudioProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"🔗 Connection: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            if processor:
                await processor.stop()
            if pc in pcs:
                pcs.remove(pc)
    
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
                            sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                        )
                        await pc.addIceCandidate(candidate)
                        
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        logger.info("🔚 WebSocket closed")
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        await pc.close()
    
    return ws

# --- HTTP Handlers ---
async def index_handler(request):
    return web.Response(text=HTML_CLIENT, content_type='text/html')

async def health_handler(request):
    return web.json_response({
        "status": "ok",
        "models_loaded": uv_pipe is not None and tts_model is not None,
        "connections": len(pcs)
    })

# --- Main Application ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down...")
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
    
    print("🚀 FIXED VOICE ASSISTANT READY!")
    print("📡 http://0.0.0.0:7860")
    print("🔧 Fixed: Speech recognition accuracy")
    print("🔧 Fixed: Complete TTS audio playback")  
    print("🔧 Fixed: Audio processing pipeline")
    print("🛑 Press Ctrl+C to stop\n")
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Stopped")
