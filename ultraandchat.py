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

# --- Setup ---
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for optimized event loop")
except ImportError:
    print("⚠️ uvloop not found, using default event loop")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence external loggers
for logger_name in ['aioice', 'aiortc', 'av']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# --- Global Variables ---
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_worker")
pcs = set()

# --- Enhanced HTML Client with Better Connection Handling ---
HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 Final Voice Assistant</title>
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
            text-align: center; max-width: 900px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
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
        .status.speaking { background: linear-gradient(45deg, #6f42c1, #007bff); animation: glow 1.5s infinite; }
        
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        @keyframes glow { 0%, 100% { box-shadow: 0 0 20px rgba(111, 66, 193, 0.5); } 50% { box-shadow: 0 0 30px rgba(111, 66, 193, 0.8); } }
        
        .conversation { 
            margin-top: 25px; padding: 25px; background: rgba(0,0,0,0.3); 
            border-radius: 15px; text-align: left; max-height: 400px; overflow-y: auto;
        }
        .message { margin: 20px 0; padding: 20px; border-radius: 12px; line-height: 1.5; }
        .user-msg { background: rgba(0, 123, 255, 0.3); margin-left: 30px; }
        .ai-msg { background: rgba(40, 167, 69, 0.3); margin-right: 30px; }
        .message strong { display: block; margin-bottom: 8px; font-size: 1.1em; }
        
        .connection-info {
            margin: 20px 0; padding: 20px; background: rgba(0,0,0,0.2);
            border-radius: 12px; text-align: left; font-family: 'Courier New', monospace; font-size: 12px;
        }
        .connection-info h3 { margin: 0 0 10px 0; font-family: inherit; }
        
        .audio-section {
            margin: 20px 0; padding: 20px; background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        .audio-section h3 { margin: 0 0 15px 0; color: #00ff88; }
        .audio-section audio { width: 100%; }
        
        .debug { 
            margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.2); 
            border-radius: 12px; font-family: 'Courier New', monospace; font-size: 12px;
            max-height: 200px; overflow-y: auto; text-align: left;
        }
        
        .tip {
            margin-top: 20px; padding: 15px; background: rgba(0,123,255,0.2);
            border-radius: 10px; font-size: 0.9em; border-left: 4px solid #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Final Voice Assistant</h1>
        
        <div class="controls">
            <button id="startBtn" onclick="start()">🎙️ Start Voice Chat</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>⏹️ Stop Chat</button>
        </div>
        
        <div id="status" class="status disconnected">🔌 Ready to Connect</div>
        
        <div class="connection-info">
            <h3>🔗 Connection Status</h3>
            <div>WebSocket: <span id="wsStatus">Disconnected</span></div>
            <div>WebRTC: <span id="rtcStatus">Disconnected</span></div>
            <div>ICE State: <span id="iceStatus">New</span></div>
            <div>Audio: <span id="audioStatus">Not Ready</span></div>
        </div>
        
        <div class="audio-section">
            <h3>🔊 AI Voice Response</h3>
            <audio id="remoteAudio" controls preload="auto"></audio>
        </div>
        
        <div id="conversation" class="conversation">
            <div style="text-align: center; opacity: 0.7; font-style: italic;">
                Voice conversation will appear here...
            </div>
        </div>
        
        <div class="tip">
            💡 <strong>Runpod Optimized:</strong> This version uses TURN servers and enhanced connection handling for container environments.
        </div>
        
        <div id="debug" class="debug">Initializing...</div>
    </div>

    <script>
        let pc, ws, localStream, startTime, connectionTimeout;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const debugDiv = document.getElementById('debug');
        const conversationDiv = document.getElementById('conversation');
        const wsStatus = document.getElementById('wsStatus');
        const rtcStatus = document.getElementById('rtcStatus');
        const iceStatus = document.getElementById('iceStatus');
        const audioStatus = document.getElementById('audioStatus');

        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            console.log(message);
            debugDiv.innerHTML += `${timestamp}: ${message}<br>`;
            debugDiv.scrollTop = debugDiv.scrollHeight;
            
            const lines = debugDiv.innerHTML.split('<br>');
            if (lines.length > 25) {
                debugDiv.innerHTML = lines.slice(-20).join('<br>');
            }
        }

        function updateConnectionInfo(ws_state, rtc_state, ice_state, audio_state) {
            if (ws_state !== undefined) wsStatus.textContent = ws_state;
            if (rtc_state !== undefined) rtcStatus.textContent = rtc_state;
            if (ice_state !== undefined) iceStatus.textContent = ice_state;
            if (audio_state !== undefined) audioStatus.textContent = audio_state;
        }

        function addMessage(text, isUser = false) {
            if (conversationDiv.innerHTML.includes('conversation will appear here')) {
                conversationDiv.innerHTML = '';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-msg' : 'ai-msg'}`;
            messageDiv.innerHTML = `<strong>${isUser ? '👤 You:' : '🤖 AI:'}</strong> ${text}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        function updateStatus(message, className) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${className}`;
            log(`Status: ${message}`);
        }

        async function start() {
            startBtn.disabled = true;
            updateStatus('🔄 Initializing...', 'connecting');
            debugDiv.innerHTML = '';
            
            // Set connection timeout
            connectionTimeout = setTimeout(() => {
                log('❌ Connection timeout (30s)');
                updateStatus('❌ Connection timeout - retrying...', 'disconnected');
                stop();
                setTimeout(() => start(), 2000);
            }, 30000);
            
            try {
                log('🎤 Requesting microphone...');
                updateConnectionInfo('Connecting', 'Initializing', 'New', 'Requesting');
                
                const constraints = {
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: { ideal: 48000, min: 16000 },
                        channelCount: 1
                    }
                };

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                const settings = localStream.getAudioTracks()[0].getSettings();
                log(`✅ Microphone: ${settings.sampleRate}Hz, Echo: ${settings.echoCancellation}`);
                updateConnectionInfo(undefined, undefined, undefined, 'Ready');

                // Enhanced WebRTC configuration with TURN servers
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' },
                        { urls: 'stun:stun.cloudflare.com:3478' },
                        { urls: 'stun:stun.relay.metered.ca:80' },
                        // Public TURN servers as fallback
                        {
                            urls: 'turn:openrelay.metered.ca:80',
                            username: 'openrelayproject',
                            credential: 'openrelayproject'
                        },
                        {
                            urls: 'turn:openrelay.metered.ca:443',
                            username: 'openrelayproject', 
                            credential: 'openrelayproject'
                        }
                    ],
                    iceCandidatePoolSize: 10,
                    bundlePolicy: 'max-bundle',
                    rtcpMuxPolicy: 'require'
                });

                log('🔗 RTCPeerConnection created with TURN servers');

                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });

                // Enhanced event handlers
                pc.ontrack = event => {
                    log('🎵 Remote audio track received');
                    if (event.streams[0]) {
                        remoteAudio.srcObject = event.streams[0];
                        updateConnectionInfo(undefined, undefined, undefined, 'Connected');
                        
                        remoteAudio.onplay = () => {
                            log('▶️ Audio playback started');
                            if (startTime) {
                                const latency = Date.now() - startTime;
                                log(`⚡ Total latency: ${latency}ms`);
                            }
                            updateStatus('🤖 AI is speaking...', 'speaking');
                        };
                        
                        remoteAudio.onended = () => {
                            log('🔇 Audio ended');
                            updateStatus('🎙️ Listening...', 'connected');
                        };
                    }
                };

                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    log(`🔗 Connection: ${state}`);
                    updateConnectionInfo(undefined, state, undefined, undefined);
                    
                    if (state === 'connected') {
                        clearTimeout(connectionTimeout);
                        updateStatus('🎙️ Ready - Speak now!', 'connected');
                        stopBtn.disabled = false;
                        log('🎉 Connection established successfully!');
                    } else if (state === 'failed') {
                        log('❌ Connection failed - retrying...');
                        stop();
                        setTimeout(() => start(), 3000);
                    } else if (state === 'disconnected') {
                        log('🔌 Connection lost');
                        stop();
                    }
                };

                pc.oniceconnectionstatechange = () => {
                    const state = pc.iceConnectionState;
                    log(`🧊 ICE: ${state}`);
                    updateConnectionInfo(undefined, undefined, state, undefined);
                    
                    if (state === 'failed') {
                        log('❌ ICE connection failed');
                        stop();
                        setTimeout(() => start(), 3000);
                    }
                };

                pc.onicecandidate = event => {
                    if (event.candidate) {
                        log(`📤 ICE candidate: ${event.candidate.type} (${event.candidate.protocol})`);
                        if (ws?.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate.toJSON()
                            }));
                        }
                    }
                };

                // WebSocket with retry logic
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws`);

                ws.onopen = async () => {
                    log('🌐 WebSocket connected');
                    updateConnectionInfo('Connected', undefined, undefined, undefined);
                    
                    try {
                        const offer = await pc.createOffer({
                            offerToReceiveAudio: true,
                            offerToReceiveVideo: false
                        });
                        await pc.setLocalDescription(offer);
                        ws.send(JSON.stringify(offer));
                        log('📤 Offer sent');
                    } catch (err) {
                        log(`❌ Offer failed: ${err.message}`);
                        throw err;
                    }
                };

                ws.onmessage = async event => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'answer') {
                        log('📥 Answer received');
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (data.type === 'speech_detected') {
                        startTime = Date.now();
                        updateStatus('🧠 Processing speech...', 'processing');
                    } else if (data.type === 'user_speech') {
                        addMessage(data.text, true);
                    } else if (data.type === 'ai_response') {
                        addMessage(data.text, false);
                    }
                };

                ws.onclose = (event) => {
                    log(`🔌 WebSocket closed: ${event.code}`);
                    updateConnectionInfo('Disconnected', undefined, undefined, undefined);
                    if (!startBtn.disabled) return; // Already stopping
                    
                    // Auto-retry connection
                    setTimeout(() => {
                        if (startBtn.disabled) start();
                    }, 5000);
                };

                ws.onerror = () => {
                    log('❌ WebSocket error');
                    updateConnectionInfo('Error', undefined, undefined, undefined);
                };

            } catch (err) {
                clearTimeout(connectionTimeout);
                log(`❌ Error: ${err.message}`);
                updateStatus(`❌ ${err.message}`, 'disconnected');
                stop();
            }
        }

        function stop() {
            clearTimeout(connectionTimeout);
            log('🛑 Stopping...');
            
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
            updateConnectionInfo('Disconnected', 'Closed', 'New', 'Not Ready');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
        }

        log('🚀 Enhanced Voice Assistant Ready');
    </script>
</body>
</html>
"""

# --- Enhanced ICE Candidate Processing ---
def parse_ice_candidate(candidate_str: str) -> dict:
    """Robust ICE candidate parser with enhanced error handling"""
    try:
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str[10:]
        
        parts = candidate_str.strip().split()
        if len(parts) < 8:
            logger.warning(f"Insufficient ICE candidate parts: {len(parts)}")
            return {}
        
        candidate_info = {
            'foundation': parts[0],
            'component': int(parts[1]),
            'protocol': parts[2].lower(),
            'priority': int(parts[3]),
            'ip': parts[4],
            'port': int(parts[5]),
            'type': parts[7].lower()
        }
        
        # Parse additional attributes
        i = 8
        while i < len(parts) - 1:
            key = parts[i].lower()
            if key == "raddr" and i + 1 < len(parts):
                candidate_info['relatedAddress'] = parts[i + 1]
                i += 2
            elif key == "rport" and i + 1 < len(parts):
                try:
                    candidate_info['relatedPort'] = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif key == "tcptype" and i + 1 < len(parts):
                candidate_info['tcpType'] = parts[i + 1]
                i += 2
            else:
                i += 1
                
        return candidate_info
        
    except (ValueError, IndexError) as e:
        logger.error(f"ICE candidate parsing error: {e}")
        return {}

# --- Simplified, Reliable VAD ---
class OptimizedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)
        self.silero_model = None
        self._load_silero()
        
    def _load_silero(self):
        try:
            logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', 
                force_reload=False, verbose=False
            )
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.error(f"❌ Silero VAD error: {e}")
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Optimized speech detection"""
        if len(audio) == 0:
            return False
            
        # Energy-based quick filter
        energy = np.sqrt(np.mean(audio ** 2))
        if energy < 0.005:
            return False
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        # Use Silero for accuracy
        if self.silero_model:
            try:
                timestamps = self.get_speech_timestamps(
                    torch.from_numpy(audio), 
                    self.silero_model,
                    sampling_rate=16000,
                    threshold=0.5,
                    min_speech_duration_ms=300
                )
                return len(timestamps) > 0
            except Exception:
                pass
                
        # Fallback to energy + basic filtering
        return energy > 0.015

# --- Optimized Audio Buffer ---
class OptimizedAudioBuffer:
    def __init__(self):
        self.sample_rate = 16000
        self.buffer_duration = 4.0  # 4 second buffer
        self.max_samples = int(self.buffer_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_duration = 1.0  # Minimum 1 second
        self.min_samples = int(self.min_duration * self.sample_rate)
        self.last_process = 0
        self.cooldown = 2.0  # 2 second cooldown between processing
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio with minimal processing"""
        # Convert to float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Basic preprocessing
        audio_data = audio_data.flatten()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Remove DC bias
        audio_data = audio_data - np.mean(audio_data)
        
        self.buffer.extend(audio_data)
    
    def should_process(self, vad: OptimizedVAD) -> Tuple[bool, Optional[np.ndarray]]:
        """Determine if audio should be processed"""
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
        """Clear the buffer"""
        self.buffer.clear()

# --- Optimized Audio Track ---
class OptimizedAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_data = None
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_samples = 960  # 20ms frames
        self._audio_lock = asyncio.Lock()
        
    async def recv(self):
        """Generate audio frames for WebRTC"""
        frame_data = np.zeros(self._frame_samples, dtype=np.int16)
        
        async with self._audio_lock:
            if (self._audio_data is not None and 
                self._position < len(self._audio_data)):
                
                end_pos = min(self._position + self._frame_samples, len(self._audio_data))
                chunk_size = end_pos - self._position
                
                if chunk_size > 0:
                    frame_data[:chunk_size] = self._audio_data[self._position:end_pos]
                    self._position += chunk_size
        
        # Create AV frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame_data]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_samples
        
        return audio_frame
    
    async def set_audio_data(self, audio_data: np.ndarray):
        """Set complete audio data for playback"""
        async with self._audio_lock:
            if len(audio_data) > 0:
                # Convert to int16 with proper scaling
                audio_scaled = np.clip(audio_data * 32767, -32768, 32767)
                self._audio_data = audio_scaled.astype(np.int16)
                self._position = 0
                
                duration = len(self._audio_data) / self._sample_rate
                logger.info(f"🔊 Audio set: {duration:.2f}s, {len(self._audio_data)} samples")
            else:
                self._audio_data = None
                logger.warning("⚠️ Empty audio data provided")

# --- Enhanced Audio Processor ---
class EnhancedAudioProcessor:
    def __init__(self, output_track, executor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = OptimizedAudioBuffer()
        self.vad = OptimizedVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None
        self._stop_event = asyncio.Event()
        
    def set_websocket(self, ws):
        self.ws = ws
        
    def add_track(self, track):
        self.input_track = track
        logger.info("✅ Audio track connected")
        
    async def start(self):
        if not self.task:
            logger.info("🎵 Starting enhanced audio processor")
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
        """Main audio processing loop with enhanced error handling"""
        frame_count = 0
        
        try:
            while not self._stop_event.is_set():
                # Skip processing while AI is speaking
                if self.is_processing:
                    await asyncio.sleep(0.2)
                    continue
                
                try:
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.5)
                    frame_count += 1
                    
                    # Log progress periodically
                    if frame_count % 100 == 0:
                        logger.info(f"📊 Processed {frame_count} frames")
                    
                except asyncio.TimeoutError:
                    continue
                except mediastreams.MediaStreamError:
                    logger.info("🔚 Media stream ended")
                    break
                except Exception as e:
                    logger.error(f"❌ Frame error: {e}")
                    continue
                
                try:
                    # Process audio frame
                    audio_data = frame.to_ndarray().flatten()
                    
                    # Convert to float32
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    # Resample if needed
                    if frame.sample_rate != 16000:
                        audio_float = librosa.resample(
                            audio_float,
                            orig_sr=frame.sample_rate,
                            target_sr=16000
                        )
                    
                    # Add to buffer
                    self.buffer.add_audio(audio_float)
                    
                    # Check for speech
                    should_process, audio_array = self.buffer.should_process(self.vad)
                    if should_process and audio_array is not None:
                        duration = len(audio_array) / 16000
                        logger.info(f"🎯 Speech detected: {duration:.2f}s")
                        
                        # Clear buffer and process
                        self.buffer.reset()
                        asyncio.create_task(self._process_speech(audio_array))
                        
                except Exception as e:
                    logger.error(f"❌ Audio processing error: {e}")
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ Audio loop error: {e}")
        finally:
            logger.info("🔚 Audio processor stopped")
    
    def _run_ultravox(self, audio_array: np.ndarray) -> str:
        """Optimized Ultravox inference"""
        try:
            # Validate input
            if len(audio_array) < 8000:  # Less than 0.5 seconds
                logger.warning("⚠️ Audio too short for processing")
                return ""
                
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=30, do_sample=False, temperature=0.1)
                
                # Extract text
                text = ""
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        text = item['generated_text']
                    elif isinstance(item, str):
                        text = item
                elif isinstance(result, str):
                    text = result
                
                # Clean text
                text = text.strip()
                
                # Remove common Ultravox artifacts
                if text.startswith(("It seems like", "I think you", "You appear to")):
                    # Extract quoted content if present
                    if '"' in text:
                        quoted_parts = text.split('"')
                        if len(quoted_parts) >= 3:
                            text = quoted_parts[1].strip()
                
                # Limit length
                return text[:120] if text else ""
                
        except Exception as e:
            logger.error(f"❌ Ultravox error: {e}")
            return ""
    
    def _run_tts(self, text: str) -> np.ndarray:
        """Optimized TTS generation"""
        try:
            if not text.strip() or len(text) < 2:
                logger.warning("⚠️ Text too short for TTS")
                return np.array([])
                
            # Limit text length
            text = text[:100]
                
            with torch.inference_mode():
                wav = tts_model.generate(text)
                
                # Convert to numpy
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                elif torch.is_tensor(wav):
                    wav = wav.numpy()
                
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz for WebRTC
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                
                # Normalize audio
                max_val = np.max(np.abs(wav_48k))
                if max_val > 0:
                    wav_48k = wav_48k / max_val * 0.8  # 80% volume
                
                return wav_48k
                
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            return np.array([])
    
    async def _process_speech(self, audio_array: np.ndarray):
        """Enhanced speech processing with better error handling"""
        if self.is_processing:
            logger.warning("⚠️ Already processing, skipping")
            return
            
        start_time = time.time()
        self.is_processing = True
        
        try:
            # Signal processing start
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'speech_detected'})
                except Exception as e:
                    logger.debug(f"WebSocket send error: {e}")
            
            # Run Ultravox inference
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(
                self.executor, self._run_ultravox, audio_array
            )
            
            if not user_text:
                logger.warning("⚠️ No text generated from speech")
                return
                
            stt_time = time.time() - start_time
            logger.info(f"💬 User: '{user_text}' (STT: {stt_time*1000:.0f}ms)")
            
            # Send user speech to client
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'user_speech', 'text': user_text})
                except Exception as e:
                    logger.debug(f"WebSocket send error: {e}")
            
            # Generate TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(
                self.executor, self._run_tts, user_text
            )
            
            if len(audio_output) > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                
                logger.info(f"⚡ TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
                
                # Send AI response to client
                if self.ws and not self.ws.closed:
                    try:
                        await self.ws.send_json({'type': 'ai_response', 'text': user_text})
                    except Exception as e:
                        logger.debug(f"WebSocket send error: {e}")
                
                # Set audio for playback
                await self.output_track.set_audio_data(audio_output)
                
                # Wait for audio to complete
                playback_duration = len(audio_output) / 48000
                logger.info(f"🎵 Playing {playback_duration:.1f}s audio")
                await asyncio.sleep(playback_duration + 2.0)  # Extra buffer
            else:
                logger.warning("⚠️ No audio generated from TTS")
            
        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}")
        finally:
            self.is_processing = False
            logger.info("✅ Speech processing complete")

# --- Model Initialization ---
def initialize_models() -> bool:
    """Initialize models with comprehensive error handling"""
    global uv_pipe, tts_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    
    try:
        # Load Ultravox
        logger.info("📥 Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("✅ Ultravox loaded successfully")
        
        # Load ChatterboxTTS
        logger.info("📥 Loading ChatterboxTTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ ChatterboxTTS loaded successfully")
        
        # Warmup models
        logger.info("🔥 Warming up models...")
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.001
        
        with torch.inference_mode():
            # Warmup Ultravox
            try:
                uv_pipe({
                    'audio': dummy_audio,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=5)
                logger.info("✅ Ultravox warmed up")
            except Exception as e:
                logger.warning(f"⚠️ Ultravox warmup issue: {e}")
            
            # Warmup TTS
            try:
                tts_model.generate("Hello")
                logger.info("✅ TTS warmed up")
            except Exception as e:
                logger.warning(f"⚠️ TTS warmup issue: {e}")
        
        logger.info("🎉 All models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- Enhanced WebSocket Handler ---
async def websocket_handler(request):
    """Enhanced WebSocket handler with better connection management"""
    ws = web.WebSocketResponse(
        heartbeat=30,
        timeout=120,
        max_msg_size=16*1024*1024
    )
    await ws.prepare(request)
    
    logger.info("🌐 WebSocket connection established")
    
    # Enhanced WebRTC configuration
    config = RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        RTCIceServer(urls="stun:stun.cloudflare.com:3478"),
        RTCIceServer(urls="stun:stun.relay.metered.ca:80"),
        # TURN servers for NAT traversal
        RTCIceServer(
            urls="turn:openrelay.metered.ca:80",
            username="openrelayproject",
            credential="openrelayproject"
        ),
        RTCIceServer(
            urls="turn:openrelay.metered.ca:443",
            username="openrelayproject", 
            credential="openrelayproject"
        )
    ])
    
    pc = RTCPeerConnection(config)
    pcs.add(pc)
    processor = None
    
    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 Track received: {track.kind}")
        
        if track.kind == "audio":
            # Create optimized audio track for response
            response_track = OptimizedAudioTrack()
            pc.addTrack(response_track)
            
            # Create enhanced processor
            processor = EnhancedAudioProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            
            # Start processing
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info(f"🔗 Connection state: {state}")
        
        if state in ["failed", "closed", "disconnected"]:
            logger.info("🧹 Cleaning up failed connection")
            if processor:
                await processor.stop()
            if pc in pcs:
                pcs.remove(pc)
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        state = pc.iceConnectionState
        logger.info(f"🧊 ICE connection state: {state}")
        
        if state == "failed":
            logger.warning("❌ ICE connection failed - connection will retry")
        elif state == "connected":
            logger.info("✅ ICE connection established")
        elif state == "disconnected":
            logger.info("🔌 ICE connection lost")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "offer":
                        logger.info("📥 Processing WebRTC offer")
                        
                        # Set remote description
                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                        )
                        
                        # Create and send answer
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        await ws.send_json({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        })
                        logger.info("📤 WebRTC answer sent")
                        
                    elif msg_type == "ice-candidate":
                        candidate_data = data.get("candidate", {})
                        if candidate_data:
                            candidate_str = candidate_data.get("candidate", "")
                            parsed = parse_ice_candidate(candidate_str)
                            
                            if parsed:
                                try:
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
                                    logger.debug(f"✅ ICE candidate added: {parsed['type']}")
                                    
                                except Exception as e:
                                    logger.error(f"❌ ICE candidate error: {e}")
                            else:
                                logger.warning("⚠️ Failed to parse ICE candidate")
                                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"❌ Message processing error: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error: {ws.exception()}")
                break
                
    except ConnectionResetError:
        logger.info("🔌 Client disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}")
    finally:
        logger.info("🔚 Closing WebSocket connection")
        
        # Cleanup
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

# --- HTTP Handlers ---
async def index_handler(request):
    """Serve main application page"""
    return web.Response(
        text=HTML_CLIENT,
        content_type='text/html',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "models": {
            "ultravox_loaded": uv_pipe is not None,
            "tts_loaded": tts_model is not None
        },
        "connections": len(pcs),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    })

# --- Application Setup ---
async def on_shutdown(app):
    """Graceful shutdown handler"""
    logger.info("🛑 Initiating shutdown...")
    
    # Close all peer connections
    close_tasks = []
    for pc in list(pcs):
        close_tasks.append(pc.close())
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    pcs.clear()
    
    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("✅ Shutdown completed")

async def main():
    """Main application entry point"""
    
    # Initialize models first
    if not initialize_models():
        logger.error("❌ Failed to initialize models - cannot start server")
        return
    
    # Create web application
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 7860)
    await site.start()
    
    print("\n" + "="*60)
    print("🚀 FINAL RUNPOD-OPTIMIZED VOICE ASSISTANT")
    print("="*60)
    print(f"📡 URL: http://0.0.0.0:7860")
    print(f"🏗️  Platform: Runpod Container")
    print(f"🔗 WebRTC: Enhanced with TURN servers")
    print(f"🧠 GPU: {'✅ Available' if torch.cuda.is_available() else '❌ CPU Only'}")
    print(f"🎤 VAD: Silero + WebRTC hybrid")
    print(f"🔊 TTS: ChatterboxTTS optimized")
    print(f"⚡ Target: <500ms latency")
    print("="*60)
    print("🔧 Key Features:")
    print("   • TURN servers for NAT traversal")
    print("   • Enhanced ICE candidate processing")
    print("   • Connection retry mechanisms")
    print("   • Optimized for container networking")
    print("   • Complete audio playback")
    print("   • Accurate speech recognition")
    print("="*60)
    print("🛑 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Keep server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown initiated...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("✅ Server stopped successfully")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print("❌ Server encountered a fatal error")
