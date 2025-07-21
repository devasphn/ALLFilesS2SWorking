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

# Silence noisy loggers
for logger_name in ['aioice.ice', 'aiortc.rtcpeerconnection', 'av.audio.resampler']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio_worker")
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
            text-align: center; max-width: 900px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
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
        button:disabled { 
            background: linear-gradient(45deg, #6c757d, #495057); cursor: not-allowed; 
            transform: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .stop-btn { background: linear-gradient(45deg, #dc3545, #c82333); }
        .stop-btn:hover { background: linear-gradient(45deg, #c82333, #a71e2a); }
        
        .status { 
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.1em;
            transition: all 0.5s ease;
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
            animation: pulse 2s infinite;
        }
        .status.speaking { 
            background: linear-gradient(45deg, #007bff, #6610f2);
            animation: speaking 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes speaking {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.03); }
        }
        
        .conversation { 
            margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.3); 
            border-radius: 15px; text-align: left; max-height: 350px; overflow-y: auto;
        }
        .message { margin: 15px 0; padding: 15px; border-radius: 10px; }
        .user-msg { background: rgba(0, 123, 255, 0.3); margin-left: 20px; }
        .ai-msg { background: rgba(40, 167, 69, 0.3); margin-right: 20px; }
        
        .metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin: 20px 0;
        }
        .metric { 
            padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; text-align: center;
        }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 0.9em; opacity: 0.8; margin-top: 5px; }
        
        .debug { 
            margin-top: 15px; padding: 15px; background: rgba(0,0,0,0.2); 
            border-radius: 10px; font-family: 'Courier New', monospace; font-size: 11px;
            max-height: 150px; overflow-y: auto; text-align: left;
        }
        
        .audio-visualizer {
            margin: 20px 0; height: 60px; background: rgba(0,0,0,0.3);
            border-radius: 10px; position: relative; overflow: hidden;
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
        
        <div class="metrics">
            <div class="metric">
                <div id="latencyValue" class="metric-value">0ms</div>
                <div class="metric-label">Response Time</div>
            </div>
            <div class="metric">
                <div id="connectionValue" class="metric-value">Offline</div>
                <div class="metric-label">Connection</div>
            </div>
            <div class="metric">
                <div id="qualityValue" class="metric-value">-</div>
                <div class="metric-label">Audio Quality</div>
            </div>
        </div>
        
        <div class="audio-visualizer" id="visualizer"></div>
        <div id="conversation" class="conversation"></div>
        <div id="debug" class="debug">System ready. Click Start to begin...</div>
        
        <audio id="remoteAudio" autoplay playsinline style="width: 100%; margin: 10px 0;"></audio>
    </div>

    <script>
        let pc, ws, localStream, startTime, audioContext, analyser;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const debugDiv = document.getElementById('debug');
        const conversationDiv = document.getElementById('conversation');
        const latencyValue = document.getElementById('latencyValue');
        const connectionValue = document.getElementById('connectionValue');
        const qualityValue = document.getElementById('qualityValue');

        function log(message) {
            console.log(message);
            debugDiv.innerHTML += new Date().toLocaleTimeString() + ': ' + message + '<br>';
            debugDiv.scrollTop = debugDiv.scrollHeight;
        }

        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-msg' : 'ai-msg'}`;
            messageDiv.innerHTML = `<strong>${isUser ? '👤 You' : '🤖 AI'}:</strong> ${text}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        function updateStatus(message, className) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${className}`;
            log(`Status: ${message}`);
        }

        function updateMetrics(latency, connection, quality) {
            if (latency !== undefined) latencyValue.textContent = `${latency}ms`;
            if (connection !== undefined) connectionValue.textContent = connection;
            if (quality !== undefined) qualityValue.textContent = quality;
        }

        async function start() {
            startBtn.disabled = true;
            updateStatus('🔄 Initializing...', 'connecting');
            debugDiv.innerHTML = '';
            conversationDiv.innerHTML = '';
            
            try {
                log('🎤 Requesting microphone access...');
                
                // Enhanced audio constraints for better quality
                const constraints = {
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: { ideal: 48000 },
                        channelCount: 1,
                        latency: { ideal: 0.01 }
                    }
                };

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                log(`✅ Microphone access granted`);
                
                // Setup audio analysis
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ 
                    sampleRate: 48000,
                    latencyHint: 'interactive'
                });
                
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }
                
                const audioTrack = localStream.getAudioTracks()[0];
                const settings = audioTrack.getSettings();
                log(`Audio: ${settings.sampleRate}Hz, ${settings.channelCount}ch`);
                updateMetrics(undefined, 'Initializing', 'High');

                // Create peer connection with optimal settings
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ],
                    iceCandidatePoolSize: 10
                });

                log('🔗 RTCPeerConnection created');

                // Add tracks with better handling
                localStream.getTracks().forEach((track, index) => {
                    log(`📤 Adding ${track.kind} track`);
                    const sender = pc.addTrack(track, localStream);
                    
                    // Set encoding parameters for better audio
                    if (track.kind === 'audio') {
                        const params = sender.getParameters();
                        if (params.encodings && params.encodings.length > 0) {
                            params.encodings[0].maxBitrate = 128000; // 128 kbps
                        }
                        sender.setParameters(params).catch(e => log(`Encoding error: ${e}`));
                    }
                });

                // Handle remote audio with enhanced processing
                pc.ontrack = event => {
                    log(`🎵 Remote track received: ${event.track.kind}`);
                    if (event.streams && event.streams[0]) {
                        remoteAudio.srcObject = event.streams[0];
                        
                        remoteAudio.onloadstart = () => log('Audio loading started');
                        remoteAudio.oncanplay = () => {
                            log('Audio can play');
                            remoteAudio.play().catch(err => {
                                log(`❌ Autoplay failed: ${err.message}`);
                            });
                        };

                        remoteAudio.onplaying = () => {
                            log('🔊 Audio playing');
                            if (startTime) {
                                const latency = Date.now() - startTime;
                                updateMetrics(latency, 'Connected', 'Excellent');
                            }
                            updateStatus('🤖 AI Speaking...', 'speaking');
                        };
                        
                        remoteAudio.onended = () => {
                            log('🔇 Audio ended');
                            if (pc && pc.connectionState === 'connected') {
                                updateStatus('🎙️ Listening...', 'connected');
                            }
                        };
                        
                        remoteAudio.onerror = (err) => {
                            log(`❌ Audio error: ${err}`);
                        };
                        
                        remoteAudio.onvolumechange = () => {
                            log(`🔊 Volume: ${remoteAudio.volume}`);
                        };
                    }
                };

                // ICE candidate handling
                pc.onicecandidate = event => {
                    if (event.candidate) {
                        log(`📤 ICE candidate: ${event.candidate.type}`);
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate.toJSON()
                            }));
                        }
                    } else {
                        log('✅ ICE gathering complete');
                    }
                };

                // Connection state monitoring
                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    log(`🔗 Connection state: ${state}`);
                    
                    if (state === 'connected') {
                        updateStatus('🎙️ Listening...', 'connected');
                        updateMetrics(undefined, 'Connected', 'Excellent');
                        stopBtn.disabled = false;
                    } else if (state === 'connecting') {
                        updateStatus('🤝 Connecting...', 'connecting');
                        updateMetrics(undefined, 'Connecting', 'Good');
                    } else if (['failed', 'closed', 'disconnected'].includes(state)) {
                        log(`❌ Connection ${state}`);
                        updateMetrics(undefined, 'Disconnected', 'Poor');
                        stop();
                    }
                };

                pc.oniceconnectionstatechange = () => {
                    log(`🧊 ICE state: ${pc.iceConnectionState}`);
                };

                // WebSocket connection
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${location.host}/ws`;
                log(`🌐 Connecting to: ${wsUrl}`);
                
                ws = new WebSocket(wsUrl);

                ws.onopen = async () => {
                    log('✅ WebSocket connected');
                    try {
                        updateStatus('📋 Creating offer...', 'connecting');
                        const offer = await pc.createOffer({
                            offerToReceiveAudio: true,
                            offerToReceiveVideo: false
                        });
                        
                        await pc.setLocalDescription(offer);
                        log('📤 Sending offer');
                        ws.send(JSON.stringify(offer));
                        
                    } catch (err) {
                        log(`❌ Offer error: ${err.message}`);
                        throw err;
                    }
                };

                ws.onmessage = async event => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'answer') {
                            log('📥 Received answer');
                            await pc.setRemoteDescription(new RTCSessionDescription(data));
                        } else if (data.type === 'speech_start') {
                            startTime = Date.now();
                            updateStatus('🧠 Processing...', 'connecting');
                        } else if (data.type === 'user_speech') {
                            addMessage(data.text, true);
                        } else if (data.type === 'ai_response') {
                            addMessage(data.text, false);
                        }
                    } catch (err) {
                        log(`❌ Message error: ${err.message}`);
                    }
                };

                ws.onclose = (event) => {
                    log(`🔌 WebSocket closed: ${event.code}`);
                    updateMetrics(undefined, 'Disconnected', 'Poor');
                    if (pc && !['closed', 'failed'].includes(pc.connectionState)) {
                        stop();
                    }
                };

                ws.onerror = (error) => {
                    log(`❌ WebSocket error`);
                    updateMetrics(undefined, 'Error', 'Poor');
                    if (pc && !['closed', 'failed'].includes(pc.connectionState)) {
                        stop();
                    }
                };

            } catch (err) {
                log(`❌ Initialization error: ${err.message}`);
                console.error('Full error:', err);
                updateStatus(`❌ Error: ${err.message}`, 'disconnected');
                updateMetrics(undefined, 'Error', 'Poor');
                stop();
            }
        }

        function stop() {
            log('🛑 Stopping connection...');
            
            // Clean up WebSocket
            if (ws) {
                ws.onclose = ws.onerror = ws.onmessage = null;
                if (ws.readyState !== WebSocket.CLOSED) {
                    ws.close();
                }
                ws = null;
            }
            
            // Clean up peer connection
            if (pc) {
                pc.onconnectionstatechange = null;
                pc.onicecandidate = null;
                pc.ontrack = null;
                if (pc.connectionState !== 'closed') {
                    pc.close();
                }
                pc = null;
            }
            
            // Clean up media
            if (localStream) {
                localStream.getTracks().forEach(track => {
                    log(`⏹️ Stopping ${track.kind} track`);
                    track.stop();
                });
                localStream = null;
            }
            
            if (remoteAudio.srcObject) {
                remoteAudio.srcObject = null;
            }
            
            if (audioContext && audioContext.state !== 'closed') {
                audioContext.close();
                audioContext = null;
            }
            
            updateStatus('🔌 Disconnected', 'disconnected');
            updateMetrics(0, 'Disconnected', '-');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
        }

        // Prevent accidental page unload
        window.addEventListener('beforeunload', stop);
        
        // Auto-focus and keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ' && e.ctrlKey) {
                e.preventDefault();
                if (startBtn.disabled) stop(); else start();
            }
        });
        
        log('🚀 Interface loaded. Press Ctrl+Space to toggle recording.');
    </script>
</body>
</html>
"""

# --- Utility Functions ---
def parse_ice_candidate(candidate_string: str) -> dict:
    """Parse ICE candidate string into components"""
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
            'type': parts[7]  # parts[6] is 'typ'
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
        logger.debug(f"ICE candidate parse error: {e}")
        return {}

# --- Enhanced VAD System ---
class AdvancedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)  # Balanced sensitivity
        self.silero_model = None
        self.load_silero()
        
    def load_silero(self):
        try:
            logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load(
                'snakers4/silero-vad', 'silero_vad', force_reload=False
            )
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"❌ Silero VAD loading error: {e}")
            self.silero_model = None
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Enhanced speech detection with multiple methods"""
        if len(audio) == 0:
            return False
            
        # Energy threshold check
        energy = np.mean(audio ** 2)
        if energy < 0.0008:  # Very quiet threshold
            return False
            
        # Ensure 16kHz for processing
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        # WebRTC VAD check
        webrtc_result = False
        try:
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            
            # Process in 20ms chunks
            chunk_size = 320  # 20ms at 16kHz
            speech_count = 0
            total_chunks = 0
            
            for i in range(0, len(audio_int16) - chunk_size + 1, chunk_size):
                chunk = audio_int16[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    try:
                        if self.webrtc_vad.is_speech(chunk.tobytes(), 16000):
                            speech_count += 1
                    except:
                        pass
                    total_chunks += 1
            
            if total_chunks > 0:
                speech_ratio = speech_count / total_chunks
                webrtc_result = speech_ratio > 0.35  # 35% speech threshold
                
        except Exception as e:
            logger.debug(f"WebRTC VAD error: {e}")
            webrtc_result = energy > 0.015
        
        # Silero VAD check for higher accuracy
        silero_result = True
        if self.silero_model is not None:
            try:
                audio_tensor = torch.from_numpy(audio)
                speech_timestamps = self.get_speech_timestamps(
                    audio_tensor, 
                    self.silero_model, 
                    sampling_rate=16000,
                    min_speech_duration_ms=300,
                    threshold=0.3
                )
                silero_result = len(speech_timestamps) > 0
            except Exception as e:
                logger.debug(f"Silero VAD error: {e}")
                silero_result = webrtc_result
        
        # Combined decision (both must agree for high confidence)
        final_result = webrtc_result and silero_result
        logger.debug(f"VAD Decision: energy={energy:.6f}, webrtc={webrtc_result}, silero={silero_result}, final={final_result}")
        
        return final_result

# --- Enhanced Audio Buffer ---
class HighQualityAudioBuffer:
    def __init__(self):
        self.sample_rate = 16000
        self.max_duration = 5.0  # Longer buffer for better context
        self.max_samples = int(self.max_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_speech_duration = 1.2  # Longer minimum for clearer speech
        self.min_samples = int(self.min_speech_duration * self.sample_rate)
        self.last_process_time = 0
        self.cooldown = 0.8  # Reasonable cooldown
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio with enhanced preprocessing"""
        # Ensure float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Enhanced preprocessing
        audio_data = audio_data.flatten()
        
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Gentle normalization (avoid clipping)
        max_val = max(np.max(np.abs(audio_data)), 0.01)
        if max_val > 0.8:  # Only normalize if too loud
            audio_data = audio_data * (0.8 / max_val)
        
        # Simple high-pass filter to reduce noise
        if len(audio_data) > 1:
            audio_data[1:] = audio_data[1:] - 0.95 * audio_data[:-1]
        
        self.buffer.extend(audio_data)
    
    def should_process(self, vad: AdvancedVAD) -> Tuple[bool, Optional[np.ndarray]]:
        """Determine if audio should be processed"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_process_time < self.cooldown:
            return False, None
            
        # Length check
        if len(self.buffer) < self.min_samples:
            return False, None
            
        # Get audio array
        audio_array = np.array(list(self.buffer), dtype=np.float32)
        
        # Energy check
        if np.max(np.abs(audio_array)) < 0.015:
            return False, None
            
        # VAD check
        if vad.detect_speech(audio_array, self.sample_rate):
            self.last_process_time = current_time
            return True, audio_array
            
        return False, None
    
    def reset(self):
        """Clear the buffer"""
        self.buffer.clear()

# --- High-Quality Audio Track ---
class HighQualityAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_queue = asyncio.Queue(maxsize=20)
        self._current_audio = None
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_size = 960  # 20ms at 48kHz
        
    async def recv(self):
        """Receive audio frame"""
        # Get new audio if current is finished
        if self._current_audio is None or self._position >= len(self._current_audio):
            try:
                self._current_audio = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.02
                )
                self._position = 0
            except asyncio.TimeoutError:
                self._current_audio = None
        
        # Create frame
        frame_data = np.zeros(self._frame_size, dtype=np.int16)
        
        # Fill frame with audio data
        if self._current_audio is not None:
            remaining = len(self._current_audio) - self._position
            copy_size = min(self._frame_size, remaining)
            if copy_size > 0:
                frame_data[:copy_size] = self._current_audio[self._position:self._position + copy_size]
                self._position += copy_size
        
        # Create AV frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame_data]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_size
        
        return audio_frame
    
    async def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the queue"""
        if len(audio_data) > 0:
            # Convert to int16 with proper clipping
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            try:
                await asyncio.wait_for(self._audio_queue.put(audio_int16), timeout=0.15)
            except asyncio.TimeoutError:
                logger.warning("⚠️ Audio queue full, dropping audio")

# --- Advanced Audio Processor ---
class AdvancedAudioProcessor:
    def __init__(self, output_track, executor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = HighQualityAudioBuffer()
        self.vad = AdvancedVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None
        self._stop_event = asyncio.Event()
        self.frame_count = 0
        
    def set_websocket(self, ws):
        """Set WebSocket for communication"""
        self.ws = ws
        
    def add_track(self, track):
        """Add input audio track"""
        self.input_track = track
        logger.info(f"✅ Audio track added: {track.kind}")
        
    async def start(self):
        """Start audio processing"""
        if not self.task:
            logger.info("🎵 Starting enhanced audio processor")
            self.task = asyncio.create_task(self._audio_loop())
            
    async def stop(self):
        """Stop audio processing"""
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
        """Main audio processing loop"""
        try:
            while not self._stop_event.is_set():
                # Skip processing while AI is speaking
                if self.is_processing:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)
                    self.frame_count += 1
                    
                    # Log every 200 frames (~4 seconds)
                    if self.frame_count % 200 == 0:
                        logger.info(f"📊 Processed {self.frame_count} audio frames")
                    
                except asyncio.TimeoutError:
                    continue
                except mediastreams.MediaStreamError:
                    logger.info("🔚 Audio stream ended")
                    break
                except Exception as e:
                    logger.error(f"❌ Frame receive error: {e}")
                    break
                
                try:
                    # Extract and process audio
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
                        logger.info(f"🎯 Speech detected! Processing {len(audio_array)/16000:.2f}s of audio")
                        self.buffer.reset()
                        
                        # Process asynchronously
                        asyncio.create_task(self._process_speech(audio_array))
                        
                except Exception as e:
                    logger.error(f"❌ Audio processing error: {e}")
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ Audio loop error: {e}", exc_info=True)
        finally:
            logger.info("🔚 Audio processor stopped")
    
    def _run_inference(self, audio_array: np.ndarray) -> str:
        """Run speech-to-text inference"""
        try:
            with torch.inference_mode():
                # Create input for Ultravox
                input_data = {
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }
                
                # Run inference with conservative settings
                result = uv_pipe(
                    input_data, 
                    max_new_tokens=60,
                    do_sample=False,
                    temperature=0.1
                )
                
                # Extract text from result
                text = ""
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        text = item['generated_text']
                    elif isinstance(item, str):
                        text = item
                elif isinstance(result, str):
                    text = result
                
                return text.strip() if text else ""
                
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            return ""
    
    def _run_tts(self, text: str) -> np.ndarray:
        """Generate text-to-speech audio"""
        try:
            if not text.strip():
                return np.array([], dtype=np.float32)
                
            with torch.inference_mode():
                # Generate TTS audio
                wav = tts_model.generate(text)
                
                # Convert to numpy if needed
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                elif torch.is_tensor(wav):
                    wav = wav.numpy()
                
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz for WebRTC
                wav_48k = librosa.resample(
                    wav, 
                    orig_sr=24000,  # ChatterboxTTS default sample rate
                    target_sr=48000
                )
                
                # Ensure reasonable volume
                if np.max(np.abs(wav_48k)) > 0:
                    wav_48k = wav_48k / max(np.max(np.abs(wav_48k)), 0.1) * 0.8
                
                return wav_48k
                
        except Exception as e:
            logger.error(f"❌ TTS generation error: {e}")
            return np.array([], dtype=np.float32)
    
    async def _process_speech(self, audio_array: np.ndarray):
        """Process detected speech"""
        if self.is_processing:
            return
            
        processing_start = time.time()
        self.is_processing = True
        
        try:
            # Signal processing start
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'speech_start'})
                except:
                    pass
            
            # Run speech-to-text
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(
                self.executor, 
                self._run_inference, 
                audio_array
            )
            
            if not user_text:
                logger.warning("⚠️ No text generated from speech")
                return
                
            stt_time = time.time() - processing_start
            logger.info(f"💬 User said: '{user_text}'")
            
            # Send user speech to client
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'user_speech', 'text': user_text})
                except:
                    pass
            
            # Generate TTS response
            tts_start = time.time()
            audio_output = await loop.run_in_executor(
                self.executor,
                self._run_tts,
                user_text
            )
            
            if audio_output.size > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - processing_start
                
                logger.info(f"⚡ Performance - STT: {stt_time*1000:.0f}ms, TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
                
                # Send AI response to client
                if self.ws and not self.ws.closed:
                    try:
                        await self.ws.send_json({'type': 'ai_response', 'text': user_text})
                    except:
                        pass
                
                # Queue complete audio for playback
                await self.output_track.add_audio(audio_output)
                
                # Wait for audio to finish playing
                playback_duration = len(audio_output) / 48000
                await asyncio.sleep(playback_duration + 0.5)
            else:
                logger.warning("⚠️ No audio generated from TTS")
            
        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}", exc_info=True)
        finally:
            self.is_processing = False

# --- CORRECTED Model Initialization ---
def initialize_models() -> bool:
    """Initialize Ultravox and TTS models"""
    global uv_pipe, tts_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on device: {device}")
    
    try:
        # CORRECT Ultravox loading - no task specification, no subscripts
        logger.info("📥 Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Apply torch.compile optimization if available
        if hasattr(torch, 'compile') and hasattr(uv_pipe, 'model'):
            try:
                uv_pipe.model = torch.compile(uv_pipe.model, mode="reduce-overhead")
                logger.info("✅ Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")
        
        logger.info("✅ Ultravox pipeline loaded successfully")
        
        # Load TTS model
        logger.info("📥 Loading ChatterboxTTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ ChatterboxTTS loaded successfully")
        
        # Warmup both models
        logger.info("🔥 Warming up models...")
        dummy_audio = np.random.randn(8000).astype(np.float32) * 0.01
        
        with torch.inference_mode():
            # Warmup Ultravox
            try:
                warmup_result = uv_pipe({
                    'audio': dummy_audio, 
                    'turns': [], 
                    'sampling_rate': 16000
                }, max_new_tokens=5)
                logger.info("✅ Ultravox warmed up")
            except Exception as e:
                logger.warning(f"⚠️ Ultravox warmup failed: {e}")
            
            # Warmup TTS
            try:
                tts_model.generate("Hello world")
                logger.info("✅ TTS warmed up")
            except Exception as e:
                logger.warning(f"⚠️ TTS warmup failed: {e}")
            
        logger.info("🎉 All models initialized and ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- WebSocket Handler ---
async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse(heartbeat=30, timeout=120)
    await ws.prepare(request)
    
    logger.info("🌐 New WebSocket connection established")
    
    # Create peer connection with optimal configuration
    configuration = RTCConfiguration([
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302")
    ])
    
    pc = RTCPeerConnection(configuration)
    pcs.add(pc)
    processor = None
    
    @pc.on("track")
    def on_track(track):
        nonlocal processor
        logger.info(f"🎧 Track received: {track.kind}")
        
        if track.kind == "audio":
            # Create high-quality response track
            response_track = HighQualityAudioTrack()
            pc.addTrack(response_track)
            
            # Create advanced audio processor
            processor = AdvancedAudioProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            
            # Start processing
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info(f"🔗 Connection state changed to: {state}")
        
        if state in ["failed", "closed", "disconnected"]:
            logger.info("🧹 Cleaning up connection")
            if processor:
                await processor.stop()
            if pc in pcs:
                pcs.remove(pc)
    
    @pc.on("icecandidateerror")
    def on_icecandidateerror(event):
        logger.error(f"❌ ICE candidate error: {event}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data["type"] == "offer":
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
                        
                    elif data["type"] == "ice-candidate":
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
                                    logger.error(f"❌ Failed to add ICE candidate: {e}")
                            else:
                                logger.warning("⚠️ Failed to parse ICE candidate")
                                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"❌ Message processing error: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error: {ws.exception()}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("🔚 WebSocket connection closing")
        
        # Clean up
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

# --- HTTP Route Handlers ---
async def index_handler(request):
    """Serve the main HTML interface"""
    return web.Response(
        text=HTML_CLIENT, 
        content_type='text/html',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Content-Type-Options': 'nosniff'
        }
    )

async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "models_loaded": {
            "ultravox": uv_pipe is not None,
            "tts": tts_model is not None
        },
        "active_connections": len(pcs),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    })

# --- Application Setup ---
async def on_shutdown(app):
    """Graceful shutdown handler"""
    logger.info("🛑 Initiating graceful shutdown...")
    
    # Close all peer connections
    shutdown_tasks = []
    for pc in list(pcs):
        shutdown_tasks.append(pc.close())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    pcs.clear()
    
    # Shutdown thread pool
    executor.shutdown(wait=True)
    logger.info("✅ Graceful shutdown completed")

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
    print("🚀 UltraFast Speech-to-Speech Server Started!")
    print("="*60)
    print(f"📡 Server URL: http://0.0.0.0:7860")
    print(f"💨 Target Latency: <500ms")
    print(f"🎯 Enhanced Audio Quality & TTS")
    print(f"🧠 GPU Acceleration: {'✅ Enabled' if torch.cuda.is_available() else '❌ Disabled'}")
    print(f"🎤 Advanced VAD: ✅ Enabled")
    print(f"🔊 Full TTS Playback: ✅ Enhanced")
    print("="*60)
    print("🛑 Press Ctrl+C to stop the server")
    print("💡 Use Ctrl+Space in browser to toggle recording")
    print("="*60 + "\n")
    
    try:
        # Keep the server running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("✅ Server stopped gracefully")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print("❌ Server encountered an error")
