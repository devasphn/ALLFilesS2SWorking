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
import torchaudio
import soundfile as sf

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
for logger_name in ['aioice.ice', 'aiortc', 'av.audio.resampler']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# --- Global Variables ---
uv_pipe, tts_model, vad_models = None, None, {}
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio_worker")
pcs = set()

# --- Ultra-Low Latency HTML Client ---
HTML_CLIENT = """
<!DOCTYPE html>
<html>
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
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            text-align: center; max-width: 600px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
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
        
        .latency { 
            margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); 
            border-radius: 10px; font-family: 'Courier New', monospace;
        }
        .metrics { display: flex; justify-content: space-around; margin-top: 15px; }
        .metric { text-align: center; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 0.9em; opacity: 0.8; }
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
        <div id="latency" class="latency" style="display:none;">
            <div class="metrics">
                <div class="metric">
                    <div id="latencyValue" class="metric-value">0ms</div>
                    <div class="metric-label">Response Time</div>
                </div>
                <div class="metric">
                    <div id="qualityValue" class="metric-value">High</div>
                    <div class="metric-label">Audio Quality</div>
                </div>
            </div>
        </div>
        <audio id="remoteAudio" autoplay playsinline></audio>
    </div>

    <script>
        let pc, ws, localStream, startTime;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const latencyDiv = document.getElementById('latency');
        const latencyValue = document.getElementById('latencyValue');

        function updateStatus(message, className) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${className}`;
        }

        function updateLatency(ms) {
            latencyValue.textContent = `${ms}ms`;
            latencyDiv.style.display = 'block';
        }

        async function start() {
            startBtn.disabled = true;
            updateStatus('🔄 Initializing...', 'connecting');
            
            try {
                // Enhanced audio constraints for better quality
                const constraints = {
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 48000,
                        channelCount: 1,
                        volume: 1.0
                    }
                };

                const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 48000,
                    latencyHint: 'interactive'
                });
                
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                
                // Enhanced RTC configuration
                pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        { urls: 'stun:stun1.l.google.com:19302' }
                    ],
                    iceCandidatePoolSize: 10
                });

                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });

                pc.ontrack = event => {
                    console.log('🎵 Remote audio track received');
                    if (remoteAudio.srcObject !== event.streams[0]) {
                        remoteAudio.srcObject = event.streams[0];
                        remoteAudio.play().catch(err => console.error("Autoplay failed:", err));

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

                pc.onicecandidate = event => {
                    if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ice-candidate',
                            candidate: event.candidate.toJSON()
                        }));
                    }
                };

                pc.onconnectionstatechange = () => {
                    const state = pc.connectionState;
                    console.log(`🔗 Connection state: ${state}`);
                    
                    if (state === 'connecting') {
                        updateStatus('🤝 Connecting...', 'connecting');
                    } else if (state === 'connected') {
                        updateStatus('🎙️ Listening...', 'connected');
                        stopBtn.disabled = false;
                    } else if (['failed', 'closed', 'disconnected'].includes(state)) {
                        stop();
                    }
                };

                // Enhanced WebSocket connection
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${location.host}/ws`);

                ws.onopen = async () => {
                    console.log('🌐 WebSocket connected');
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    ws.send(JSON.stringify(offer));
                };

                ws.onmessage = async event => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'answer' && !pc.currentRemoteDescription) {
                        await pc.setRemoteDescription(new RTCSessionDescription(data));
                    } else if (data.type === 'speech_start') {
                        startTime = Date.now();
                        updateStatus('🧠 Processing...', 'connecting');
                    }
                };

                const closeHandler = () => {
                    if (pc && pc.connectionState !== 'closed') stop();
                };
                
                ws.onclose = closeHandler;
                ws.onerror = closeHandler;

            } catch (err) {
                console.error('❌ Initialization error:', err);
                updateStatus(`❌ Error: ${err.message}`, 'disconnected');
                stop();
            }
        }

        function stop() {
            console.log('🛑 Stopping connection...');
            
            if (ws) {
                ws.onclose = null;
                ws.onerror = null;
                ws.close();
                ws = null;
            }
            
            if (pc) {
                pc.onconnectionstatechange = null;
                pc.onicecandidate = null;
                pc.ontrack = null;
                pc.close();
                pc = null;
            }
            
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            updateStatus('🔌 Disconnected', 'disconnected');
            latencyDiv.style.display = 'none';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
        }

        // Prevent page unload without cleanup
        window.addEventListener('beforeunload', stop);
    </script>
</body>
</html>
"""

# --- Enhanced VAD System ---
class MultiVAD:
    def __init__(self):
        self.silero_model = None
        self.webrtc_vad = webrtcvad.Vad(3)  # Most aggressive mode
        self.load_silero()
        
    def load_silero(self):
        try:
            logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.error(f"❌ Silero VAD failed: {e}")
            self.silero_model = None
    
    def detect_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        if len(audio_data) == 0:
            return False
            
        # Energy-based detection (fast)
        energy = np.mean(audio_data ** 2)
        if energy < 0.001:  # Very quiet
            return False
            
        # WebRTC VAD (fast, reliable)
        try:
            if sample_rate != 16000:
                audio_16k = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_16k = audio_data
                
            # Convert to int16 for WebRTC VAD
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Check in chunks
            chunk_size = 320  # 20ms at 16kHz
            chunks = [audio_int16[i:i+chunk_size] for i in range(0, len(audio_int16), chunk_size)]
            
            speech_chunks = 0
            for chunk in chunks:
                if len(chunk) == chunk_size:
                    try:
                        if self.webrtc_vad.is_speech(chunk.tobytes(), 16000):
                            speech_chunks += 1
                    except:
                        pass
                        
            webrtc_result = speech_chunks > len(chunks) * 0.3
            
        except Exception:
            webrtc_result = energy > 0.01
        
        # Silero VAD (more accurate, slower)
        silero_result = True
        if self.silero_model is not None:
            try:
                audio_tensor = torch.from_numpy(audio_16k if sample_rate != 16000 else audio_data)
                timestamps = self.get_speech_timestamps(
                    audio_tensor, 
                    self.silero_model, 
                    sampling_rate=16000,
                    min_speech_duration_ms=200,
                    threshold=0.5
                )
                silero_result = len(timestamps) > 0
            except Exception:
                silero_result = webrtc_result
        
        # Combine results (both must agree for high confidence)
        return webrtc_result and silero_result

# --- Optimized Audio Processing ---
class OptimizedAudioBuffer:
    def __init__(self, max_duration: float = 2.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_speech_duration = 0.3  # seconds
        self.min_samples = int(self.min_speech_duration * sample_rate)
        self.last_process_time = 0
        self.process_cooldown = 0.2  # seconds
        
    def add_audio(self, audio_data: np.ndarray) -> None:
        # Ensure float32 and normalize
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Clip to prevent overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        self.buffer.extend(audio_data.flatten())
    
    def get_audio_array(self) -> np.ndarray:
        return np.array(list(self.buffer), dtype=np.float32)
    
    def should_process(self, vad: MultiVAD) -> Tuple[bool, Optional[np.ndarray]]:
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_process_time < self.process_cooldown:
            return False, None
            
        # Check minimum length
        if len(self.buffer) < self.min_samples:
            return False, None
            
        audio_array = self.get_audio_array()
        
        # Quick energy check
        if np.max(np.abs(audio_array)) < 0.01:
            return False, None
            
        # VAD check
        if not vad.detect_speech(audio_array, self.sample_rate):
            return False, None
            
        self.last_process_time = current_time
        return True, audio_array
    
    def reset(self) -> None:
        self.buffer.clear()

# --- High-Performance Audio Track ---
class OptimizedResponseTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=50)  # Prevent memory buildup
        self._current_chunk = None
        self._chunk_position = 0
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 48000)
        self._sample_rate = 48000
        self._frame_samples = 960  # 20ms at 48kHz
        
    async def recv(self):
        frame = np.zeros(self._frame_samples, dtype=np.int16)
        
        # Fill frame from current chunk or get new chunk
        if self._current_chunk is None or self._chunk_position >= len(self._current_chunk):
            try:
                self._current_chunk = await asyncio.wait_for(self._queue.get(), timeout=0.01)
                self._chunk_position = 0
                self._queue.task_done()
            except asyncio.TimeoutError:
                pass  # Send silence
        
        if self._current_chunk is not None and self._chunk_position < len(self._current_chunk):
            end_pos = min(self._chunk_position + self._frame_samples, len(self._current_chunk))
            chunk_data = self._current_chunk[self._chunk_position:end_pos]
            frame[:len(chunk_data)] = chunk_data
            self._chunk_position = end_pos
        
        # Create frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_samples
        
        return audio_frame
    
    async def queue_audio(self, audio_float32: np.ndarray) -> None:
        if audio_float32.size > 0:
            # Convert to int16 and add to queue
            audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
            try:
                await asyncio.wait_for(self._queue.put(audio_int16), timeout=0.1)
            except asyncio.TimeoutError:
                logger.warning("Audio queue full, dropping audio")

# --- Enhanced Audio Processor ---
class UltraFastProcessor:
    def __init__(self, output_track: OptimizedResponseTrack, executor: ThreadPoolExecutor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = OptimizedAudioBuffer()
        self.vad = MultiVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None  # For sending events
        
    def set_websocket(self, ws):
        self.ws = ws
        
    def add_track(self, track):
        self.input_track = track
        
    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._audio_loop())
            
    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
    
    async def _audio_loop(self):
        try:
            while True:
                # Skip processing if AI is currently speaking
                if self.is_processing:
                    try:
                        # Drain audio to prevent buildup
                        await asyncio.wait_for(self.input_track.recv(), timeout=0.01)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.01)
                    continue
                
                try:
                    frame = await self.input_track.recv()
                except mediastreams.MediaStreamError:
                    logger.info("🔚 Audio stream ended")
                    break
                
                # Convert frame to numpy array
                audio_data = frame.to_ndarray().flatten()
                
                # Resample if needed
                if frame.sample_rate != 16000:
                    audio_data = librosa.resample(
                        audio_data.astype(np.float32) / 32768.0 if audio_data.dtype == np.int16 else audio_data,
                        orig_sr=frame.sample_rate,
                        target_sr=16000
                    )
                else:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Add to buffer
                self.buffer.add_audio(audio_data)
                
                # Check if we should process
                should_process, audio_array = self.buffer.should_process(self.vad)
                if should_process and audio_array is not None:
                    self.buffer.reset()
                    logger.info(f"🎯 Processing {len(audio_array)/16000:.2f}s of audio")
                    
                    # Signal processing start
                    if self.ws:
                        await self._safe_ws_send({'type': 'speech_start'})
                    
                    # Process asynchronously
                    asyncio.create_task(self._process_speech(audio_array))
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ Audio loop error: {e}", exc_info=True)
        finally:
            logger.info("🔚 Audio processor stopped")
    
    async def _safe_ws_send(self, data):
        try:
            if self.ws and not self.ws.closed:
                await self.ws.send_json(data)
        except Exception:
            pass
    
    def _blocking_inference(self, audio_array: np.ndarray) -> str:
        """Run model inference in thread pool"""
        try:
            with torch.inference_mode():
                # Prepare input
                input_data = {
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }
                
                # Run inference
                result = uv_pipe(input_data, max_new_tokens=50, do_sample=True, temperature=0.7)
                
                # Parse result
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        return item['generated_text'].strip()
                    elif isinstance(item, str):
                        return item.strip()
                elif isinstance(result, str):
                    return result.strip()
                    
                return ""
                
        except Exception as e:
            logger.error(f"❌ Inference error: {e}", exc_info=True)
            return ""
    
    def _blocking_tts(self, text: str) -> np.ndarray:
        """Generate TTS in thread pool"""
        try:
            if not text.strip():
                return np.array([], dtype=np.float32)
                
            with torch.inference_mode():
                # Generate audio
                wav = tts_model.generate(text).cpu().numpy().flatten()
                
                # Resample to 48kHz for WebRTC
                wav_48k = librosa.resample(
                    wav.astype(np.float32),
                    orig_sr=24000,
                    target_sr=48000
                )
                
                return wav_48k
                
        except Exception as e:
            logger.error(f"❌ TTS error: {e}", exc_info=True)
            return np.array([], dtype=np.float32)
    
    async def _process_speech(self, audio_array: np.ndarray):
        """Process speech with optimized pipeline"""
        processing_start = time.time()
        self.is_processing = True
        
        try:
            # Run inference in thread pool
            loop = asyncio.get_running_loop()
            
            # Speech-to-text
            text_result = await loop.run_in_executor(
                self.executor, 
                self._blocking_inference, 
                audio_array
            )
            
            if not text_result:
                logger.warning("⚠️ No text generated")
                return
                
            logger.info(f"💬 Generated: '{text_result}'")
            inference_time = time.time() - processing_start
            
            # Text-to-speech
            tts_start = time.time()
            audio_output = await loop.run_in_executor(
                self.executor,
                self._blocking_tts,
                text_result
            )
            
            if audio_output.size == 0:
                logger.warning("⚠️ No audio generated")
                return
                
            tts_time = time.time() - tts_start
            total_time = time.time() - processing_start
            
            logger.info(f"⚡ Timings - Inference: {inference_time*1000:.0f}ms, TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
            
            # Queue audio for playback
            await self.output_track.queue_audio(audio_output)
            
            # Calculate playback duration and wait
            playback_duration = len(audio_output) / 48000
            await asyncio.sleep(playback_duration + 0.1)  # Small buffer
            
        except Exception as e:
            logger.error(f"❌ Speech processing error: {e}", exc_info=True)
        finally:
            self.is_processing = False

# --- Model Initialization ---
def initialize_models() -> bool:
    global uv_pipe, tts_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing models on {device}")
    
    try:
        # Load Ultravox with optimizations
        logger.info("📥 Loading Ultravox...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            use_fast=True
        )
        
        # Optimize with torch.compile if available
        if hasattr(torch, 'compile'):
            try:
                uv_pipe.model = torch.compile(uv_pipe.model, mode="reduce-overhead")
                logger.info("✅ Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")
        
        logger.info("✅ Ultravox loaded successfully")
        
        # Load TTS
        logger.info("📥 Loading Chatterbox TTS...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("✅ TTS loaded successfully")
        
        # Warmup models
        logger.info("🔥 Warming up models...")
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
        
        with torch.inference_mode():
            # Warmup Ultravox
            uv_pipe({'audio': dummy_audio, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=5)
            # Warmup TTS
            tts_model.generate("Hello")
            
        logger.info("🎉 All models ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- WebSocket Handler ---
async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30, timeout=60)
    await ws.prepare(request)
    
    # Enhanced RTC configuration
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
        logger.info(f"🎧 Received {track.kind} track")
        
        if track.kind == "audio":
            # Create optimized response track
            response_track = OptimizedResponseTrack()
            pc.addTrack(response_track)
            
            # Create processor
            processor = UltraFastProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            
            # Start processing
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info(f"🔗 Connection state: {state}")
        
        if state in ["failed", "closed", "disconnected"]:
            if pc in pcs:
                pcs.remove(pc)
            await pc.close()
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data["type"] == "offer":
                        # Set remote description
                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                        )
                        
                        # Create answer
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        # Send answer
                        await ws.send_json({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        })
                        
                    elif data["type"] == "ice-candidate" and data.get("candidate"):
                        try:
                            candidate_data = data["candidate"]
                            if candidate_string := candidate_data.get("candidate"):
                                # Parse ICE candidate
                                if candidate_string.startswith("candidate:"):
                                    candidate_string = candidate_string[10:]
                                
                                parts = candidate_string.split()
                                if len(parts) >= 8:
                                    candidate = RTCIceCandidate(
                                        component=int(parts[1]),
                                        foundation=parts[0],
                                        ip=parts[4],
                                        port=int(parts[5]),
                                        priority=int(parts[3]),
                                        protocol=parts[2],
                                        type=parts[7],
                                        sdpMid=candidate_data.get("sdpMid"),
                                        sdpMLineIndex=candidate_data.get("sdpMLineIndex")
                                    )
                                    await pc.addIceCandidate(candidate)
                        except Exception as e:
                            logger.error(f"❌ ICE candidate error: {e}")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error: {ws.exception()}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error: {e}", exc_info=True)
    finally:
        logger.info("🔚 WebSocket connection closed")
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
        "status": "healthy",
        "models_loaded": uv_pipe is not None and tts_model is not None,
        "active_connections": len(pcs)
    })

# --- Application Setup ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down...")
    
    # Close all peer connections
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    
    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("✅ Shutdown complete")

async def main():
    # Initialize models
    if not initialize_models():
        logger.error("❌ Failed to initialize models")
        return
    
    # Create application
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
    
    print("🚀 UltraFast Speech-to-Speech Server Started!")
    print("📡 Server: http://0.0.0.0:7860")
    print("💨 Optimized for <500ms latency")
    print("🎯 Ready for real-time conversations!")
    print("\n🛑 Press Ctrl+C to stop")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
