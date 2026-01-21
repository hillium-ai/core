# loqus_core/aura/ears.py

import threading
import queue
import numpy as np
import time
from typing import Optional, Tuple, Callable

# Import HippoLink - this should be available in the actual system
try:
    from loqus_backend.hippo_link import HippoLink
except ImportError:
    # Fallback to a mock or handle as needed for testing
    class HippoLink:
        @staticmethod
        def get_instance():
            return HippoLink()
        def write_conversation(self, text):
            print(f"HippoLink: {text}")

try:
    import pyaudio
    import torch
    # from silero_vad import VADIterator  # Note: silero-vad might have versioning issues
    # import faster_whisper
except ImportError as e:
    print(f"Missing required dependencies: {e}")

class VADFilter:
    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self.vad_model = None
        
    def load_model(self):
        # Implementation depends on silero-vad version
        pass
            
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        # Placeholder for VAD logic
        return np.mean(np.abs(audio_chunk)) > self.sensitivity


class STTEngine:
    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = None
        
    def load_model(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            
    def transcribe(self, audio_tensor: np.ndarray) -> str:
        if self.model is None:
            return ""
        
        try:
            segments, info = self.model.transcribe(audio_tensor, beam_size=5)
            return " ".join([segment.text for segment in segments])
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""


class ListenerThread(threading.Thread):
    def __init__(self, vad_filter: VADFilter, stt_engine: STTEngine, hippo_link_writer: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.vad_filter = vad_filter
        self.stt_engine = stt_engine
        self.hippo_link_writer = hippo_link_writer
        self.daemon = True
        self.is_running = False
        self.chunk_size = 1024
        self.sample_rate = 16000
        
    def run(self):
        self.is_running = True
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            p.terminate()
            return
        
        audio_buffer = []
        recording = False
        
        try:
            while self.is_running:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                is_speech = self.vad_filter.is_speech(audio_chunk)
                
                if is_speech and not recording:
                    recording = True
                    audio_buffer = [audio_chunk]
                elif not is_speech and recording:
                    recording = False
                    if audio_buffer:
                        audio_tensor = np.concatenate(audio_buffer)
                        text = self.stt_engine.transcribe(audio_tensor)
                        if text and self.hippo_link_writer:
                            self.hippo_link_writer(text)
                elif recording:
                    audio_buffer.append(audio_chunk)
                    
                if len(audio_buffer) > 200:
                    audio_buffer = audio_buffer[-100:]
                    
        except Exception as e:
            print(f"Error in ListenerThread: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    def stop(self):
        self.is_running = False


def start_ears_pipeline(hippo_link_writer: Callable[[str], None]) -> ListenerThread:
    vad_filter = VADFilter(sensitivity=0.1)
    stt_engine = STTEngine(model_size="tiny", device="cpu")
    stt_engine.load_model()
    listener = ListenerThread(vad_filter, stt_engine, hippo_link_writer)
    listener.start()
    return listener
