import os
import subprocess
import tempfile
import time
import html
import logging
from typing import Optional
from contextlib import contextmanager

import pyaudio
from loqus_core.aura.hippolink import HippoLink

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to HippoLink instance
hippo = HippoLink.get_instance()

# Piper-TTS binary path (adjust as needed)
PiperTTS_BINARY = "piper"

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

MAX_TEXT_LENGTH = 1000  # Prevent DoS attacks


class TTSEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._validate_piper()

    def _validate_piper(self):
        """Check if Piper-TTS binary is available."""
        try:
            subprocess.run([PiperTTS_BINARY, "--help"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"Piper-TTS binary not found or not executable: {PiperTTS_BINARY}")
            # Try to find alternative TTS engine
            if not self._try_fallback_tts():
                raise RuntimeError(f"No suitable TTS engine found. Required: {PiperTTS_BINARY}")

    def _try_fallback_tts(self) -> bool:
        """Try to use alternative TTS engine (e.g., espeak)."""
        try:
            subprocess.run(["espeak", "--version"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=True)
            logger.info("Fallback TTS engine (espeak) is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("No fallback TTS engine available")
            return False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for TTS processing."""
        # Basic normalization: strip whitespace and replace problematic characters
        text = text.strip()
        # Replace common problematic sequences
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        # HTML escape to prevent injection
        text = html.escape(text)
        return text

    def _validate_input(self, text: str) -> bool:
        """Validate input text for safety and length."""
        if not isinstance(text, str):
            logger.warning("Input text is not a string")
            return False
        
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text exceeds maximum length of {MAX_TEXT_LENGTH}")
            return False
        
        if not text.strip():
            logger.info("Empty text provided")
            return False
        
        return True

    @contextmanager
    def _speaking_context(self):
        """Context manager for handling speaking state."""
        try:
            hippo.set_is_speaking(True)
            yield
        finally:
            hippo.set_is_speaking(False)

    def speak(self, text: str) -> bool:
        """Generate speech from text and play it."""
        # Validate input
        if not self._validate_input(text):
            return False

        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Set speaking flag using context manager
        with self._speaking_context():
            try:
                # Create temporary file for audio output
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_filename = tmp_file.name

                # Generate audio using Piper-TTS
                cmd = [PiperTTS_BINARY]
                if self.model_path:
                    cmd.extend(["--model", self.model_path])
                cmd.extend(["--output", tmp_filename])

                # Use subprocess to run Piper-TTS
                result = subprocess.run(
                    cmd,
                    input=normalized_text,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=30  # 30 second timeout
                )

                # Play audio
                self._play_audio(tmp_filename)
                
                logger.info("TTS playback completed successfully")
                return True

            except subprocess.TimeoutExpired:
                logger.error("TTS generation timed out")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(f"Error generating audio with Piper-TTS: {e}")
                logger.error(f"Error output: {e.stderr}")
                # Try fallback TTS engine
                return self._fallback_tts(normalized_text)
            except Exception as e:
                logger.error(f"Unexpected error during TTS or playback: {e}")
                return False
            finally:
                # Clean up temporary file
                try:
                    if 'tmp_filename' in locals():
                        os.unlink(tmp_filename)
                except OSError as e:
                    logger.error(f"Error cleaning up temporary file: {e}")

    def _fallback_tts(self, text: str) -> bool:
        """Fallback to alternative TTS engine."""
        try:
            logger.info("Attempting fallback TTS with espeak")
            cmd = ["espeak", "-w", "/tmp/fallback.wav", text]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30
            )
            self._play_audio("/tmp/fallback.wav")
            return True
        except Exception as e:
            logger.error(f"Fallback TTS also failed: {e}")
            return False

    def _play_audio(self, filename: str) -> None:
        """Play audio file using pyaudio."""
        try:
            # Check if file exists
            if not os.path.exists(filename):
                logger.error(f"Audio file does not exist: {filename}")
                return
            
            # Open audio file
            with open(filename, "rb") as f:
                # Initialize PyAudio
                p = pyaudio.PyAudio()

                # Open stream
                stream = p.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               output=True)

                # Read and play audio
                data = f.read(CHUNK)
                while data:
                    stream.write(data)
                    data = f.read(CHUNK)

                # Close stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                logger.info(f"Audio playback completed: {filename}")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
