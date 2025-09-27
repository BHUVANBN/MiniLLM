#!/usr/bin/env python3
"""
Voice Chat Module for MiniLLM

Provides speech-to-text and text-to-speech functionality for voice conversations.
"""

import threading
import queue
import time
import logging
from typing import Callable, Optional

try:
    import speech_recognition as sr
    import pyttsx3
    import pyaudio
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"Voice chat dependencies not available: {e}")
    print("Install with: pip install speechrecognition pyttsx3 pyaudio")

logger = logging.getLogger(__name__)

class VoiceChat:
    def __init__(self):
        self.available = VOICE_AVAILABLE
        if not self.available:
            return
            
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Initialize text-to-speech
        self.tts_engine = None
        
        # Voice settings
        self.listening = False
        self.speaking = False
        
        # Callbacks
        self.on_speech_recognized = None
        self.on_listening_start = None
        self.on_listening_stop = None
        self.on_speaking_start = None
        self.on_speaking_stop = None
        self.on_error = None
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize voice components."""
        try:
            # Initialize microphone
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 50)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            logger.info("Voice chat initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            self.available = False
    
    def set_callbacks(self, 
                     on_speech_recognized: Optional[Callable] = None,
                     on_listening_start: Optional[Callable] = None,
                     on_listening_stop: Optional[Callable] = None,
                     on_speaking_start: Optional[Callable] = None,
                     on_speaking_stop: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set callback functions for voice events."""
        self.on_speech_recognized = on_speech_recognized
        self.on_listening_start = on_listening_start
        self.on_listening_stop = on_listening_stop
        self.on_speaking_start = on_speaking_start
        self.on_speaking_stop = on_speaking_stop
        self.on_error = on_error
    
    def start_listening(self):
        """Start listening for speech in a separate thread."""
        if not self.available or self.listening:
            return False
        
        def listen_thread():
            try:
                self.listening = True
                if self.on_listening_start:
                    self.on_listening_start()
                
                with self.microphone as source:
                    logger.info("Listening for speech...")
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                
                self.listening = False
                if self.on_listening_stop:
                    self.on_listening_stop()
                
                # Recognize speech
                try:
                    logger.info("Recognizing speech...")
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"Recognized: {text}")
                    
                    if self.on_speech_recognized:
                        self.on_speech_recognized(text)
                        
                except sr.UnknownValueError:
                    error_msg = "Could not understand audio"
                    logger.warning(error_msg)
                    if self.on_error:
                        self.on_error(error_msg)
                        
                except sr.RequestError as e:
                    error_msg = f"Speech recognition service error: {e}"
                    logger.error(error_msg)
                    if self.on_error:
                        self.on_error(error_msg)
                        
            except sr.WaitTimeoutError:
                self.listening = False
                if self.on_listening_stop:
                    self.on_listening_stop()
                error_msg = "Listening timeout - no speech detected"
                logger.warning(error_msg)
                if self.on_error:
                    self.on_error(error_msg)
                    
            except Exception as e:
                self.listening = False
                if self.on_listening_stop:
                    self.on_listening_stop()
                error_msg = f"Listening error: {e}"
                logger.error(error_msg)
                if self.on_error:
                    self.on_error(error_msg)
        
        # Start listening in background thread
        threading.Thread(target=listen_thread, daemon=True).start()
        return True
    
    def stop_listening(self):
        """Stop listening for speech."""
        self.listening = False
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        if not self.available or self.speaking:
            return False
        
        def speak_thread():
            try:
                self.speaking = True
                if self.on_speaking_start:
                    self.on_speaking_start()
                
                logger.info(f"Speaking: {text[:50]}...")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
                self.speaking = False
                if self.on_speaking_stop:
                    self.on_speaking_stop()
                    
            except Exception as e:
                self.speaking = False
                if self.on_speaking_stop:
                    self.on_speaking_stop()
                error_msg = f"Speech synthesis error: {e}"
                logger.error(error_msg)
                if self.on_error:
                    self.on_error(error_msg)
        
        # Start speaking in background thread
        threading.Thread(target=speak_thread, daemon=True).start()
        return True
    
    def stop_speaking(self):
        """Stop current speech synthesis."""
        if self.available and self.tts_engine:
            try:
                self.tts_engine.stop()
                self.speaking = False
                if self.on_speaking_stop:
                    self.on_speaking_stop()
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def is_listening(self):
        """Check if currently listening."""
        return self.listening
    
    def is_speaking(self):
        """Check if currently speaking."""
        return self.speaking
    
    def is_available(self):
        """Check if voice chat is available."""
        return self.available
    
    def get_microphones(self):
        """Get list of available microphones."""
        if not self.available:
            return []
        
        try:
            mic_list = []
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                mic_list.append((index, name))
            return mic_list
        except Exception as e:
            logger.error(f"Error getting microphones: {e}")
            return []
    
    def set_microphone(self, device_index: int):
        """Set the microphone device."""
        if not self.available:
            return False
        
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            # Re-adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            return True
        except Exception as e:
            logger.error(f"Error setting microphone: {e}")
            return False
    
    def get_voices(self):
        """Get list of available TTS voices."""
        if not self.available or not self.tts_engine:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            for voice in voices:
                voice_list.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown')
                })
            return voice_list
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []
    
    def set_voice(self, voice_id: str):
        """Set the TTS voice."""
        if not self.available or not self.tts_engine:
            return False
        
        try:
            self.tts_engine.setProperty('voice', voice_id)
            return True
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def set_speech_rate(self, rate: int):
        """Set the speech rate (words per minute)."""
        if not self.available or not self.tts_engine:
            return False
        
        try:
            # Clamp rate between 50 and 300
            rate = max(50, min(300, rate))
            self.tts_engine.setProperty('rate', rate)
            return True
        except Exception as e:
            logger.error(f"Error setting speech rate: {e}")
            return False
    
    def set_volume(self, volume: float):
        """Set the speech volume (0.0 to 1.0)."""
        if not self.available or not self.tts_engine:
            return False
        
        try:
            # Clamp volume between 0.0 and 1.0
            volume = max(0.0, min(1.0, volume))
            self.tts_engine.setProperty('volume', volume)
            return True
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return False
