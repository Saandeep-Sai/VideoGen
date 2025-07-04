import torch
import google.generativeai as genai
import pygame
import subprocess
import os
import re
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dotenv import load_dotenv
import torchaudio as ta
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import mediainfo
import threading
import sys
import logging
from code_fixer import TargetedCodeFixer  # Import the TargetedCodeFixer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fix Windows encoding issues
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    print("WARNING: ChatterboxTTS not available. Install with: pip install chatterbox-tts")
    CHATTERBOX_AVAILABLE = False

load_dotenv()

class EnhancedVideoGenerator:
    """
    Enhanced automated video generation system with local Chatterbox-TTS integration
    for natural-sounding narration and voice demos.
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "generated_videos", device: str = None):
        """Initialize the enhanced video generator with local TTS."""
        self.gemini_api_key = gemini_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("SETUP: CUDA available, using GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print("SETUP: MPS available, using Apple Silicon GPU")
            else:
                self.device = "cpu"
                print("SETUP: Using CPU")
        else:
            self.device = device
        
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel("gemini-2.5-pro")
        
        pygame.mixer.init()
        
        self.tts_model = None
        self.tts_available = False
        self.code_fixer = TargetedCodeFixer()  # Initialize the code fixer
        
        if CHATTERBOX_AVAILABLE:
            self._initialize_tts()
        else:
            print("ERROR: ChatterboxTTS not available. Using fallback TTS method.")
        
        self.available_voices = self._get_voice_configurations()
        
        print("SUCCESS: Enhanced Video Generator initialized!")
        print(f"VOICES: Available voice configurations: {len(self.available_voices)}")
        print(f"DEVICE: {self.device}")
        print(f"TTS: Available: {self.tts_available}")
    
    def _initialize_tts(self):
        """Initialize TTS with proper error handling and device management."""
        print("LOADING: Chatterbox TTS model...")
        
        try:
            print(f"ATTEMPT: Loading TTS on {self.device}...")
            self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
            self.tts_available = True
            print(f"SUCCESS: Chatterbox TTS model loaded successfully on {self.device}!")
        except Exception as e:
            print(f"ERROR: Failed to load TTS model on {self.device}: {e}")
            if self.device != "cpu":
                print("FALLBACK: Falling back to CPU...")
                try:
                    self.tts_model = ChatterboxTTS.from_pretrained(device="cpu")
                    self.device = "cpu"
                    self.tts_available = True
                    print("SUCCESS: TTS model loaded on CPU")
                except Exception as e2:
                    print(f"ERROR: Failed to load TTS model on CPU: {e2}")
                    self.tts_available = False
            else:
                self.tts_available = False
    
    def _get_voice_configurations(self) -> Dict[str, Dict]:
        """Get available voice configurations for TTS."""
        return {
            "educational": {
                "teacher": {
                    "speed": 1.0,
                    "temperature": 0.7,
                    "pitch_scale": 1.0,
                    "description": "Clear and engaging teacher voice"
                },
                "professor": {
                    "speed": 0.9,
                    "temperature": 0.6,
                    "pitch_scale": 0.95,
                    "description": "Authoritative academic tone"
                }
            },
            "friendly": {
                "guide": {
                    "speed": 1.1,
                    "temperature": 0.8,
                    "pitch_scale": 1.05,
                    "description": "Warm and friendly guide voice"
                },
                "narrator": {
                    "speed": 1.0,
                    "temperature": 0.7,
                    "pitch_scale": 1.0,
                    "description": "Smooth narrator voice"
                }
            },
            "technical": {
                "expert": {
                    "speed": 0.85,
                    "temperature": 0.5,
                    "pitch_scale": 0.9,
                    "description": "Technical expert voice"
                },
                "demo": {
                    "speed": 1.05,
                    "temperature": 0.9,
                    "pitch_scale": 1.1,
                    "description": "Demonstration voice"
                }
            }
        }
    
    def list_voices(self, category: Optional[str] = None) -> None:
        """Display available voice configurations."""
        print("\nAVAILABLE VOICE CONFIGURATIONS")
        print("=" * 60)
        
        if not self.tts_available:
            print("WARNING: TTS not available - voices are for reference only")
        
        categories = [category] if category else self.available_voices.keys()
        
        for cat in categories:
            if cat in self.available_voices:
                print(f"\n{cat.upper()} VOICES:")
                for voice_name, config in self.available_voices[cat].items():
                    print(f"  {voice_name}:")
                    print(f"     Speed: {config['speed']}x")
                    print(f"     Temperature: {config['temperature']}")
                    print(f"     Description: {config['description']}")
                    print()
    
    def play_voice_demo(self, voice_name: str, demo_text: str = None) -> None:
        """Play a voice demo using the local TTS model."""
        if not self.tts_available:
            print("ERROR: TTS not available - cannot play voice demo")
            return
        
        if demo_text is None:
            demo_text = "Hello! This is a demonstration of my voice. I'll be narrating your educational video with clear pronunciation and natural intonation."
        
        print(f"DEMO: Playing demo for voice configuration: {voice_name}")
        
        voice_config = None
        for category in self.available_voices.values():
            if voice_name in category:
                voice_config = category[voice_name]
                break
        
        if not voice_config:
            print(f"ERROR: Voice configuration '{voice_name}' not found")
            return
        
        try:
            demo_audio_path = self._generate_local_audio(
                demo_text, 
                voice_config,
                "temp_demo.wav"
            )
            
            pygame.mixer.music.load(demo_audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
            
            if os.path.exists(demo_audio_path):
                os.remove(demo_audio_path)
            print("SUCCESS: Demo playback complete")
            
        except Exception as e:
            print(f"ERROR: Demo playback failed: {e}")
    
    def _generate_local_audio(self, text: str, voice_config: Dict, output_filename: str) -> str:
        """Generate audio using local Chatterbox-TTS model with proper tensor handling."""
        if not self.tts_available:
            raise Exception("TTS model not available")
        
        try:
            print(f"  AUDIO: Generating audio for: {text[:50]}...")
            
            wav = self.tts_model.generate(
                text,
                temperature=voice_config.get("temperature", 0.7)
            )
            
            if wav.dim() > 1:
                wav = wav.squeeze()
            
            if wav.device.type != 'cpu':
                wav = wav.cpu()
            
            speed = voice_config.get("speed", 1.0)
            if speed != 1.0 and speed > 0:
                original_length = wav.shape[0]
                new_length = int(original_length / speed)
                if new_length > 0:
                    wav_reshaped = wav.unsqueeze(0).unsqueeze(0)
                    wav_resampled = torch.nn.functional.interpolate(
                        wav_reshaped,
                        size=new_length,
                        mode='linear',
                        align_corners=False
                    )
                    wav = wav_resampled.squeeze()
            
            sample_rate = getattr(self.tts_model, 'sr', 22050)
            if hasattr(self.tts_model, 'sample_rate'):
                sample_rate = self.tts_model.sample_rate
            
            output_path = self.output_dir / output_filename
            if output_path.exists():
                try:
                    output_path.unlink()
                    print(f"CLEANUP: Removed existing audio file: {output_path}")
                except Exception as e:
                    print(f"WARNING: Could not remove existing audio file {output_path}: {e}")
            
            if wav.dim() == 1:
                wav_to_save = wav.unsqueeze(0)
            else:
                wav_to_save = wav
            
            ta.save(str(output_path), wav_to_save, sample_rate)
            
            return str(output_path)
            
        except Exception as e:
            raise Exception(f"Local TTS generation failed: {e}")
    
    def _generate_fallback_audio(self, text: str, output_filename: str) -> str:
        """Generate fallback audio using system TTS or silence."""
        print(f"  FALLBACK: Generating fallback audio for: {text[:50]}...")
        
        output_path = self.output_dir / output_filename
        if output_path.exists():
            try:
                output_path.unlink()
                print(f"CLEANUP: Removed existing audio file: {output_path}")
            except Exception as e:
                print(f"WARNING: Could not remove existing audio file {output_path}: {e}")
        
        try:
            if os.name == 'posix':
                result = subprocess.run([
                    'say', '-o', str(output_path).replace('.wav', '.aiff'), text
                ], capture_output=True)
                
                if result.returncode == 0:
                    audio = AudioSegment.from_file(str(output_path).replace('.wav', '.aiff'))
                    audio.export(str(output_path), format="wav")
                    os.remove(str(output_path).replace('.wav', '.aiff'))
                    return str(output_path)
            
            elif os.name == 'nt':
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                if os.path.exists(output_path):
                    return str(output_path)
            
        except Exception as e:
            print(f"  WARNING: System TTS failed: {e}")
        
        print("  SILENCE: Creating silence as final fallback")
        duration_seconds = len(text.split()) * 0.5
        silence = AudioSegment.silent(duration=int(duration_seconds * 1000))
        silence.export(str(output_path), format="wav")
        
        return str(output_path)
    
    def create_enhanced_audio(self, narration_segments: List[Dict], 
                             voice_name: str, output_filename: str) -> str:
        """
        Create enhanced synchronized audio using local TTS or fallback methods.
        """
        print(f"AUDIO: Creating enhanced audio with voice: {voice_name}")
        
        voice_config = None
        for category in self.available_voices.values():
            if voice_name in category:
                voice_config = category[voice_name]
                break
        
        if not voice_config:
            voice_config = self.available_voices["friendly"]["narrator"]
            print(f"WARNING: Voice '{voice_name}' not found, using default narrator")
        
        temp_files = []
        audio_clips = []
        
        for i, segment in enumerate(narration_segments):
            print(f"  SEGMENT: Generating segment {i+1}/{len(narration_segments)}")
            
            try:
                temp_filename = f"temp_enhanced_audio_{i:03d}.wav"
                
                clean_text = re.sub(r'\[.*?\]', '', segment["text"]).strip()
                
                if self.tts_available:
                    audio_path = self._generate_local_audio(
                        clean_text, 
                        voice_config,
                        temp_filename
                    )
                else:
                    audio_path = self._generate_fallback_audio(
                        clean_text,
                        temp_filename
                    )
                
                temp_files.append(audio_path)
                
                audio = AudioSegment.from_wav(audio_path)
                
                expected_duration = segment.get("duration", 10.0) * 1000
                current_duration = len(audio)
                
                if current_duration > expected_duration:
                    audio = audio[:int(expected_duration)]
                elif current_duration < expected_duration:
                    silence_needed = expected_duration - current_duration
                    audio = audio + AudioSegment.silent(duration=int(silence_needed))
                
                audio_clips.append(audio)
                print(f"    SUCCESS: Segment {i+1} complete ({len(audio)/1000:.1f}s)")
                
            except Exception as e:
                print(f"    WARNING: Segment {i+1} failed: {e}, using silence")
                duration_ms = int(segment.get("duration", 3.0) * 1000)
                audio_clips.append(AudioSegment.silent(duration=duration_ms))
        
        master_audio = AudioSegment.silent(duration=0)
        
        for i, segment in enumerate(narration_segments):
            if i < len(audio_clips):
                master_audio += audio_clips[i]
        
        audio_path = self.output_dir / output_filename
        if audio_path.exists():
            try:
                audio_path.unlink()
                print(f"CLEANUP: Removed existing audio file: {audio_path}")
            except Exception as e:
                print(f"WARNING: Could not remove existing audio file {audio_path}: {e}")
        
        master_audio.export(str(audio_path), format="mp3")
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"  WARNING: Could not remove temp file {temp_file}: {e}")
        
        print(f"SUCCESS: Enhanced audio created: {audio_path}")
        return str(audio_path)
    
    def fix_manim_script(self, script: str) -> str:
        """Clean up common issues in Manim scripts while preserving proper indentation."""
        logging.info("🧹 Starting Manim script cleanup...")

        # First, normalize all line endings
        script = script.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into lines to work with indentation properly
        lines = script.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Preserve leading whitespace (indentation)
            leading_whitespace = len(line) - len(line.lstrip())
            indent = line[:leading_whitespace]
            content = line[leading_whitespace:]
            
            # Apply fixes to content only, not indentation
            if content.strip():
                # Fix list slice animate color set
                content = re.sub(
                    r'rects\[(.*?)\:(.*?)\]\.animate\.set_color\((.*?)\)',
                    r'[rect.animate.set_color(\3) for rect in rects[\1:\2]]',
                    content
                )
                
                # Fix broken inline if-else list comprehensions
                content = re.sub(
                    r"self\.play\(\*\[FadeOut\(obj\) for obj in \[(.*?)\] if 'success_text' in locals\(\) else \[(.*?)\]\]\)",
                    r"objects_to_fade = [\1] if 'success_text' in locals() else [\2]\n" + indent + "self.play(*[FadeOut(obj) for obj in objects_to_fade])",
                    content
                )
                
                # Replace problematic symbols
                content = re.sub(r'Checkmark\(.*?color\s*=\s*([^)]+)\)', r'Text("✓", color=\1, font_size=48)', content)
                content = re.sub(r'Cross\(.*?\)', r'Text("✗", font_size=48)', content)
                content = re.sub(r'SVGMobject\("([^"]+)"\)\.scale\((.*?)\)', r'Rectangle(width=2, height=2, color=WHITE, fill_opacity=0.5).scale(\2)', content)
                content = re.sub(r'SVGMobject\("([^"]+)"\)', r'Rectangle(width=2, height=2, color=WHITE, fill_opacity=0.5)', content)
                
                # Fix positioning issues
                content = re.sub(r'(\w+)\.move_to\(([^,]+),\s*buff\s*=\s*([^)]+)\)', r'\1.move_to(\2)\n' + indent + r'\1.next_to(\2, buff=\3)', content)
                content = re.sub(r'\.to_edge\((.*?)\)', r'.move_to(\1 * 0.9)', content)
                content = re.sub(r'\.shift\((.*?)\s*\*\s*([\d.]+)\)', lambda m: f'.shift({m.group(1)} * min({m.group(2)}, 3.0))', content)
                
                # Fix direction typos
                wrong_dirs = {'RIGTH': 'RIGHT', 'LEFFT': 'LEFT', 'UPP': 'UP', 'DOWNN': 'DOWN'}
                for wrong, correct in wrong_dirs.items():
                    content = re.sub(rf'\b{wrong}\b', correct, content)
                
                # Fix Text formatting issues - be more careful with this regex
                content = re.sub(
                    r'Text\((.*?),\s*font_size\s*=\s*(\d+\.?\d*)\s*(.*?)\)',
                    lambda m: f'Text({m.group(1)}, font_size=min({m.group(2)}, 36){m.group(3)})',
                    content
                )
                
                # Remove problematic width parameters from Text objects
                content = re.sub(r'Text\((.*?),\s*width\s*=\s*[\d.]+\s*(.*?)\)', r'Text(\1\2)', content)
                
                # Add proper spacing after commas in function calls (but not in strings)
                # This is a simple approach - you might need a more sophisticated parser for complex cases
                content = re.sub(r',(?! )(?![^"]*"[^"]*$)', r', ', content)
            
            # Reconstruct the line with original indentation
            fixed_line = indent + content if content.strip() else original_line
            fixed_lines.append(fixed_line)
        
        # Join lines back together
        script = '\n'.join(fixed_lines)
        
        # Clean up any double spaces that might have been introduced (but not at line starts)
        lines = script.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Only clean non-empty lines
                leading_whitespace = len(line) - len(line.lstrip())
                indent = line[:leading_whitespace]
                content = line[leading_whitespace:]
                # Clean up double spaces in content only
                cleaned_content = re.sub(r'  +', ' ', content)
                cleaned_lines.append(indent + cleaned_content)
            else:
                cleaned_lines.append(line)  # Keep empty lines as-is
        
        script = '\n'.join(cleaned_lines)
        
        logging.info("✅ Manim script cleanup complete with preserved indentation.")
        return script
    def suggest_voice_for_topic(self, topic: str, level: str = "beginner") -> str:
        """Suggest the best voice configuration for a given topic and level."""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["algorithm", "data structure", "programming", "computer science"]):
            return "professor" if level == "advanced" else "teacher"
        elif any(word in topic_lower for word in ["math", "physics", "chemistry", "science"]):
            return "professor"
        elif any(word in topic_lower for word in ["technical", "engineering", "system", "architecture"]):
            return "expert"
        elif level == "beginner":
            return "guide"
        else:
            return "narrator"
    
    def generate_video_with_voice(self, topic: str, voice_name: str = None, 
                                 duration: int = 60, level: str = "beginner",
                                 language: str = "English", quality: str = "high") -> str:
        """
        Enhanced video generation with local voice synthesis, creating video before audio.
        """
        if voice_name is None:
            voice_name = self.suggest_voice_for_topic(topic, level)
            print(f"VOICE: Auto-selected voice: {voice_name}")
        
        print("STARTING ENHANCED VIDEO GENERATION")
        print("=" * 60)
        print(f"Topic: {topic}")
        print(f"Voice: {voice_name}")
        print(f"Duration: {duration} seconds")
        print(f"Level: {level}")
        print(f"Language: {language}")
        print(f"Quality: {quality}")
        print(f"Device: {self.device}")
        print(f"TTS Available: {self.tts_available}")
        print("=" * 60)
        
        try:
            # Step 1: Generate content using Gemini
            content_data = self.generate_content_alternative(topic, duration, level, language)
            
            # Step 2: Create Manim video first
            video_filename = f"{topic.lower().replace(' ', '_')}_video.mp4"
            video_path = self.create_manim_video(
                content_data["manim_script"],
                "GeneratedAnimation",
                quality,
                video_filename
            )
            
            # Step 3: Create enhanced audio
            audio_filename = f"{topic.lower().replace(' ', '_')}_enhanced_audio.mp3"
            audio_path = self.create_enhanced_audio(
                content_data["narration_segments"], 
                voice_name,
                audio_filename
            )
            
            # Step 4: Combine video and audio
            final_filename = f"{topic.lower().replace(' ', '_')}_enhanced_final.mp4"
            final_path = str(self.output_dir / final_filename)
            
            combined_path = self.combine_video_audio(video_path, audio_path, final_path)
            
            # Step 5: Save enhanced metadata
            metadata = content_data["metadata"]
            voice_config = None
            for category in self.available_voices.values():
                if voice_name in category:
                    voice_config = category[voice_name]
                    break
            if not voice_config:
                voice_config = self.available_voices["friendly"]["narrator"]
            
            metadata.update({
                "voice_name": voice_name,
                "voice_config": voice_config,
                "tts_engine": "chatterbox_local" if self.tts_available else "fallback",
                "device": self.device,
                "tts_available": self.tts_available
            })
            
            metadata_path = self.output_dir / f"{topic.lower().replace(' ', '_')}_enhanced_metadata.json"
            if metadata_path.exists():
                try:
                    metadata_path.unlink()
                    print(f"CLEANUP: Removed existing metadata file: {metadata_path}")
                except Exception as e:
                    print(f"WARNING: Could not remove existing metadata file {metadata_path}: {e}")
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("\nENHANCED VIDEO GENERATION COMPLETE!")
            print("=" * 60)
            print(f"Final video: {combined_path}")
            print(f"Audio file: {audio_path}")
            print(f"Metadata: {metadata_path}")
            print("=" * 60)
            
            return combined_path
            
        except Exception as e:
            print(f"\nERROR: {e}, attempting fallback")
            return self._generate_fallback_video(topic, voice_name, duration, level, quality, language)
    
    def _generate_fallback_video(self, topic: str, voice_name: str, duration: int,
                                level: str, quality: str, language: str) -> str:
        """Generate a fallback video if primary generation fails."""
        print("FALLBACK: Falling back to basic video generation")
        try:
            content_data = {
                "manim_script": self._generate_basic_manim_script(topic),
                "narration_segments": self._generate_basic_narration(topic, duration),
                "metadata": {
                    "topic": topic,
                    "duration": duration,
                    "level": level,
                    "language": language
                }
            }
            
            video_filename = f"{topic.lower().replace(' ', '_')}_video.mp4"
            video_path = self.create_manim_video(
                content_data["manim_script"],
                "GeneratedAnimation",
                quality,
                video_filename
            )
            
            audio_filename = f"{topic.lower().replace(' ', '_')}_enhanced_audio.mp3"
            audio_path = self.create_enhanced_audio(
                content_data["narration_segments"], 
                voice_name,
                audio_filename
            )
            
            final_filename = f"{topic.lower().replace(' ', '_')}_enhanced_final.mp4"
            final_path = str(self.output_dir / final_filename)
            
            combined_path = self.combine_video_audio(video_path, audio_path, final_path)
            
            metadata = content_data["metadata"]
            voice_config = self.available_voices["friendly"]["narrator"]
            metadata.update({
                "voice_name": voice_name,
                "voice_config": voice_config,
                "tts_engine": "chatterbox_local" if self.tts_available else "fallback",
                "device": self.device,
                "tts_available": self.tts_available
            })
            
            metadata_path = self.output_dir / f"{topic.lower().replace(' ', '_')}_enhanced_metadata.json"
            if metadata_path.exists():
                try:
                    metadata_path.unlink()
                    print(f"CLEANUP: Removed existing metadata file: {metadata_path}")
                except Exception as e:
                    print(f"WARNING: Could not remove existing metadata file {metadata_path}: {e}")
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("\nFALLBACK VIDEO GENERATION COMPLETE!")
            print(f"Final video: {combined_path}")
            return combined_path
            
        except Exception as e:
            print(f"ERROR: Fallback generation failed: {e}")
            raise
    
    def generate_content_alternative(self, topic: str, duration: int = 180, 
                                level: str = "beginner", language: str = "English") -> Dict:
        """
        Generate synchronized Manim script and narration segments for a given topic.
        """
        print(f"AI: Generating synchronized content for: {topic}")
        
        try:
            print("AI: Sending request to Gemini for content generation...")
            response = self.gemini.generate_content(f"""
            IMPORTANT CODING INSTRUCTIONS FOR GEMINI (Manim v0.19.0 + Python 3.11) ⚠️

            You are tasked with generating a 100% error-free Manim Python animation script AND synchronized narration segments, based on a given topic. Manim must run successfully in version 0.19.0 using Python 3.11. Follow these exact rules to avoid all common runtime and syntax errors.

            ==================
            🧠 OBJECTIVE
            ==================
            Generate BOTH:
            1. A complete Manim animation Python script using class name: `GeneratedAnimation`
            2. A set of narration segments, formatted properly and synchronized with video

            ==================
            📚 TOPIC INPUT FORMAT
            ==================
            You will receive the topic as: "{topic}"
            The total duration will be: {duration} seconds.

            ==================
            🎙️ NARRATION RULES
            ==================
            1. Narration must fully cover the {duration} seconds with no gaps or overlaps.
            2. Use slow, clear language (~100–120 words per minute).
            3. Add expressive emotions in brackets: [Excited], [Thoughtful], [Emphasize], etc.
            4. Format narration like this (exactly):

            SEGMENT: start_time | duration | narration text

            (Example)
            SEGMENT: 0 | 6 | [Confident] Let's explore the basics of Third Normal Form.
            SEGMENT: 6 | 4 | [Curious] Why is normalization important in databases?

            5. Each `start_time` must be the sum of all previous durations (no floating-point errors).
            6. Keep narration human, clear, and simple. Every segment must describe the animation shown.

            ==================
            🎬 MANIM CODE RULES
            ==================

            --- IMPORTS ---
            • Use ONLY this import (no aliases, no other modules):
            ```python
            from manim import *
            ```

            --- CLASS & STRUCTURE ---
            • Class name must be: `GeneratedAnimation`
            • Define helper functions where appropriate (e.g., `create_table()`)
            • Script must run without edits in Manim v0.19.0

            --- ALLOWED OBJECTS ---
            ✅ Use only these built-in Manim mobjects:
            - Text
            - Rectangle, Circle, Line, Arrow, DashedLine
            - VGroup, Group
            - Table (only if correctly used)
            ❌ DO NOT use:
            - SVGMobject, Checkmark, Cross, or custom symbols
            - External files (.svg, .png, etc.)

            --- TEXT RULES ---
            1. Every `Text(...)` must use this format exactly:
            ```python
            Text("Some Text", font_size=min(N, 36), color=COLOR)
            ```
            • N must be a numeric value ≤ 36 (e.g., 28, 32)
            • Do NOT pass other named arguments inside `min(...)` (e.g., `min(32, 36, width=7.0)` is INVALID)
            • Use colors like: BLUE, GREEN, RED, YELLOW, WHITE

            2. NEVER let text overlap or overflow.
            • Prefer shorter text and use `next_to()` for spacing.
            • Maximum of 2–3 elements on screen at once.

            --- POSITIONING RULES ---
            • Never use:
            ```python
            mobject.move_to(position, buff=...)
            ```
            • Instead, always separate position and buffer:
            ```python
            element.move_to(position)
            element.next_to(other_element, direction, buff=0.3)
            ```

            --- TABLE RULES ---
            • Use this helper function definition for all tables:
            ```python
            def create_table(headers: List[str], data: List[List[str]], title_text: str) -> VGroup:
            ```
            • Only use the built-in `Table` class properly.
            • Each row must be the same length (no ragged data).
            • Never use keyword arguments like `include_header_line` unless confirmed supported.

            --- ANIMATION SYNC ---
            • Use `self.wait(duration)` after every animation to match narration segment.
            • Duration must match narration segment duration exactly.
            • Use `FadeOut(...)` before adding new content to avoid overlap.
            • Avoid long animation chains unless necessary.

            --- CODING SAFETY RULES ---
            • Do not use argument include_header in MObjects __init__ methods
            ⚠️ Never use `element.move_to(..., buff=...)` — this will cause a crash in Manim.
            ✅ Instead:
            - Use `element.move_to(POSITION)` to place at an absolute position
            - Use `element.next_to(other_mobject, DIRECTION, buff=...)` for relative positioning with spacing
            • NEVER perform math on strings (e.g., `"5" ** 2` is invalid)
            • Always use `int(...)` or `float(...)` before math ops.
            • Do NOT use unsupported `min(...)` with extra args (e.g., width, height)
            • Do NOT use `**kwargs` with classes unless verified.
            • All durations passed to `self.wait()` must be ≥ 0.1
            • Avoid calling attributes like `.center`, `.height`, `.width` — use `.get_center()`, etc.
            
            --Sample Code---
            from manim import *
            
            class GeneratedAnimation(Scene):
                def construct(self):
                    title = Text("{topic}", font_size=36, color=BLUE)
                    title.move_to(UP * 3.5)
                    self.play(Write(title))
                    self.wait(2)
                    
                    intro_text = Text("Let's learn about {topic.lower()}", font_size=32, color=WHITE)
                    intro_text.move_to(ORIGIN)
                    self.play(Write(intro_text))
                    self.wait(3)
                    
                    content_rect = Rectangle(width=8, height=4, color=GREEN, fill_opacity=0.1)
                    content_rect.move_to(ORIGIN)
                    self.play(Create(content_rect))
                    self.wait(2)
                    
                    concept1 = Text("Key Concept 1", font_size=24, color=YELLOW)
                    concept1.move_to(UP * 1)
                    self.play(Write(concept1))
                    self.wait(2)
                    
                    concept2 = Text("Key Concept 2", font_size=24, color=YELLOW)
                    concept2.move_to(ORIGIN)
                    self.play(Write(concept2))
                    self.wait(2)
                    
                    concept3 = Text("Key Concept 3", font_size=24, color=YELLOW)
                    concept3.move_to(DOWN * 1)
                    self.play(Write(concept3))
                    self.wait(2)
                    
                    summary = Text("Summary: Understanding {topic}", font_size=28, color=GREEN)
                    summary.move_to(DOWN * 3.5)
                    self.play(Write(summary))
                    self.wait(3)
                    
                    self.play(*[FadeOut(mob) for mob in self.mobjects])
                    self.wait(1)

            ==================
            ✅ OUTPUT FORMAT (MUST FOLLOW)
            ==================

            ---NARRATION---
            SEGMENT: start | duration | narration text
            SEGMENT: ...

            ---MANIM---
            from manim import *

            class GeneratedAnimation(Scene):
                def construct(self):
                    # Manim animation code here

            ==================
            IF YOU BREAK ANY RULE ABOVE, THE SCRIPT WILL FAIL TO EXECUTE.
            ==================

            Be clean, strict, and follow instructions with full accuracy.
            """)
        except Exception as e:
            print(f"⚠️ Gemini content generation failed: {e}")
            
        try:
            print("=" * 60)
            print("AI: Content generation response received")
            print("AI: Response length:", len(response.text))
            content = response.text.strip()
            
            if "---MANIM---" not in content or "---NARRATION---" not in content:
                raise ValueError("Response format incorrect")
            
            # Split the content properly
            content_parts = content.split("---MANIM---")
            if len(content_parts) < 2:
                raise ValueError("MANIM section not found")
            
            # Get the part after ---MANIM---
            manim_part = content_parts[1]
            
            # If there's a ---NARRATION--- section, split it out
            if "---NARRATION---" in manim_part:
                manim_script = manim_part.split("---NARRATION---")[0].strip()
                narration_block = manim_part.split("---NARRATION---")[1].strip()
            else:
                # If narration comes first, get it from the first part
                manim_script = manim_part.strip()
                narration_block = content_parts[0].split("---NARRATION---")[1].strip() if "---NARRATION---" in content_parts[0] else ""

            # Clean up code block markers but preserve indentation
            if manim_script.startswith("```python"):
                # Find the first newline after ```python
                first_newline = manim_script.find('\n')
                if first_newline != -1:
                    manim_script = manim_script[first_newline + 1:]
            elif manim_script.startswith("```"):
                # Find the first newline after ```
                first_newline = manim_script.find('\n')
                if first_newline != -1:
                    manim_script = manim_script[first_newline + 1:]
            
            if manim_script.endswith("```"):
                # Remove the ending ``` but keep everything else
                last_backticks = manim_script.rfind("```")
                if last_backticks != -1:
                    manim_script = manim_script[:last_backticks]

            # Ensure proper imports and class name
            if "from manim import *" not in manim_script:
                manim_script = "from manim import *\n\n" + manim_script
            
            # Fix class name if needed
            if "class GeneratedAnimation" not in manim_script:
                manim_script = manim_script.replace("class ", "class GeneratedAnimation", 1)
            
            # Debug: Print the extracted script to see if indentation is preserved
            print("DEBUG: Extracted Manim script:")
            print("=" * 40)
            print(manim_script)
            print("=" * 40)
            
            # Parse narration segments
            narration_segments = []
            current_time = 0.0
            
            for line in narration_block.splitlines():
                line = line.strip()
                if line.startswith("SEGMENT:"):
                    try:
                        # Remove "SEGMENT:" and split by "|"
                        segment_content = line[8:].strip()
                        parts = segment_content.split('|')
                        if len(parts) >= 3:
                            start_time = float(parts[0].strip())
                            duration_seg = float(parts[1].strip())
                            text = parts[2].strip()
                            
                            narration_segments.append({
                                "start_time": start_time,
                                "duration": duration_seg,
                                "text": text
                            })
                            current_time = start_time + duration_seg
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Skipping invalid narration segment: {line}, error: {e}")
                        continue
            
            if not narration_segments:
                print("⚠️ No valid narration segments found, using fallback")
                narration_segments = self._generate_basic_narration(topic, duration)
            
            # Clean up the Manim script
            print("🔧 Cleaning up Manim script...")
            manim_script = self.fix_manim_script(manim_script)
            
            # Adjust narration duration if needed
            total_duration = sum(segment["duration"] for segment in narration_segments)
            if abs(total_duration - duration) > 1.0:
                print(f"⚠️ Adjusting narration duration from {total_duration}s to {duration}s")
                scale_factor = duration / total_duration if total_duration > 0 else 1.0
                current_time = 0.0
                for segment in narration_segments:
                    segment["duration"] *= scale_factor
                    segment["start_time"] = current_time
                    current_time += segment["duration"]
            
            return {
                "manim_script": manim_script,
                "narration_segments": narration_segments,
                "metadata": {
                    "topic": topic,
                    "duration": duration,
                    "level": level,
                    "language": language
                }
            }
        
        except Exception as e:
            print(f"⚠️ Content generation failed: {e}, using fallback")
            return {
                "manim_script": self._generate_basic_manim_script(topic),
                "narration_segments": self._generate_basic_narration(topic, duration),
                "metadata": {
                    "topic": topic,
                    "duration": duration,
                    "level": level,
                    "language": language
                }
            }   
    def _generate_basic_manim_script(self, topic: str) -> str:
        """Generate a basic Manim script as fallback with text constraints."""
        return f'''from manim import *

class GeneratedAnimation(Scene):
    def construct(self):
        title = Text("{topic}", font_size=36, width=7.0, color=BLUE)
        title.move_to(UP * 3.5)  # Keep within frame with padding
        self.play(Write(title))
        self.wait(2)
        
        intro_text = Text("Let's learn about {topic.lower()}", font_size=32, width=7.0, color=WHITE)
        intro_text.move_to(ORIGIN)  # Center text
        self.play(Write(intro_text))
        self.wait(3)
        
        content_rect = Rectangle(width=8, height=4, color=GREEN, fill_opacity=0.1)
        content_rect.move_to(ORIGIN)
        self.play(Create(content_rect))
        self.wait(2)
        
        concept1 = Text("Key Concept 1", font_size=24, width=7.0, color=YELLOW)
        concept1.move_to(UP * 1)
        self.play(Write(concept1))
        self.wait(2)
        
        concept2 = Text("Key Concept 2", font_size=24, width=7.0, color=YELLOW)
        concept2.move_to(ORIGIN)
        self.play(Write(concept2))
        self.wait(2)
        
        concept3 = Text("Key Concept 3", font_size=24, width=7.0, color=YELLOW)
        concept3.move_to(DOWN * 1)
        self.play(Write(concept3))
        self.wait(2)
        
        summary = Text("Summary: Understanding {topic}", font_size=28, width=7.0, color=GREEN)
        summary.move_to(DOWN * 3.5)
        self.play(Write(summary))
        self.wait(3)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
'''
    
    def _generate_basic_narration(self, topic: str, duration: int) -> List[Dict]:
        """Generate basic narration segments as fallback."""
        segment_duration = duration / 8
        segments = [
            {"start_time": 0.0, "duration": segment_duration, "text": f"Welcome to our tutorial on {topic}"},
            {"start_time": segment_duration, "duration": segment_duration, "text": f"Today we'll learn the fundamental concepts of {topic.lower()}"},
            {"start_time": segment_duration * 2, "duration": segment_duration, "text": "Let's start by understanding the basic structure"},
            {"start_time": segment_duration * 3, "duration": segment_duration, "text": "The first key concept is very important"},
            {"start_time": segment_duration * 4, "duration": segment_duration, "text": "The second concept builds upon the first"},
            {"start_time": segment_duration * 5, "duration": segment_duration, "text": "The third concept completes our understanding"},
            {"start_time": segment_duration * 6, "duration": segment_duration, "text": f"In summary, {topic.lower()} is a fundamental concept"},
            {"start_time": segment_duration * 7, "duration": segment_duration, "text": "Thank you for watching this tutorial"}
        ]
        return segments
    
    def create_manim_video(self, manim_script: str, class_name: str = "GeneratedAnimation",
                      quality: str = "high", output_filename: str = "generated_video.mp4") -> str:
        """Create Manim video from script with error handling and code fixing."""
        print("🎬 Creating Manim video...")
        
        quality_flags = {
            "low": "-ql",
            "medium": "-qm",
            "high": "-qh",
            "ultra": "-qk"
        }
        
        script_path = self.output_dir / "temp_animation.py"
        max_retries = 3  # Maximum number of retries for fixing errors
        
        for attempt in range(max_retries):
            print(f"🛠️ Attempt {attempt + 1}/{max_retries} to execute Manim script")
            
            try:
                # Write/overwrite the script file with current version
                if script_path.exists():
                    try:
                        script_path.unlink()
                        print(f"🗑️ Removed existing script file: {script_path}")
                    except Exception as e:
                        print(f"⚠️ Could not remove existing script file {script_path}: {e}")
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(manim_script)
                
                quality_flag = quality_flags.get(quality, "-qh")
                
                cmd = [
                    "manim", quality_flag, "--disable_caching",
                    "temp_animation.py", class_name
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.output_dir))
                except Exception as e:
                    logging.info(f"⚠️ Error occurred while running Manim command: {e}")    
                
                if result.returncode == 0:
                    # Success! Find and move the video file
                    media_dir = self.output_dir / "media" / "videos" / "temp_animation"
                    
                    for quality_dir in media_dir.glob("*"):
                        if quality_dir.is_dir():
                            for video_file in quality_dir.glob(f"{class_name}.mp4"):
                                final_path = self.output_dir / output_filename
                                if final_path.exists():
                                    try:
                                        final_path.unlink()
                                        print(f"🗑️ Removed existing video file: {final_path}")
                                    except Exception as e:
                                        print(f"⚠️ Could not remove existing video file {final_path}: {e}")
                                video_file.rename(final_path)
                                print(f"✅ Video created: {final_path}")
                                
                                # Clean up temporary script
                                try:
                                    if script_path.exists():
                                        script_path.unlink()
                                except Exception as e:
                                    print(f"⚠️ Could not delete temp script: {e}")
                                
                                return str(final_path)
                
                else:
                    # Error occurred, try to fix it
                    print(f"⚠️ Manim command failed with output: {result.stderr}")
                    error_msg = result.stderr
                    
                    # Use TargetedCodeFixer to fix the script IN-PLACE
                    print(f"🔧 Attempting to fix script errors...")
                    
                    # Get the fixed code and overwrite the original file
                    fixed_result = self.code_fixer.fix_file_with_error(str(script_path), error_msg, return_fixed_code=True)
                    
                    if fixed_result and fixed_result[0] is not None:
                        fixed_code, fixed_file_path = fixed_result
                        print(f"✅ Script fixed successfully")
                        
                        # Update manim_script with the fixed code for next iteration
                        manim_script = fixed_code
                        
                        # Clean up the temporary fixed file if it exists and is different from our main file
                        if fixed_file_path and Path(fixed_file_path).exists() and Path(fixed_file_path) != script_path:
                            try:
                                Path(fixed_file_path).unlink()
                                print(f"🗑️ Removed temporary fixed script: {fixed_file_path}")
                            except Exception as e:
                                print(f"⚠️ Could not remove temporary fixed script {fixed_file_path}: {e}")
                        
                        # Continue to next iteration with the fixed code
                        continue
                    else:
                        print(f"❌ Failed to fix script errors")
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Manim failed after {max_retries} attempts: {error_msg}")
            
            except Exception as e:
                print(f"⚠️ Error during Manim execution: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Manim failed after {max_retries} attempts: {str(e)}")
        
        raise FileNotFoundError("Generated video file not found after all retries")
    
    def combine_video_audio(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Combine video and audio using FFmpeg."""
        print("🎞️ Combining video and audio...")
        
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"🗑️ Removed existing file: {output_path}")
            except Exception as e:
                print(f"⚠️ Could not remove existing file {output_path}: {e}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ FFmpeg command failed with output: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"✅ Final video created: {output_path}")
        return output_path

def demo_voice_selection():
    """Demo function to showcase voice selection and testing."""
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("❌ Please set GEMINI_API_KEY environment variable")
        return
    
    generator = EnhancedVideoGenerator(GEMINI_API_KEY)
    
    print("🎙️ VOICE SELECTION DEMO")
    print("=" * 50)
    
    generator.list_voices()
    
    if not generator.tts_available:
        print("\n⚠️ TTS not available - skipping voice demos")
        print("📦 To enable TTS, install: pip install chatterbox-tts")
        return
    
    demo_voices = ["teacher", "professor", "guide", "narrator", "expert"]
    demo_text = "Hello! I'm demonstrating different voice configurations for your educational videos. Each voice has unique characteristics that work well for different types of content."
    
    for voice in demo_voices:
        print(f"\n🎵 Testing voice: {voice}")
        try:
            response = input("Press Enter to play demo, 's' to skip, or 'q' to quit: ").lower()
            if response == 'q':
                break
            elif response == 's':
                continue
            
            generator.play_voice_demo(voice, demo_text)
        except KeyboardInterrupt:
            print("\nDemo interrupted")
            break
        except Exception as e:
            print(f"⚠️ Demo failed: {e}, continuing to next voice")
    
    print("\n🎬 Generating test video with selected voice...")
    topic = "Binary Search Algorithm"
    voice = "teacher"
    
    try:
        final_video = generator.generate_video_with_voice(
            topic=topic,
            voice_name=voice,
            duration=180,
            level="beginner",
            quality="medium"
        )
        try:
            pygame.mixer.music.load(pygame.mixer.Sound(pygame.mixer.Sound(buffer=pygame.sndarray.make_sound(
                (32767 * 0.5 * (torch.sin(2.0 * torch.pi * torch.arange(44100) * 440 / 44100))).numpy().astype("int16")
            ))))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
        except Exception as e:
            print(f"⚠️ Could not play notification sound: {e}")
        
        print(f"✅ Test video created: {final_video}")
        
    except Exception as e:
        print(f"⚠️ Test video failed: {e}")

def install_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        print("✅ PyTorch found")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import torchaudio
        print("✅ Torchaudio found")
    except ImportError:
        missing_deps.append("torchaudio")
    
    try:
        import google.generativeai
        print("✅ Google GenerativeAI found")
    except ImportError:
        missing_deps.append("google-generativeai")
    
    try:
        import pygame
        print("✅ Pygame found")
    except ImportError:
        missing_deps.append("pygame")
    
    try:
        from pydub import AudioSegment
        print("✅ Pydub found")
    except ImportError:
        missing_deps.append("pydub")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterboxTTS found")
    except ImportError:
        print("⚠️  ChatterboxTTS not found (optional for advanced TTS)")
        print("   Install with: pip install chatterbox-tts")
    
    try:
        result = subprocess.run(["manim", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Manim found")
        else:
            print("❌ Manim not working properly")
            missing_deps.append("manim")
    except FileNotFoundError:
        print("❌ Manim not found")
        missing_deps.append("manim")
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg found")
        else:
            print("❌ FFmpeg not working properly")
            missing_deps.append("ffmpeg")
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        missing_deps.append("ffmpeg")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📦 Install missing Python packages with:")
        for dep in missing_deps:
            if dep not in ["manim", "ffmpeg"]:
                print(f"   pip install {dep}")
        if "manim" in missing_deps:
            print("   pip install manim")
        if "ffmpeg" in missing_deps:
            print("   Install FFmpeg from: https://ffmpeg.org/download.html")
        return False
    else:
        print("\n✅ All dependencies found!")
        return True

def main():
    """Main function with enhanced features and error handling."""
    
    print("🎬 Enhanced Video Generator")
    print("=" * 50)
    
    if not install_dependencies():
        print("\n❌ Please install missing dependencies before continuing")
        return
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("   export GEMINI_API_KEY='your_api_key_here'")
        return
    
    print("\n🎯 What would you like to do?")
    print("1. Demo different voices")
    print("2. Generate a video with custom settings")
    print("3. Generate a quick test video")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        demo_voice_selection()
        return
    
    try:
        generator = EnhancedVideoGenerator(GEMINI_API_KEY)
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return
    
    if choice == "2":
        print("\n📚 Custom Video Generation")
        print("-" * 30)
        
        generator.list_voices("educational")
        
        topic = input("\n📚 Enter topic: ") or "Stack Data Structure"
        
        print(f"\n🎙️ Available voices: teacher, professor, guide, narrator, expert")
        voice = input("🎙️ Enter voice name (or press Enter for auto-suggestion): ").strip() or None
        
        try:
            duration = int(input("⏱️ Enter duration in seconds (default: 60): ") or "60")
        except ValueError:
            duration = 60
        
        level = input("🎯 Enter level (beginner/intermediate/advanced, default: beginner): ").strip() or "beginner"
        quality = input("🎥 Enter quality (low/medium/high/ultra, default: high): ").strip() or "high"
        
        try:
            final_video = generator.generate_video_with_voice(
                topic=topic,
                voice_name=voice,
                duration=duration,
                level=level,
                quality=quality
            )
            print(f"\n🎉 Success! Video created: {final_video}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    elif choice == "3":
        print("\n⚡ Quick Test Video Generation")
        print("-" * 30)
        
        topic = "Operations performed on a B+ Tree"
        voice = "teacher"
        
        print(f"📚 Topic: {topic}")
        print(f"🎙️ Voice: {voice}")
        print(f"⏱️ Duration: 180 seconds")
        print(f"🎥 Quality: medium")
        
        try:
            final_video = generator.generate_video_with_voice(
                topic=topic,
                voice_name=voice,
                duration=180,
                level="beginner",
                quality="medium"
            )
            try:
                pygame.mixer.music.load(pygame.mixer.Sound(pygame.mixer.Sound(buffer=pygame.sndarray.make_sound(
                    (32767 * 0.5 * (torch.sin(2.0 * torch.pi * torch.arange(44100) * 440 / 44100))).numpy().astype("int16")
                ))))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ Could not play notification sound: {e}")
            
            print(f"🎉 Success! Test video created: {final_video}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()