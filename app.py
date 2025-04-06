import streamlit as st
import os
import tempfile
from pathlib import Path
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
import re
import time
import pickle
import uuid
import sys
import subprocess
import requests
import yt_dlp  # Use yt-dlp instead of pytube for more reliable YouTube downloads
import moviepy.editor as mp  # For extracting audio from video files

st.set_page_config(
    page_title="Audio/Video Transcription & Summary Agent",
    page_icon="🎧",
    layout="wide"
)

# Define paths for model storage
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

WHISPER_MODEL_PATH = MODELS_DIR / "whisper_model.pkl"
SUMMARIZER_MODEL_PATH = MODELS_DIR / "summarizer_model.pkl"

@st.cache_resource
def load_transcription_model():
    """Load Whisper model from pickle if exists, otherwise download and save"""
    if WHISPER_MODEL_PATH.exists() and os.path.getsize(WHISPER_MODEL_PATH) > 1000000:  # Check if file exists and is a reasonable size
        st.sidebar.info("Loading Whisper model from saved file...")
        try:
            with open(WHISPER_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            st.sidebar.success("✅ Loaded Whisper model from file")
            return model
        except Exception as e:
            st.sidebar.warning(f"Failed to load saved model: {e}. Downloading fresh model...")
    
    # Load Whisper model (tiny size for quick testing, change to "base" or "small" for better results)
    model = whisper.load_model("tiny")
    
    # Save model to pickle file
    try:
        with open(WHISPER_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        st.sidebar.success("✅ Saved Whisper model to file")
    except Exception as e:
        st.sidebar.warning(f"Failed to save model: {e}")
    
    return model

@st.cache_resource
def load_summarization_model():
    """Load summarization model from pickle if exists, otherwise download and save"""
    if SUMMARIZER_MODEL_PATH.exists() and os.path.getsize(SUMMARIZER_MODEL_PATH) > 1000000:  # Check if file exists and is a reasonable size
        st.sidebar.info("Loading summarization model from saved file...")
        try:
            with open(SUMMARIZER_MODEL_PATH, 'rb') as f:
                summarizer = pickle.load(f)
            st.sidebar.success("✅ Loaded summarization model from file")
            return summarizer
        except Exception as e:
            st.sidebar.warning(f"Failed to load saved model: {e}. Downloading fresh model...")
    
    # Load a smaller, faster model for summarization
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    # Save model to pickle file
    try:
        with open(SUMMARIZER_MODEL_PATH, 'wb') as f:
            pickle.dump(summarizer, f)
        st.sidebar.success("✅ Saved summarization model to file")
    except Exception as e:
        st.sidebar.warning(f"Failed to save model: {e}")
    
    return summarizer

def extract_audio_from_youtube(youtube_url):
    """Download audio from YouTube video using yt-dlp (more reliable than pytube)"""
    try:
        # Create temp dir for audio
        temp_dir = tempfile.mkdtemp()
        output_filename = os.path.join(temp_dir, "audio.mp3")
        
        st.write("Downloading from YouTube... This may take a moment.")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_filename.replace('.mp3', ''),
            'quiet': True,
            'no_warnings': True
        }
        
        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            title = info_dict.get('title', 'YouTube_Audio')
        
        # The actual filename after yt-dlp processing
        actual_output_file = output_filename.replace('.mp3', '') + '.mp3'
        
        if os.path.exists(actual_output_file):
            st.write(f"✅ Downloaded audio file ({os.path.getsize(actual_output_file) / 1024 / 1024:.2f} MB)")
            return actual_output_file, title
        else:
            st.error(f"Expected output file not found: {actual_output_file}")
            # List files in the directory for debugging
            st.write(f"Files in directory: {os.listdir(temp_dir)}")
            return None, None
            
    except Exception as e:
        st.error(f"Error downloading YouTube audio: {str(e)}")
        # Try an alternative method if the first fails
        try:
            st.write("Trying alternative download method...")
            output_file = os.path.join(temp_dir, "audio_alt.mp3")
            cmd = f'yt-dlp -x --audio-format mp3 -o "{output_file.replace(".mp3", "")}.%(ext)s" {youtube_url}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if os.path.exists(output_file):
                return output_file, "YouTube Audio"
            else:
                st.error(f"Alternative method failed. Error: {result.stderr}")
                return None, None
        except Exception as alt_e:
            st.error(f"Alternative download method also failed: {str(alt_e)}")
            return None, None

def extract_audio_from_video(video_path):
    """Extract audio from video file using moviepy"""
    try:
        st.write("Extracting audio from video file...")
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
        
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        if os.path.exists(audio_path):
            st.write(f"✅ Extracted audio file ({os.path.getsize(audio_path) / 1024 / 1024:.2f} MB)")
            return audio_path
        else:
            st.error("Failed to extract audio from video")
            return None
    except Exception as e:
        st.error(f"Error extracting audio from video: {e}")
        return None

def transcribe_audio(audio_file_path, transcription_model):
    """Transcribe audio file using Whisper with improved error handling and debugging"""
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            st.error(f"Audio file not found at path: {audio_file_path}")
            return None
        
        # Display file info for debugging
        file_size_mb = os.path.getsize(audio_file_path) / (1024*1024)
        st.write(f"Transcribing file: {os.path.basename(audio_file_path)} (Size: {file_size_mb:.2f} MB)")
        
        if file_size_mb < 0.01:
            st.error("Audio file is too small, possibly corrupted or empty.")
            return None
        
        # Transcribe the audio file
        st.write("Starting transcription with Whisper model... This may take several minutes for longer files.")
        result = transcription_model.transcribe(audio_file_path)
        
        if not result or not result.get("text"):
            st.warning("Transcription completed but no text was extracted. The audio might not contain speech.")
            return None
            
        return result["text"]
    except FileNotFoundError:
        st.error(f"Audio file not found. Please check if the file exists: {audio_file_path}")
        return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        st.error(f"Error type: {type(e).__name__}, Python version: {sys.version}")
        
        # Try using ffmpeg to check if the audio file is valid
        try:
            st.write("Checking audio file validity...")
            cmd = f'ffmpeg -i "{audio_file_path}" -f null - 2>&1'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if "Invalid data found" in result.stderr:
                st.error("The audio file appears to be corrupted or in an unsupported format.")
            else:
                st.write("Audio file seems valid. The issue may be with the transcription model.")
        except Exception as ffmpeg_e:
            st.error(f"Failed to check audio file: {ffmpeg_e}")
        
        return None

def chunk_text(text, max_chunk_size=1000):
    """Split text into chunks for summarization"""
    # Split by sentences to avoid cutting in the middle of a sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_text(text, summarizer):
    """Generate a summary of the text"""
    try:
        # If text is too long, chunk it and summarize each chunk
        if len(text) > 1000:
            chunks = chunk_text(text)
            summaries = []
            
            with st.spinner("Generating summary (this may take a few minutes for long content)..."):
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks):
                    progress_bar.progress((i + 1) / len(chunks))
                    chunk_summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                    summaries.append(chunk_summary)
            
            # Combine chunk summaries and create a final summary
            intermediate_summary = " ".join(summaries)
            if len(intermediate_summary) > 1000:
                final_summary = summarizer(intermediate_summary, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
                return final_summary
            return intermediate_summary
        else:
            return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

# Function to check FFmpeg installation
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("✅ FFmpeg is installed")
            return True
        else:
            st.sidebar.error("❌ FFmpeg check failed")
            return False
    except Exception:
        st.sidebar.error("❌ FFmpeg not found. Please install it for audio processing.")
        st.sidebar.markdown("""
        **To install FFmpeg:**
        - Windows: Download from [here](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
        - Mac: `brew install ffmpeg`
        - Linux: `sudo apt-get install ffmpeg`
        """)
        return False

# App title and description
st.title("🎧 Audio/Video Transcription & Summary Agent")
st.markdown("""
Upload an audio/video file or provide a YouTube URL to get a transcript and summary.
""")

# Check dependencies
ffmpeg_available = check_ffmpeg()

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Running on: {device}")

# Install required packages if missing
st.sidebar.header("Dependencies")
with st.sidebar.expander("Check and install dependencies"):
    if st.button("Install missing dependencies"):
        try:
            st.write("Installing yt-dlp...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            st.write("Installing moviepy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
            st.success("Dependencies installed successfully!")
            st.warning("Please restart the app for changes to take effect")
        except Exception as e:
            st.error(f"Error installing dependencies: {e}")

# Load models
with st.spinner("Loading AI models (this might take a minute on first run)..."):
    try:
        transcription_model = load_transcription_model()
        summarization_model = load_summarization_model()
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Input selection
input_method = st.radio("Choose input method:", ["YouTube URL", "Upload File"])

transcript = None
title = None

if input_method == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube URL:")
    
    if youtube_url and st.button("Process"):
        with st.spinner("Working on YouTube video..."):
            audio_file, title = extract_audio_from_youtube(youtube_url)
            
            if audio_file:
                st.success(f"✅ Downloaded: {title}")
                
                with st.spinner("Transcribing audio..."):
                    start_time = time.time()
                    transcript = transcribe_audio(audio_file, transcription_model)
                    transcription_time = time.time() - start_time
                    
                if transcript:
                    st.success(f"✅ Transcription completed in {transcription_time:.2f} seconds")
                    # Clean up temp file but add error handling
                    try:
                        os.remove(audio_file)
                    except Exception as e:
                        st.warning(f"Note: Could not remove temporary file: {e}")
                
else:  # Upload File
    uploaded_file = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "mp4", "avi", "mov", "m4a", "flac", "ogg"])
    
    if uploaded_file and st.button("Process"):
        title = uploaded_file.name
        
        # Create a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        
        # Save uploaded file to temp directory with proper error handling
        temp_dir = tempfile.mkdtemp()
        uploaded_file_path = os.path.join(temp_dir, unique_filename)
        
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File saved temporarily: {os.path.basename(uploaded_file_path)} ({os.path.getsize(uploaded_file_path) / (1024*1024):.2f} MB)")
        
        # Determine if it's a video file and extract audio if needed
        audio_path = uploaded_file_path
        is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))
        
        if is_video:
            st.info("Video file detected. Extracting audio...")
            audio_path = extract_audio_from_video(uploaded_file_path)
            if not audio_path:
                st.error("Failed to extract audio from video file.")
                st.stop()
        
        with st.spinner("Transcribing audio..."):
            start_time = time.time()
            transcript = transcribe_audio(audio_path, transcription_model)
            transcription_time = time.time() - start_time
            
        if transcript:
            st.success(f"✅ Transcription completed in {transcription_time:.2f} seconds")
            # Clean up temp files but add error handling
            try:
                if is_video and audio_path != uploaded_file_path:
                    os.remove(audio_path)
                os.remove(uploaded_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                st.warning(f"Note: Could not remove all temporary files: {e}")
        else:
            st.error("Transcription failed. Please check the file format and try again.")

# Display transcript and generate summary if transcript exists
if transcript:
    st.header("Transcript")
    st.write(transcript)
    
    st.header("Summary")
    with st.spinner("Generating summary..."):
        start_time = time.time()
        summary = summarize_text(transcript, summarization_model)
        summary_time = time.time() - start_time
    
    if summary:
        st.success(f"✅ Summary generated in {summary_time:.2f} seconds")
        st.write(summary)
    
    # Download options
    col1, col2 = st.columns(2)
    
    if transcript:
        with col1:
            st.download_button(
                "Download Transcript",
                transcript,
                file_name=f"{title.split('.')[0]}_transcript.txt" if title else "transcript.txt",
                mime="text/plain"
            )
    
    if summary:
        with col2:
            st.download_button(
                "Download Summary",
                summary,
                file_name=f"{title.split('.')[0]}_summary.txt" if title else "summary.txt",
                mime="text/plain"
            )

# Display information about the app
st.sidebar.header("About this App")
st.sidebar.markdown("""
This app uses:
- **Whisper** for audio transcription
- **BART** for text summarization
- **yt-dlp** for YouTube video processing (more reliable than pytube)
- **MoviePy** for extracting audio from video files
- **Local model storage** (.pkl files)

No API keys required! Everything runs locally.
""")

# Display debugging information
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Python Version: {sys.version}")
    st.sidebar.write(f"PyTorch Version: {torch.__version__}")
    st.sidebar.write(f"Working Directory: {os.getcwd()}")
    st.sidebar.write(f"Models Directory: {MODELS_DIR.absolute()}")
    st.sidebar.write(f"Whisper Model File Exists: {WHISPER_MODEL_PATH.exists()}")
    if WHISPER_MODEL_PATH.exists():
        st.sidebar.write(f"Whisper Model File Size: {os.path.getsize(WHISPER_MODEL_PATH) / (1024*1024):.2f} MB")
    st.sidebar.write(f"Summarizer Model File Exists: {SUMMARIZER_MODEL_PATH.exists()}")
    if SUMMARIZER_MODEL_PATH.exists():
        st.sidebar.write(f"Summarizer Model File Size: {os.path.getsize(SUMMARIZER_MODEL_PATH) / (1024*1024):.2f} MB")
