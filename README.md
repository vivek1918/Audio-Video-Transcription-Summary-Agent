# 🎧 Audio/Video Transcription & Summary Agent

A Streamlit web application that transcribes audio/video content and generates concise summaries using AI models.

## Features

- **Multiple Input Options**:
  - Upload audio files (MP3, WAV, FLAC, OGG, M4A)
  - Upload video files (MP4, AVI, MOV)
  - Provide YouTube URL for direct processing
  
- **AI-powered Processing**:
  - Audio transcription using OpenAI's Whisper model
  - Text summarization using Facebook's BART model
  - No API keys required - all processing happens locally

- **User-friendly Interface**:
  - Simple file upload or URL input
  - Progress tracking with status updates
  - Download options for both transcripts and summaries

- **Efficient Processing**:
  - Local model caching for faster subsequent runs
  - Chunked processing of long content
  - Audio extraction from video files

## Requirements

- Python 3.7+
- FFmpeg (required for audio processing)
- GPU recommended but not required (CPU fallback available)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/vivek1918/Audio-Video-Transcription-Summary-Agent.git
   cd AI_AGENT
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Or install the following packages:
   ```
   pip install streamlit torch transformers openai-whisper yt-dlp moviepy
   ```

3. Install FFmpeg:
   - Windows: Download from (https://www.gyan.dev/ffmpeg/builds/) and add to PATH
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Choose your input method:
   - Enter a YouTube URL
   - Upload an audio or video file

4. Click "Process" and wait for the transcription and summarization to complete

5. View and download the results

## How It Works

1. **Input Processing**:
   - For YouTube URLs: Audio is extracted using yt-dlp
   - For video files: Audio is extracted using MoviePy
   - Audio files are processed directly

2. **Transcription**:
   - The Whisper model converts speech to text
   - Processed in one pass for shorter content

3. **Summarization**:
   - Long transcripts are split into manageable chunks
   - Each chunk is summarized using the BART model
   - Chunk summaries are combined and summarized again if needed

4. **Model Management**:
   - Models are downloaded on first run and cached locally
   - Cached models are loaded on subsequent runs for faster startup

## Troubleshooting

- **FFmpeg Issues**: Make sure FFmpeg is properly installed and available in your PATH
- **Memory Errors**: If processing large files causes memory errors, try:
  - Using a smaller Whisper model (tiny or base)
  - Processing shorter audio segments
- **Download Issues**: If YouTube download fails, try updating yt-dlp:
  ```
  pip install -U yt-dlp
  ```
- **Model Loading Fails**: Check your internet connection and ensure you have sufficient disk space

## Advanced Configuration

You can modify the following parameters in the code:

- Whisper model size: Change `whisper.load_model("tiny")` to "base", "small", or "medium" for better accuracy (at the cost of speed)
- Summary length: Adjust `max_length` and `min_length` parameters in the `summarize_text` function
- Chunk size: Modify `max_chunk_size` in the `chunk_text` function

## Performance Considerations

- First run will be slower as models need to be downloaded
- GPU acceleration significantly improves transcription speed
- Processing time depends on file length and model size
- Consider using smaller models for faster results or larger models for better accuracy

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription capability
- [Hugging Face Transformers](https://huggingface.co/facebook/bart-large-cnn) for the BART summarization model
- [Streamlit](https://streamlit.io/) for the web interface framework
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video processing
- [MoviePy](https://zulko.github.io/moviepy/) for video manipulation
