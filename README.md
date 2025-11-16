🎥 YouTube Video Q&A Bot

This project allows you to download a YouTube video, extract audio, transcribe it into text, embed the transcript into a vector store, and ask questions about the video content using LangChain and Groq LLMs.

🚀 Features

📥 Download YouTube videos and extract audio (yt-dlp).

🎙 Transcribe audio to text using Groq Whisper (whisper-large-v3-turbo).

📑 Create LangChain documents from transcripts.

🧠 Embed transcripts using FastEmbed (BAAI/bge-small-en-v1.5).

🤖 Question Answering (QA) on transcripts using Groq LLM (llama3-8b-8192).

📦 Requirements

Install dependencies:

pip install yt-dlp openai httpx requests python-dotenv groq tiktoken langchain docarray

⚙️ Setup

Clone this repository:

git clone https://github.com/Mohammed-Taha20/Transcribe-Ask

cd yt-video-qa


Install dependencies (see above).

Set up environment variables:
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key


Note: Ensure you have FFmpeg installed and update its path in the script


Run the script:

python main.py

