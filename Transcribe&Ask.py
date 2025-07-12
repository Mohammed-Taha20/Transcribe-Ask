import doctest
import os
import glob
from httpx import Client
import openai
import yt_dlp as youtube_dl
from yt_dlp import DownloadError
import docarray
import requests
from dotenv import load_dotenv
from groq import Groq
import tiktoken
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import FastEmbedEmbeddings



def downloading_video(youtube_url):    
    # Directory to store the downloaded video
    output_dir = r".\files\video"
    
    # Config for youtube-dl
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
                
            }
        ],
        "ffmpeg_location": r"E:\programs\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True
    }
    # check for the directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print that the video is being downloaded
    print(f"Downloading video from {youtube_url}...")
    
    # catch some errors during the download
    try :
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
                print("Download failed, trying again...")
                ydl.download([youtube_url])
        

def get_audio_file(youtube_URL):
    downloading_video(youtube_URL)
    # find all audio files
    output_dir = r".\files\audio"
    audio_files = glob.glob(os.path.join(output_dir, "*.mp3"))
    # get my audio file, if no audio files found, return None
    my_audio_file = audio_files[0] if audio_files else None
    
    print (my_audio_file)
    return my_audio_file


def transcribe(youtube_URL):
    audio_file = get_audio_file(youtube_URL)
    output_dir = r".\files\transcribe\transcribe.txt"
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    
    # transcribe the audio file
    print ("converting the audio file to text file...")
    with open(audio_file, "rb") as audio:
        transcription = client.audio.transcriptions.create(
        file=(audio_file, audio.read()),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        )

        result = transcription.text
        print(" Full response JSON:", result)
        transcript = result

    # write the transcript to a text file
    if output_dir is not None:
        with open (output_dir,"w",encoding="utf-8") as f :
            f.write(transcript)

    print ("Transcription complete. Transcript saved to", output_dir)
    return transcript

def Create_docs():
    loader = TextLoader(r".\files\transcribe\transcribe.txt")
    docs = loader.load()  # LangChain documents
    print("Documents created from the transcript.")
    return docs

def Embed_docs():
    docs = Create_docs()

    embedding = FastEmbedEmbeddings(  # Uses BAAI/bge-small-en-v1.5 by default
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir="E:/my_models/"
    )

    dp = DocArrayInMemorySearch.from_documents(docs, embedding)
    return dp

def QA_chain():
    dp = Embed_docs()
    retriever = dp.as_retriever()
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_stuff

def answer_question(question):
    qa_stuff = QA_chain()
    query = question
    response = qa_stuff.invoke({"query": question})
    print("Answer:", response["result"])

if __name__ == "__main__":
    youtube_URL = input("Enter the YouTube URL: ")
    load_dotenv()
    transcribe(youtube_URL)
    while True:
        question = input("what are your question about this video")
    answer_question (question)


