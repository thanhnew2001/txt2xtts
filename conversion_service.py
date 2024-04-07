import os
import secrets
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
# Assuming these are your custom functions

import os
import time
import torch
import torchaudio
import re

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from transformers import MarianMTModel, MarianTokenizer
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
import json
import torch
from torch.nn import functional as F
import numpy as np

import io
from pydub import AudioSegment
import base64
import uuid 

from flask import Flask, request, jsonify, make_response, send_file, render_template, Response, send_from_directory
from flask_cors import CORS

from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub

import nltk
nltk.download('punkt')
from threading import Thread
from sendmail import send_secure_email  # Ensure this is your function for sending emails

import boto3
from botocore.exceptions import NoCredentialsError

app = Flask(__name__,  static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes

# Directory where files are saved
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DOWNLOAD_FOLDER = os.path.join(app.root_path, 'downloads')
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
FILE_DIRECTORY = os.path.join(app.root_path, 'audios')
os.makedirs(FILE_DIRECTORY, exist_ok=True)

# Use this if there is no user_id sent
SPEAKER_WAV_PATH = "female_voice.wav"  # Update this path
    
# Set environment variable for Coqui TTS agreement
os.environ["COQUI_TOS_AGREED"] = "1"

# Define the model name and path
xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

model_manager = ModelManager()

# Check if the model directory exists, if not, attempt to download the model
model_path = os.path.join(get_user_data_dir("tts"), xtts_model_name.replace("/", "--"))
config_path = os.path.join(model_path, "config.json")

if not os.path.exists(model_path):
    print(f"Model directory not found. Attempting to download the model to: {model_path}")
    model_manager.download_model(xtts_model_name)
else:
    print(f"Model directory already exists: {model_path}")

# Check if the config file exists after attempting to download
if os.path.exists(config_path):
    print("Model config file found. Proceeding with model initialization.")
    
    # Initialize the model from the configuration
    config = XttsConfig()
    config.load_json(config_path)
    xtts_model = Xtts.init_from_config(config)
    xtts_model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_path, "model.pth"),
        vocab_path=os.path.join(model_path, "vocab.json"),
        checkpoint_dir= model_path,
        eval=True,
        use_deepspeed=True
    )
    print("CUDA Available:", torch.cuda.is_available())

    xtts_model.cuda()  # Use CUDA if available, else consider .to("cpu")
    
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model configuration file not found at: {config_path}")

# Dictionary to store speaker encoding latents for reuse
speaker_latents_cache = {}

import wave

def merge_wav_files(output_file_path, input_files):
    # Ensure we have files to process
    if not input_files:
        raise ValueError("No input files provided for merging.")
    
    # Open the output file
    output_wav = wave.open(output_file_path, 'wb')
    
    # Use parameters from the first input file
    with wave.open(input_files[0], 'rb') as first_wav_file:
        output_wav.setparams(first_wav_file.getparams())
    
    for wav_file in input_files:
        with wave.open(wav_file, 'rb') as input_wav:
            # Read data from input file and write it to the output file
            output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))
    
    # Ensure to close the file
    output_wav.close()

def generate_audio_mp3(prompt, language, speaker_wav_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xtts_model.to(device)
    
        prompt = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", prompt)
        
        # Check if speaker latents are already calculated
        if speaker_wav_path in speaker_latents_cache:
            gpt_cond_latent, speaker_embedding = speaker_latents_cache[speaker_wav_path]
        else:
            start_time_latents = time.time()
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=speaker_wav_path, 
                gpt_cond_len=30, 
                gpt_cond_chunk_len=4, 
                max_ref_length=60
            )
            latents_time = time.time() - start_time_latents
            # Cache the latents for future use
            speaker_latents_cache[speaker_wav_path] = (gpt_cond_latent, speaker_embedding)
        
        start_time_inference = time.time()
    
        if language == "zh":
            language = "zh-cn"
        out = xtts_model.inference(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
    
        output_fileid = f"{uuid.uuid4()}"
        output_filename = os.path.join(FILE_DIRECTORY, f"out_{output_fileid}.wav")
        torchaudio.save(output_filename, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    
        return output_filename

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xtts_supported_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko']

def check_language_existence(lang):
    if lang in xtts_supported_langs or lang == "zh":
        return True
    else:
        return False

def extract_text_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file at {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

ALLOWED_EXTENSIONS_TEXT = {'txt'}
ALLOWED_EXTENSIONS_AUDIO = {'wav'}

def allowed_file(filename, file_type):
    # Select the correct set of allowed extensions based on file_type
    if file_type.lower() == 'text':
        allowed_extensions = ALLOWED_EXTENSIONS_TEXT
    elif file_type.lower() == 'audio':
        allowed_extensions = ALLOWED_EXTENSIONS_AUDIO
    else:
        return False  # Return False if the file_type is not recognized
    
    # Check if the file extension is in the allowed set
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def clean_text(text):
    # Remove multiple consecutive spaces, newlines, and tabs with a single space
    text = re.sub(r'[\s]+', ' ', text)
    # Keep only alphabets, digits, commas, periods, and spaces; remove other characters
    text = re.sub(r'[^a-zA-Z0-9,\. ]', '', text)
    return text

def split_long_sentences(text, max_length=245):
    new_text = []
    sentences = text.split('.')
    for sentence in sentences:
        while len(sentence) > max_length:
            # Try to find a comma to split at
            split_at = sentence.rfind(',', 0, max_length)
            if split_at == -1:  # No comma found, split at max_length
                split_at = max_length
            new_text.append(sentence[:split_at])
            sentence = sentence[split_at+1:]  # +1 to skip the comma/space
        new_text.append(sentence)
    return '. '.join(new_text)

def clean_file(input, output):
    final_content = ''
    # Read the original file content
    with open(input, 'r', encoding='utf-8') as file:
        original_content = file.read()
        # Clean the entire content first
        final_content = split_long_sentences(clean_text(original_content))
    # Save the cleaned and split content to a new file
    with open(output, 'w', encoding='utf-8') as file:
        file.write(final_content)


def conversion_processing(message_body):
    try:   
        if message_body is None:
            print("Message empty")
            return False
        
        start_time = time.time()  # Record the start time

        # Extract parameters
        file_content = message_body.get('file_content')
        target_lang = message_body.get('target_lang')
        recipient_email = message_body.get('recipient_email')
        host_url = message_body.get('host_url')
        unique_id = message_body.get('unique_id')
        speaker_wav_path = message_body.get('speaker_wav_path')
            
        # Save the file_content to a file
        file_path = os.path.join(UPLOAD_FOLDER, f'{unique_id}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)
        
        print(f"File saved to {file_path}")
            # Now you can proceed with converting the text to audio and emailing the user
    
        
        clean_file(file_path, file_path + "_cleaned.txt")
        
        original_text = extract_text_from_file(file_path + "_cleaned.txt")
        target_lang = target_lang

        if not check_language_existence(target_lang):
            return jsonify({"error": f"Language '{target_lang}' not supported"}), 400

        sentences = sent_tokenize(original_text)
        print(sentences)

        wave_files = []
        for sentence in sentences:
            sentence_audio_path = generate_audio_mp3(sentence, target_lang, speaker_wav_path)
            wave_files.append(sentence_audio_path)

        output_wav_path = os.path.join(DOWNLOAD_FOLDER, unique_id + ".wav")
        merge_wav_files(output_wav_path, wave_files)
    
        print("Done!!")
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the total execution time
        print(f"Total execution time: {execution_time} seconds")

        file_name = 'output_wav_path.wav'
        bucket_name = 'xtts'
        presigned_url = upload_file_to_s3(file_name, bucket_name)
        print(presigned_url)


        # Send email notification with the download link
        email_subject = "Your audio is ready!"
        email_body = f"Your processed audio is ready. You can download it from: <a href='{presigned_url}'>here</a>"
        send_secure_email(email_subject, email_body, recipient_email, "aivideo@tad-learning.edu.vn", "thanh123!@#")
        print(f"Email sent to {recipient_email}")

    except Exception as e:
        print(e)



def upload_file_to_s3(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        # Generate a presigned URL for the uploaded file
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                         Params={'Bucket': bucket_name,
                                                                 'Key': object_name},
                                                         ExpiresIn=3600) # URL expires in 1 hour
        return presigned_url
    except NoCredentialsError:
        print("Credentials not available")
        return None


# Initialize a Boto3 client for SQS
sqs = boto3.client('sqs', region_name='ap-southeast-1')
# Specify your SQS queue URL
queue_url = 'https://sqs.ap-southeast-1.amazonaws.com/467469515596/xtts'

def read_message_from_sqs(sqs, queue_url):
    try:
        # Receive a message from the SQS queue
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,  # Limit to only one message
            WaitTimeSeconds=10  # Long polling
        )
        
        if 'Messages' in response:  # Check if message is available
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            
            # Process the message
            print("Message received:", message['Body'])
            
            # Remove the message from the queue to prevent it from being read again
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            return json.loads(message['Body'])
        else:
            print("No messages")
            return None
    except Exception as e:
        print("Error reading message:", e)
        return None

# Usage example
# file_name = 'female_voice.wav'
# bucket_name = 'xtts'
# presigned_url = upload_file_to_s3(file_name, bucket_name)
# print(presigned_url)

# start the main program #
# Assuming message_body is the JSON string you received from SQS
message_body = read_message_from_sqs(sqs, queue_url)  # This calls the function from the previous example

print(message_body)

conversion_processing(message_body)
