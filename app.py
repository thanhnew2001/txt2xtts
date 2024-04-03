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
SPEAKER_WAV_PATH = "giongnu.wav"  # Update this path
    
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

ALLOWED_EXTENSIONS = {'txt'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/download_audio/<unique_id>', methods=['GET'])
def download_audio(unique_id):
    """Endpoint to download the audio."""
    audio_path = os.path.join(DOWNLOAD_FOLDER, unique_id + ".wav")

    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    else:
        return jsonify({'error': 'Video not found.'}), 404

def background_processing(filepath, target_lang, recipient_email, host_url, unique_id):   
    start_time = time.time()  # Record the start time
    original_text = extract_text_from_file(filepath)
    target_lang = target_lang

    if not check_language_existence(target_lang):
        return jsonify({"error": f"Language '{target_lang}' not supported"}), 400

    sentences = sent_tokenize(original_text)
    print(sentences)

    wave_files = []
    for sentence in sentences:
        sentence_audio_path = generate_audio_mp3(sentence, target_lang, SPEAKER_WAV_PATH)
        wave_files.append(sentence_audio_path)

    output_wav_path = os.path.join(DOWNLOAD_FOLDER, unique_id + ".wav")
    merge_wav_files(output_wav_path, wave_files)
   
    print("Done!!")
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the total execution time
    print(f"Total execution time: {execution_time} seconds")
    # Construct the download link using the passed host_url
    download_link = f"{host_url}download_audio/{unique_id}" 

    # Send email notification with the download link
    email_subject = "Your audio is ready!"
    email_body = f"Your processed audio is ready. You can download it from: {download_link}"
    send_secure_email(email_subject, email_body, recipient_email, "aivideo@tad-learning.edu.vn", "thanh123!@#")
    print(f"Email sent to {recipient_email}")

@app.route('/upload_speech', methods=['POST'])
def upload_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "No selected file or file type not allowed"}), 400

    # Generate a random string to append to the file name
    random_str = secrets.token_hex(8)
    
    # Secure the filename and append the random string
    filename_secure = secure_filename(file.filename)
    filename_base, file_extension = os.path.splitext(filename_secure)
    filename_randomized = f"{filename_base}_{random_str}{file_extension}"
    filepath = os.path.join(UPLOAD_FOLDER, filename_randomized)
    file.save(filepath)

    # Pass the captured host URL to the background thread
    recipient_email = request.form.get('recipient_email')
    host_url = request.host_url  # Capture the host URL
    target_lang = request.form.get("target_lang", "en")

    Thread(target=background_processing, args=(filepath, target_lang, recipient_email, host_url, random_str)).start()
    return jsonify({'message': 'Your audio is being processed. You will receive the result in your mailbox.'})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
