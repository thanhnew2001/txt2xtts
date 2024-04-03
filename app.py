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


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Directory where files are saved
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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

def generate_audio_mp3(prompt, language, speaker_wav_path):
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

    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(output_filename)

          # Convert the audio to MP3 and save it directly to a file
        audio.export(output_filename + ".mp3", format="mp3", bitrate="22k")
        return None
    
        # Convert the audio to MP3 and store in a BytesIO object
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3", bitrate="22k")
        mp3_io.seek(0)  # Go to the beginning of the BytesIO object

        # Encode the MP3 data as a base64 string
        mp3_data = mp3_io.getvalue()
        mp3_base64 = base64.b64encode(mp3_data).decode('utf-8')
 
        if speaker_wav_path != SPEAKER_WAV_PATH:           
            # os.remove(os.path.join(FILE_DIRECTORY, output_filename))
            a = 1
        return mp3_base64

    except Exception as e:
        print(f"An error occurred: {e}")
        if speaker_wav_path != SPEAKER_WAV_PATH:
            # os.remove(os.path.join(FILE_DIRECTORY, output_filename))
            a = 1
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

    original_text = extract_text_from_file(filepath)
    target_lang = request.form.get("target_lang", "en")

    if not check_language_existence(target_lang):
        return jsonify({"error": f"Language '{target_lang}' not supported"}), 400

    sentences = sent_tokenize(original_text)
    print(sentences)
    combined_audio = AudioSegment.empty()

    for sentence in sentences:
        sentence_audio_path = generate_audio_mp3(sentence, target_lang, SPEAKER_WAV_PATH)
        #sentence_audio = AudioSegment.from_mp3(sentence_audio_path)
        #combined_audio += sentence_audio
        #os.remove(sentence_audio_path)  # Cleanup individual sentence audio files

    # Output file name includes the original file name plus the random string
    
    #output_filename = f"{filename_base}_{random_str}.mp3"
    #output_mp3_path = os.path.join(FILE_DIRECTORY, output_filename)
    #combined_audio.export(output_mp3_path, format="mp3")

    #return send_file(output_mp3_path, as_attachment=True, attachment_filename=output_filename)
    return "Hello"

if __name__ == '__main__':
    app.run(debug=True, port=5002)
