import os
import time
import re
import json
import io
import base64
import uuid 
import secrets
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify, make_response, send_file, render_template, Response, send_from_directory
from flask_cors import CORS
import nltk
import boto3
from botocore.exceptions import NoCredentialsError

nltk.download('punkt')

app = Flask(__name__,  static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for all routes

ALLOWED_EXTENSIONS_TEXT = {'txt'}
ALLOWED_EXTENSIONS_AUDIO = {'wav'}

# Directory where files are saved
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# Initialize a Boto3 client for SQS
sqs = boto3.client('sqs', region_name='ap-southeast-1')
# Specify your SQS queue URL
queue_url = 'https://sqs.ap-southeast-1.amazonaws.com/467469515596/xtts'

def send_to_sqs(queue_url, message_body):
    # Create an SQS client
    # Send the message to SQS
    response = sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
    return response

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
        
@app.route('/upload_speech', methods=['POST'])
def upload_speech():
    # Get text file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, 'text'):
        return jsonify({"error": "No selected file or file type not allowed"}), 400

    # Generate a random string to append to the file name
    random_str = secrets.token_hex(8)
    
    # Secure the filename and append the random string
    filename_secure = secure_filename(file.filename)
    filename_base, file_extension = os.path.splitext(filename_secure)
    filename_randomized = f"{filename_base}_{random_str}{file_extension}"
    filepath = os.path.join(UPLOAD_FOLDER, filename_randomized)
    file.save(filepath)

     # Check for an optional voice file
    voice_file = request.files.get('voice_file')
    own_voice_url = None
    if voice_file and allowed_file(voice_file.filename, 'audio'):
        # Process the voice file similarly, using a different directory if needed
        voice_filename_secure = secure_filename(voice_file.filename)
        voice_filepath = os.path.join(UPLOAD_FOLDER, voice_filename_secure)
        voice_file.save(voice_filepath)
        speaker_wav_path = 'own_voice'
        # upload the voice to s3
        own_voice_url = upload_file_to_s3(voice_filepath, "xtts")
        
    else:
        # Fallback to a default or selected voice option
        speaker_wav_path = request.form.get("voice", "female_voice.wav")  # Use default path or form option

    
    # Pass the captured host URL to the background thread
    recipient_email = request.form.get('recipient_email')
    host_url = request.host_url  # Capture the host URL
    target_lang = request.form.get("target_lang", "en")

    # Read the text file content
    with open(filepath, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    # Construct the message including the file's content
    message_body = json.dumps({
        'file_content': file_content,  # Add the file content directly
        'target_lang': target_lang,
        'recipient_email': recipient_email,
        'host_url': request.host_url,
        'unique_id': random_str,
        'speaker_wav_path': speaker_wav_path,
        'own_voice_url': own_voice_url
    })
            
    # Send the message to SQS
    response = send_to_sqs(queue_url, message_body)
    print(f"Message sent to SQS with ID: {response['MessageId']}")

    return jsonify({'message': 'Your audio is being processed. You will receive the result in your mailbox.'})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
