import os
import secrets
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
# Assuming these are your custom functions
from your_module import generate_audio_mp3, check_language_existence, SPEAKER_WAV_PATH


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
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_randomized)
    file.save(filepath)

    original_text = extract_text_from_file(filepath)
    target_lang = request.form.get("target_lang", "en")

    if not check_language_existence(target_lang):
        return jsonify({"error": f"Language '{target_lang}' not supported"}), 400

    sentences = sent_tokenize(original_text)
    combined_audio = AudioSegment.empty()

    for sentence in sentences:
        sentence_audio_path = generate_audio_mp3(sentence, target_lang, SPEAKER_WAV_PATH)
        sentence_audio = AudioSegment.from_mp3(sentence_audio_path)
        combined_audio += sentence_audio
        os.remove(sentence_audio_path)  # Cleanup individual sentence audio files

    # Output file name includes the original file name plus the random string
    output_filename = f"{filename_base}_{random_str}.mp3"
    output_mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    combined_audio.export(output_mp3_path, format="mp3")

    return send_file(output_mp3_path, as_attachment=True, attachment_filename=output_filename)
