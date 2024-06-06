import re  # Used for regular expressions
import requests  # Used for making HTTP requests
import json  # Used for working with JSON data
import os  # Used for handling file paths

# Define constants for the script
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
XI_API_KEY = ""  # Your API key for authentication
VOICE_ID = ""  # ID of the voice model to use

# Define the path to the text file containing the data
TEXT_FILE_PATH = r"D:\voice_sample\UNZIP\female_text\5247_G2A4E7_LSM.txt"

# Define the directory where you want to save the output MP3 files
OUTPUT_DIRECTORY = r"D:\voice_sample\AIvoice4"
# Ensure the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# Open the text file and read its contents
with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Iterate over the lines and generate AI voice for each text
for idx, text in enumerate(lines, start=1):
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Construct the URL for the Text-to-Speech API request
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

    # Set up headers for the API request, including the API key for authentication
    headers = {
        "Accept": "application/json",
        "xi-api-key": XI_API_KEY
    }

    # Set up the data payload for the API request, including the text and voice settings
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    # Make the POST request to the TTS API with headers and data, enabling streaming response
    response = requests.post(tts_url, headers=headers, json=data, stream=True)

    # Check if the request was successful
    if response.ok:
        # Open the output file in write-binary mode
        output_path = os.path.join(OUTPUT_DIRECTORY, f"5247_G2A4E7_LSM_{idx:08}.mp3")
        with open(output_path, "wb") as f:
            # Read the response in chunks and write to the file
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        # Inform the user of success
        print(f"Audio stream for {idx:08} saved successfully.")
    else:
        # Print the error message if the request was not successful
        print(f"Error processing {idx:08}: {response.text}")

print("작업이 완료되었습니다.")
