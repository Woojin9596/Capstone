"""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'This is Home!'

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug = True)
"""   


# 진짜 코드 #
from flask import Flask, request, jsonify
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import pydub
from flask_cors import CORS
import requests

# Matplotlib의 백엔드를 설정
import matplotlib
matplotlib.use('agg')

app = Flask(__name__)
CORS(app)

# uploads 폴더를 애플리케이션 루트에 생성
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# VGG19 모델 정의
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 마지막 512 채널 레이어들을 제거
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# cap.pth 모델 불러오기
model = VGG19()
model.load_state_dict(torch.load("D:/noise1.pth", map_location=torch.device('cpu')))
model.eval()

# 이미지 전처리 함수 정의
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image {image_path}")

    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

# 이미지 예측을 수행하고 결과를 반환하는 함수
def predict_image(image_path, model):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        prediction = round(output.item() * 100, 2)  # 예측값에 100을 곱하고, 소수점 2자리까지 반올림
    return prediction

## mel 변환
def create_gray_mel_spectrogram(audio_file, output_image_gray):
    # 샘플링 주파수
    sampling_rate = 16000
    # FFT 윈도우 크기
    n_fft = 400
    # hop_length
    hop_length = 160
    # Mel 밴드의 수
    n_mels = 128
    # 최대 주파수
    fmax = 16000
    try:
        # WAV 파일 로드
        y, sr = librosa.load(audio_file, sr=sampling_rate)
        # Mel-spectrogram 생성
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
        # Mel-spectrogram을 데시벨 단위로 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 그레이 스케일로 변환
        mel_spec_db_gray = np.flip((mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) * 255.0, axis=0).astype(np.uint8)
        # Matplotlib figure 생성
        plt.figure(figsize=(10, 4))
        plt.axis('off')  # 축 제거
        # 그레이 스케일 Mel spectrogram 표시
        plt.imshow(mel_spec_db_gray, cmap='gray', aspect='auto', origin='upper', extent=[0, len(y) / sr, 0, fmax])
        # 이미지 저장
        plt.savefig(output_image_gray, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"An error occurred: {e}")

# m4a to wav
def convert_to_wav(input_file, output_file):
    try:
        sound = pydub.AudioSegment.from_file(input_file, format="m4a")
        sound.export(output_file, format="wav")
        return output_file
    except Exception as e:
        print(f"An error occurred while converting to WAV: {e}")
        return None

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        
        # 요청 로깅
        print("Received a request to process audio.")
        
        if 'audio' not in request.files:
            return 'No file part', 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return 'No selected file', 400
        
        # 파일 정보 출력
        print('Selected file:')
        print('Name:', audio_file.filename)  # 선택된 파일의 이름
        print('Type:', audio_file.mimetype)  # 선택된 파일의 MIME 타입
        
        # 파일을 업로드 폴더에 저장
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)
        
        # Mel spectrogram 이미지를 저장할 경로 설정
        output_image_gray = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(audio_file.filename)[0]}_mel_spec_gray.png")
        
        # Mel spectrogram 생성 및 저장
        create_gray_mel_spectrogram(audio_path, output_image_gray)    

        # flask --> flutter로 전달할 때 사용!!  (성공적으로 처리되었다는 응답 반환)
        # return jsonify({'result': 'success', 'image_path': output_image_gray})
        
        # 성공적으로 처리되었다는 응답 반환
        response_data = {'result': 'success', 'image_path': output_image_gray}
        print("Response data:", response_data)  # 응답 데이터 출력
        #print(output_image_gray)
        #return jsonify(response_data)
        
        input_image_path = output_image_gray
        print(input_image_path)     
        # 이미지 예측 수행 및 결과 출력
        prediction = predict_image(input_image_path, model)
        print(f"Image: {input_image_path}, Prediction: {prediction}%")           
    
        return jsonify({"result": "AI: " + str(prediction) + '%'})
        
                
    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    try:
        data = request.get_json()
        message = data.get('message', '')
        print('Received message from Expo:', message)
        # 받은 메시지를 그대로 다시 클라이언트로 보내기
        return jsonify({'message': f'Server received message: {message}'}), 200
    except Exception as e:
        print('An error occurred:', e)
        return jsonify({'error': 'An error occurred on the server'}), 500

if __name__ == '__main__':
    app.run(host='172.17.75.62', port=5000, debug = True)
    





"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/endpoint', methods=['POST'])
def handle_post_request():
    data = request.json  # 요청으로부터 JSON 데이터 추출
    print('클라이언트로부터의 메시지:', data['message'])
    # 클라이언트로 응답 전송
    return jsonify({'message': 'Hello from Flask!'})

if __name__ == '__main__':
    app.run(debug=True)
"""
