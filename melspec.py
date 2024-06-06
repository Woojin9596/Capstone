import os
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        
# 입력 폴더와 출력 폴더 경로 설정
input_folder_ai = r"D:\voice_sample"
output_folder_ai = r"I:\내 드라이브\Capstone"

# 출력 폴더가 없으면 생성합니다.
# if not os.path.exists(output_folder_real):
#     os.makedirs(output_folder_real)

if not os.path.exists(output_folder_ai):
    os.makedirs(output_folder_ai)

def process_files(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.mp3') or filename.endswith('.wav'):
                audio_file = os.path.join(root, filename)
                output_image_gray = os.path.join(output_folder_ai, filename.split('.')[0] + '_gray.png')
                create_gray_mel_spectrogram(audio_file, output_image_gray)

# process_files 함수를 사용하여 real 데이터셋 변환 및 저장
# process_files(input_folder_real, output_folder_real)

# process_files 함수를 사용하여 ai 데이터셋 변환 및 저장
process_files(input_folder_ai, output_folder_ai)
