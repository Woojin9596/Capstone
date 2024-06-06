import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
model.load_state_dict(torch.load(r"D:\noise1.pth", map_location=torch.device('cpu')))
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

# 검증할 이미지 폴더 경로
input_folder_path_ai = r"D:\validation_data_n_775308\naver_ai"
input_folder_path_real = r"D:\validation_data_n_775308\noise_dataset"

# 각 이미지에 대한 예측값과 실제값 저장
predictions = []
true_labels = []

# AI 데이터셋 검증
image_paths_ai = [os.path.join(input_folder_path_ai, filename) for filename in os.listdir(input_folder_path_ai) if filename.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_paths_ai:
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        prediction = output.item()  # 시그모이드 함수 호출을 제거
    predictions.append(prediction)
    true_labels.append(1)  # AI 데이터셋의 실제 레이블은 1

# Real 데이터셋 검증
image_paths_real = [os.path.join(input_folder_path_real, filename) for filename in os.listdir(input_folder_path_real) if filename.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_paths_real:
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
        prediction = output.item()  # 시그모이드 함수 호출을 제거
    predictions.append(prediction)
    true_labels.append(0)  # Real 데이터셋의 실제 레이블은 0

# 이진 분류 지표 계산
accuracy = accuracy_score(true_labels, [1 if pred >= 0.5 else 0 for pred in predictions])
precision = precision_score(true_labels, [1 if pred >= 0.5 else 0 for pred in predictions])
recall = recall_score(true_labels, [1 if pred >= 0.5 else 0 for pred in predictions])
f1 = f1_score(true_labels, [1 if pred >= 0.5 else 0 for pred in predictions])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
