
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

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

# 검증할 이미지 파일 경로
input_image_path = "D:/validation_data_n_775308/ai/6_gray.png"   # 이미지 파일 경로를 적절히 수정해주세요.

# 이미지 예측 수행 및 결과 출력
prediction = predict_image(input_image_path, model)
print(f"Image: {input_image_path}, Prediction: {prediction}%")
