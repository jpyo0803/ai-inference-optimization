from fastapi import FastAPI, UploadFile, File
import torch
from contextlib import asynccontextmanager
import torchvision.transforms as transforms
from model import ResNet_CIFAR
from PIL import Image
import io
import time
import os

# GPU Support 필요
assert torch.cuda.is_available(), "GPU is not available"
print("GPU is available. Using GPU for inference.")

# 전역 변수로 모델과 디바이스 설정
model = None
transform = None

DEVICE = torch.device("cuda")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform

    model = ResNet_CIFAR(depth=20)
    weight_path = 'weight/resnet_20.pth'

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {weight_path}.")
    else:
        raise FileNotFoundError(f"Weight file not found at {weight_path}.")

    model.to(DEVICE)
    model.eval()

    # 전처리 변환 설정
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print(f"Model loaded successfully on {DEVICE}.")
    yield
    print("Shutting down server.")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

    end_time = time.time()

    inference_time = end_time - start_time

    return {
        "class_id": int(predicted.item()),
        "inference_time_sec": inference_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)