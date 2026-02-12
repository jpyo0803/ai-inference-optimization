from fastapi import FastAPI, UploadFile, File
import torch
from contextlib import asynccontextmanager
import torchvision.transforms as transforms
from model import ResNet50
from PIL import Image
import io
import time
import os
import asyncio
from typing import List

# GPU Support 필요
assert torch.cuda.is_available(), "GPU is not available"
print("GPU is available. Using GPU for inference.")

# 전역 변수로 모델과 디바이스 설정
model = None
transform = None
request_queue = None # (Input Tensor, Future Object)

QUEUE_SIZE = 1000
DEVICE = torch.device("cuda")

async def inference_worker():
    print("Background inference worker started.")

    while True:
        input_tensor, response_future, req_arrival_time = await request_queue.get()

        try:
            with torch.no_grad():
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, dim=1)

                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
            
            result = {
                "predictions": predicted.cpu().tolist(),
                "inference_time": inference_time,
                "queue_time": start_time - req_arrival_time
            }

            if not response_future.done():
                response_future.set_result(result)
        except Exception as e:
            if not response_future.done():
                response_future.set_exception(e)
        finally:
            request_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, request_queue

    model = ResNet50(num_classes=10)
    weight_path = 'weight/resnet50.pth'

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
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])

    print(f"Model loaded successfully on {DEVICE}.")

    request_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    worker_task = asyncio.create_task(inference_worker())

    yield
    print("Shutting down server.")

app = FastAPI(lifespan=lifespan)

@app.post("/predict_batch")
async def predict(files: List[UploadFile] = File(...)):
    '''
        클라이언트로부터 배치 요청을 받아 전처리 후 큐에 넣고 기다림 (비동기)
    '''

    req_arrival_time = time.time()

    batch_tensors = []
    for file in files:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        batch_tensors.append(transform(image)) # 전처리 후 리스트에 추가

    # (B, C, H, W) 형태의 배치 텐서 생성
    input_tensor = torch.stack(batch_tensors).to(DEVICE)

    loop = asyncio.get_running_loop()
    response_future = loop.create_future()

    await request_queue.put((input_tensor, response_future, req_arrival_time))
    result = await response_future

    total_time = time.time() - req_arrival_time

    return {
        "batch_size": len(files),
        "predictions": result["predictions"],
        "server_inference_time_sec": result["inference_time"],
        "queue_wait_time_sec": result["queue_time"],
        "total_processing_time_sec": total_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)