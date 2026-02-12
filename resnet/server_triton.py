from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import time
import asyncio
import numpy as np

# Triton gRPC 비동기 클라이언트
import tritonclient.grpc.aio as grpcclient
import os

# 설정
MODEL_NAME = os.getenv("MODEL_NAME", "resnet_onnx")  # Triton에 등록한 모델 이름
TRITON_URL = "localhost:8001"
INPUT_NAME = "input"        # ONNX export 시 지정한 입력 이름
OUTPUT_NAME = "output"      # ONNX export 시 지정한 출력 이름

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

app = FastAPI()
triton_client = None

@app.on_event("startup")
async def startup_event():
    global triton_client
    print(f"Connecting to Triton at {TRITON_URL}...")
    try:
        # 비동기 gRPC 클라이언트 생성
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        if await triton_client.is_server_ready():
            print("Triton Server is Ready!")
        else:
            print("Triton Server is connected but not ready.")
    except Exception as e:
        print(f"Failed to connect: {e}")

# 전처리를 별도 함수로 분리 (비동기)
def preprocess_batch(image_bytes_list: List[bytes]) -> np.ndarray:
    batch_tensors = []
    for data in image_bytes_list:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        # PyTorch Transform 그대로 사용
        batch_tensors.append(transform(image))
    
    # (B, C, H, W) Tensor 생성
    # 주의: Triton에 보낼 때는 GPU(.to(device))로 보내지 않고 CPU 상태로 둡니다.
    tensor = torch.stack(batch_tensors)
    
    # Triton 전송을 위해 Numpy로 변환
    return tensor.numpy()

@app.post("/predict_batch")
async def predict(files: List[UploadFile] = File(...)):
    req_arrival_time = time.time()

    # 파일 읽기 (비동기 IO)
    image_bytes_list = []
    for file in files:
        image_bytes_list.append(await file.read())

    # 전처리 (CPU 작업)
    # 기존 로직을 run_in_executor로 감싸서 비동기성을 유지합니다.
    # 이렇게 안 하면 이미지 처리하는 동안 서버가 멈춥니다.
    loop = asyncio.get_running_loop()
    input_numpy = await loop.run_in_executor(None, preprocess_batch, image_bytes_list)
    
    preprocess_done_time = time.time()

    # Triton 요청 생성
    inputs = [grpcclient.InferInput(INPUT_NAME, input_numpy.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_numpy)
    
    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    # Triton 추론
    # 큐에 넣는 대신 바로 Triton에게 비동기로 보내고 대기
    response = await triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    
    # 결과 파싱
    output_data = response.as_numpy(OUTPUT_NAME)
    predictions = np.argmax(output_data, axis=1).tolist()
    
    total_time = time.time() - req_arrival_time
    inference_time = time.time() - preprocess_done_time # 네트워크 + 순수 추론 시간

    return {
        "batch_size": len(files),
        "predictions": predictions,
        "server_inference_time_sec": inference_time, # Triton 왕복 시간
        "queue_wait_time_sec": 0.0,                  # Triton이 알아서 관리하므로 여기선 0
        "total_processing_time_sec": total_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)