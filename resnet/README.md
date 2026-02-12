## ResNet-50 최적화

### 성능 실험 1: CIFAR-10 Test Dataset
**Environment:** NVIDIA GeForce RTX 4060 Ti, Triton Inference Server 24.12
| 최적화 기법 | Accuracy (%) | Avg Inference Latency (ms) | Model Size (MB) | 모델 경량화 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (PyTorch)** | 81.55% | 2.41 ms | 91 MB | FP32 |
| **ONNX + Triton** | 81.50% | 1.92 ms | 91 MB | FP32 |
| **TensorRT + Triton** | 81.48% | 1.12 ms | 48 MB | FP16 |

## 실행방법
0. **사전준비**
    ```sh
    $ pip install -r requirements.txt
    ```

1. **ResNet-50 학습하기 (완료후에 weight 디렉토리에 ```resnet50.pth``` 가중치 저장)**
    ```sh
    $ python train.py
    ```
2. **ONNX 변환 (model_repository/resnet_onnx/1에 model.onnx와 model.onnx.data 생성)**
    ```sh
    $ python export_onnx.py
    ```
3. **TensorRT 변환 (model_repository/resnet_trt/1에 model.plan 생성)**
    ```sh
    $ docker run --gpus all --rm -it \
        -v $(pwd):/workspace \
        -w /workspace \
        nvcr.io/nvidia/tritonserver:24.12-py3 \
        /usr/src/tensorrt/bin/trtexec \
        --onnx=model_repository/resnet_onnx/1/model.onnx \
        --saveEngine=model_repository/resnet_trt/1/model.plan \
        --fp16 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:8x3x32x32 \
        --maxShapes=input:16x3x32x32
    ```
4. **추론 서버 실행**
    - Baseline 서버 모드 (FastAPI Endpoint에서 모델 직접 서빙)
        ```sh
        $ cd docker && docker compose up server-baseline -d
        ```
    - ONNX + Triton 서버 모드 (도커 컨테이너 내부에서 Triton Inference 서버와 FastAPI Endpoint 동시 실행)
        ```sh
        $ cd docker && docker compose up server-onnx-triton -d
        ```
    - TensorRT + Triton 서버 모드 (도커 컨테이너 내부에서 Triton Inference 서버와 FastAPI Endpoint 동시 실행)
        ```sh
        $ cd docker && docker compose up server-trt-triton -d
        ```
5. **제대로 모델 서빙이 준비되어있는지 확인 (성능 실험 1)**
    ```sh
    $ python clinet_tester.py
    ```
