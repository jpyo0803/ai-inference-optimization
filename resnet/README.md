## ResNet-50 최적화

## 성능 벤치마크

## 실행방법
0. 사전준비
    ```sh
    $ pip install -r requirements.txt
    ```

1. ResNet-50 학습하기 (완료후에 weight 디렉토리에 ```resnet50.pth``` 가중치 저장)
    ```sh
    $ python train.py
    ```
2. Server 실행
    - Baseline 서버 실행 모드
        ```sh
        $ python server_baseline_gpu.py
        ```
    - **ONNX + Triton** Inference 서버 모드
        a. ONNX 모델로 변환
        ```sh
        $ python export_onnx.py
        ```
        b. Triton Inference 서버 실행 (Docker 필요, gRPC 포트 사용)
        ```sh
        $ docker run --gpus all --rm \
            -p 8001:8001 -p 8002:8002 \
            -v $(pwd)/model_repository:/models \
            nvcr.io/nvidia/tritonserver:24.12-py3 \
            tritonserver --model-repository=/models
        ```
        c. Endpoint 서버 실행 (새로운 터미널)
        ```sh
        $ python server_onnx_triton_py
        ```
    - ONNX + **TensorRT** + Triton Inference 서버 모드
        a. TensorRT 변환 (.plan 생성)
        ```sh
        docker run --gpus all --rm -it \
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
        b. Triton Inference 서버 실행 (Docker 필요, gRPC 포트 사용)
        ```sh
        $ docker run --gpus all --rm \
            -p 8001:8001 -p 8002:8002 \
            -v $(pwd)/model_repository:/models \
            nvcr.io/nvidia/tritonserver:24.12-py3 \
            tritonserver --model-repository=/models
        ```
        c. Endpoint 서버 실행 (새로운 터미널)
        ```sh
        $ python server_trt_triton_py
        ```
3. 제대로 모델 서빙이 준비되어있는지 확인 (새로운 터미널에서)
    ```sh
    $ python clinet_tester.py
    ```
    실행 결과는 다음과 같음
    ```sh
    ========================================
    Total Images Processed: 10000
    Accuracy: 81.50%
    Avg Server Inference Time (Pure GPU): 2.15 ms
    Avg End-to-End Latency (Network + Inference): 3.45 ms
    ========================================
    ``` 
4. locust를 활용한 실제 부하 테스트 (-u 옵션을 통해 유저수 조절)
    ```sh
    python -m locust -f locustfile.py --headless -u 1 -r 1 -t 30s --host http://localhost:8000
    ```
