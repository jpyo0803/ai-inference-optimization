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
3. 제대로 모델 서빙이 준비되어있는지 확인 (새로운 터미널에서)
    ```sh
    $ python clinet_tester.py
    ```
    실행 결과는 다음과 같이 나옴
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