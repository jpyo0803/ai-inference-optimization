import requests
import torchvision
import time
from tqdm import tqdm
import io

# 서버 주소
URL = "http://localhost:8000/predict_batch"

def run_test():
    # 테스트 데이터셋 로드 (예시: CIFAR10)
    # 실제로는 서버 모델과 클래스가 맞는 데이터셋을 사용해야 합니다.
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True
    )
    
    correct = 0
    total = 0
    total_server_inference_time = 0.0 # 서버 내부 순수 추론 시간 합
    total_e2e_time = 0.0              # 네트워크 포함 전체 걸린 시간 합

    print(f"Start testing {len(test_dataset)} images via API...")

    # 루프를 돌며 요청 전송
    for i in tqdm(range(len(test_dataset))):
        image, label = test_dataset[i] # image는 PIL 객체
        
        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # 요청 시작 시간
        req_start = time.time()
        
        # POST 요청
        try:
            files = [('files', ('image.jpg', img_bytes, 'image/jpeg'))]
            response = requests.post(URL, files=files)
            response.raise_for_status() # 에러 체크
            
            result = response.json()
            
            # 응답 파싱
            pred_id = result['predictions'][0]
            server_time = result['server_inference_time_sec']
            
            # 정확도 체크 (주의: 모델의 클래스 인덱스와 데이터셋 라벨이 일치한다고 가정)
            if pred_id == label:
                correct += 1
                
            total_server_inference_time += server_time
            
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue
            
        # 요청 종료 시간
        req_end = time.time()
        total_e2e_time += (req_end - req_start)
        total += 1

        # 테스트용으로 100개만 하고 멈추고 싶다면 아래 주석 해제
        # if total >= 100: break

    # 결과 출력
    accuracy = 100 * correct / total
    avg_server_time = (total_server_inference_time / total) * 1000
    avg_e2e_time = (total_e2e_time / total) * 1000

    print("\n" + "="*40)
    print(f"Total Images Processed: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg Server Inference Time (Pure GPU): {avg_server_time:.2f} ms")
    print(f"Avg End-to-End Latency (Network + Inference): {avg_e2e_time:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    run_test()