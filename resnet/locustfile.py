import random
import time
import itertools
from locust import HttpUser, task, events
from torchvision.datasets import CIFAR10
import io

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8
MIN_DELAY = 0.5      # 최소 대기 시간 (초)
MAX_DELAY = 2.0      # 최대 대기 시간 (초)

BASE_SEED = 2024     # 재현성을 위한 기본 시드
DATASET_PATH = './data'

_cifar_images = []       # 이미지 바이트 데이터를 담을 전역 리스트
_user_id_counter = itertools.count() # 유저에게 0, 1, 2... ID를 부여하기 위한 카운터

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
        데이터셋을 메모리에 로드하고 바이트로 변환해둡니다.
    """
    global _cifar_images
    print("Loading CIFAR-10 Dataset into memory... (One-time setup)")
    
    # 전체 데이터셋 로드
    full_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True)
    
    num_samples = min(2000, len(full_dataset))
    
    for i in range(num_samples):
        img, _ = full_dataset[i]
        
        # 이미지를 바이트로 미리 변환
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        _cifar_images.append(img_bytes)
    
    print(f"Loaded {len(_cifar_images)} images ready for deterministic testing.")

class DeterministicUser(HttpUser):
    """
        각 인스턴스가 자신만의 Random Number Generator(RNG)를 가집니다.
        따라서 어떤 상황에서도 동일한 시퀀스(배치크기, 딜레이)를 재현합니다.
    """

    def on_start(self):
        # 고유 User ID 발급 (0, 1, 2, ...)
        self.user_id = next(_user_id_counter)
        
        # 유저별 고유 시드 설정
        # 예: User 0 -> Seed 2024, User 1 -> Seed 2025 ...
        my_seed = BASE_SEED + self.user_id
        self.rng = random.Random(my_seed)
        
        # 데이터셋 포인터 연결
        self.dataset = _cifar_images
        
        print(f"[User #{self.user_id}] Spawned with Seed {my_seed}")

    @task
    def send_inference_request(self):
        # Deterministic Batch Size 결정
        batch_size = self.rng.randint(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        
        # Deterministic Image Selection
        # 데이터셋에서 랜덤하게(하지만 시드에 의해 고정된 순서로) 이미지 선택
        selected_indices = self.rng.sample(range(len(self.dataset)), batch_size)
        
        # multipart/form-data 형식 구성
        files = []
        for i, idx in enumerate(selected_indices):
            img_bytes = self.dataset[idx]
            # ('key', ('filename', content, 'mime_type'))
            files.append(('files', (f'img_{idx}.jpg', img_bytes, 'image/jpeg')))

        # Request Sending
        # catch_response=True로 직접 성공/실패 판정
        with self.client.post("/predict_batch", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    # 응답 파싱 (Latency 메트릭 확인용)
                    res_json = response.json()
                    server_process_time = res_json.get('total_processing_time_sec', 0)
                    
                    # Locust UI에 커스텀 메트릭을 찍고 싶다면:
                    # events.request.fire(...) 등을 사용할 수 있음
                    response.success()
                except Exception as e:
                    response.failure(f"JSON Parse Error: {e}")
            else:
                response.failure(f"Status {response.status_code}: {response.text}")

        # Deterministic Delay (Wait Time)
        # Locust의 wait_time 속성 대신 직접 sleep을 주어
        # RNG 시퀀스가 꼬이지 않게 완벽하게 통제합니다.
        sleep_time = self.rng.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(sleep_time)