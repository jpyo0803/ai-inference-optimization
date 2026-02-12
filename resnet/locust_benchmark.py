import time
import random
import itertools
import io
import sys
import gevent
from locust import HttpUser, task, events
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP
from torchvision.datasets import CIFAR10

# 벤치마크 설정
TOTAL_REQUESTS_TO_PROCESS = 1000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 128
MIN_DELAY = 0.1
MAX_DELAY = 0.5

BASE_SEED = 42
DATASET_PATH = './data'
# 전역 변수
_cifar_images = []
_user_id_counter = itertools.count()
_current_request_count = 0

# 모니터링 & 강제 종료 로직
def monitor_and_stop(environment):
    """
    별도의 백그라운드 스레드(Greenlet)에서 1초마다 요청 수를 검사합니다.
    """
    while True:
        time.sleep(1)
        # 이미 종료 중이면 루프 탈출
        if environment.runner.state in [STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP]:
            break
            
        # 목표 달성 체크
        if _current_request_count >= TOTAL_REQUESTS_TO_PROCESS:
            print(f"\nReached target ({TOTAL_REQUESTS_TO_PROCESS}). Stopping gracefully...")
            environment.runner.quit()
            gevent.spawn_later(3, lambda: sys.exit(0))
            break

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    gevent.spawn(monitor_and_stop, environment)

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global _cifar_images
    print("[Init] Loading CIFAR-10 Dataset...")
    try:
        full_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True)
        num_samples = min(2000, len(full_dataset))
        for i in range(num_samples):
            img, _ = full_dataset[i]
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            _cifar_images.append(img_byte_arr.getvalue())
        print(f"[Init] Loaded {len(_cifar_images)} images.")
    except Exception as e:
        print(f"Dataset Load Error: {e}")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    global _current_request_count
    _current_request_count += 1

# 유저 시나리오
class BenchmarkUser(HttpUser):
    def on_start(self):
        self.user_id = next(_user_id_counter)
        self.my_seed = BASE_SEED + self.user_id
        self.rng = random.Random(self.my_seed)
        self.dataset = _cifar_images
        
    @task
    def inference(self):
        if not self.dataset: return

        # 재현 가능한 랜덤 Batch Size
        batch_size = self.rng.randint(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        
        # 재현 가능한 랜덤 Image Selection
        selected_indices = self.rng.sample(range(len(self.dataset)), batch_size)
        
        files = []
        for idx in selected_indices:
            files.append(('files', (f'img_{idx}.jpg', self.dataset[idx], 'image/jpeg')))

        # Request
        with self.client.post("/predict_batch", files=files, name="Inference", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Status: {response.status_code}")
        
        # 재현 가능한 랜덤 Delay
        sleep_time = self.rng.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(sleep_time)