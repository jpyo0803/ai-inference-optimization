import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

# 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Calibration을 위한 데이터셋 준비 (Augmentation은 사용 X)
calib_transforms = Compose([
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

def get_calibration_loader(batch_size=1):
    # 편의상 Test 셋을 씁니다.
    dataset = CIFAR10(root='data/processed', train=False, download=True, transform=calib_transforms)
    
    # 약 100~500장이면 충분합니다.
    # DataLoader가 데이터를 하나씩 뱉어주게 합니다.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# PyTorch DataLoader와 연동되는 Calibrator 클래스
class CIFAR10EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file="cifar10_calibration.cache", total_images=500):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.cache_file = cache_file
        self.dataloader = dataloader
        self.iterator = iter(dataloader) # PyTorch Iterator 생성
        self.total_images = total_images
        self.current_images = 0
        self.batch_size = dataloader.batch_size
        
        # GPU 메모리 할당 (Batch, 3, 32, 32) * 4 bytes(float32)
        # CIFAR10은 32x32 입니다.
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 32 * 32 * 4) 

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # 정해진 수만큼(예: 500장) 돌렸으면 종료
        if self.current_images >= self.total_images:
            return None

        try:
            # DataLoader에서 (이미지, 라벨) 가져오기
            data, _ = next(self.iterator)
            
            # Tensor -> Numpy 변환
            # PyTorch Tensor는 이미 NCHW 포맷이므로 transpose 불필요
            batch_numpy = data.numpy().astype(np.float32)
            
            # 메모리 연속성 보장 (1차원으로 쭉 폄)
            batch_numpy = np.ascontiguousarray(batch_numpy.ravel())
            
            # CPU -> GPU 복사
            cuda.memcpy_htod(self.device_input, batch_numpy)
            
            self.current_images += self.batch_size
            print(f"Calibration Progress: {self.current_images}/{self.total_images}")
            
            return [int(self.device_input)]
            
        except StopIteration:
            return None

    def read_calibration_cache(self):
        # 기존 캐시가 있으면 재사용
        if os.path.exists(self.cache_file):
            print(f"Loading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 캐시 저장
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class CIFAR10MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataloader, cache_file="cifar10_calibration.cache", total_images=500):
        trt.IInt8MinMaxCalibrator.__init__(self)
        
        self.cache_file = cache_file
        self.dataloader = dataloader
        self.iterator = iter(dataloader) # PyTorch Iterator 생성
        self.total_images = total_images
        self.current_images = 0
        self.batch_size = dataloader.batch_size
        
        # GPU 메모리 할당 (Batch, 3, 32, 32) * 4 bytes(float32)
        # CIFAR10은 32x32 입니다.
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 32 * 32 * 4) 

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # 정해진 수만큼(예: 500장) 돌렸으면 종료
        if self.current_images >= self.total_images:
            return None

        try:
            # DataLoader에서 (이미지, 라벨) 가져오기
            data, _ = next(self.iterator)
            
            # Tensor -> Numpy 변환
            # PyTorch Tensor는 이미 NCHW 포맷이므로 transpose 불필요
            batch_numpy = data.numpy().astype(np.float32)
            
            # 메모리 연속성 보장 (1차원으로 쭉 폄)
            batch_numpy = np.ascontiguousarray(batch_numpy.ravel())
            
            # CPU -> GPU 복사
            cuda.memcpy_htod(self.device_input, batch_numpy)
            
            self.current_images += self.batch_size
            print(f"Calibration Progress: {self.current_images}/{self.total_images}")
            
            return [int(self.device_input)]
            
        except StopIteration:
            return None

    def read_calibration_cache(self):
        # 기존 캐시가 있으면 재사용
        if os.path.exists(self.cache_file):
            print(f"Loading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 캐시 저장
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# 엔진 빌드 메인 함수
import os

def build_engine_cifar10(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # ONNX 파싱
    if not os.path.exists(onnx_file_path):
        print(f"Error: {onnx_file_path} not found.")
        return

    if not parser.parse_from_file(onnx_file_path):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    profile = builder.create_optimization_profile()
    
    # 네트워크의 첫 번째 입력 텐서 이름 가져오기 (보통 'input')
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # (Min, Opt, Max) Shape 설정
    # Min: 배치 1
    # Opt: 배치 128 (주로 쓰는 크기)
    # Max: 배치 128 (최대 허용 크기)
    profile.set_shape(input_name, (1, 3, 32, 32), (128, 3, 32, 32), (128, 3, 32, 32))
    
    # 설정한 프로파일을 Config에 등록
    config.add_optimization_profile(profile)

    # INT8 설정
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16) # INT8 안되는 층은 FP16으로

    # PyTorch DataLoader 연결
    # 배치 사이즈 1로 500장 정도만 보정에 사용
    calib_loader = get_calibration_loader(batch_size=1)
    calibrator = CIFAR10MinMaxCalibrator(calib_loader, cache_file="cifar10.cache", total_images=500)
    config.int8_calibrator = calibrator

    # 엔진 빌드
    print("Building TensorRT Engine with Calibration...")
    plan = builder.build_serialized_network(network, config)
    
    if plan:
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        print(f"\nBuild Success! Saved to: {engine_file_path}")
        print("Now you have a calibrated INT8 model.")
    else:
        print("Build Failed.")

if __name__ == "__main__":
    cuda.init()
    device = cuda.Device(0)  # 0번 GPU 선택
    ctx = device.make_context()

    # 가지고 계신 ONNX 파일 경로를 적어주세요
    ONNX_MODEL = "model_repository/resnet_onnx/1/model.onnx"       
    TRT_ENGINE = "resnet_cifar10_int8.engine"
    
    build_engine_cifar10(ONNX_MODEL, TRT_ENGINE)