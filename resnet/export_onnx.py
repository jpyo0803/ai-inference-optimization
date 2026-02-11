import torch
import torch.nn as nn
from model import ResNet50
import os

WEIGHTS_PATH = "weight/resnet50.pth"
OUTPUT_PATH = "model_repository/resnet_onnx/1/model.onnx"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def export_onnx():
    model = ResNet50(num_classes=10)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"Weight file {WEIGHTS_PATH} not found. Exiting.")
        return

    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input size

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {OUTPUT_PATH}")

if __name__ == "__main__":
    export_onnx()