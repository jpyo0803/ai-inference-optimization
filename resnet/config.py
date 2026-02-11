import torch

class Config:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.batch_size = 64
        self.lr = 0.1
        self.momentum=0.9
        self.weight_decay=1e-4
        self.num_epoch=30 # 64k/390