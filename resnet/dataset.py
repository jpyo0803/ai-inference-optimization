from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

transforms = Compose([
    RandomCrop((32, 32), padding=4),
    # pad 4 zeros around and randomly crop an image of a size 32 x 32. This helps train translational invariant property
    RandomHorizontalFlip(p=0.5), # flip image horizontally with 50% probability
    ToTensor(), # Convert 'PIL.Image' or 'numpy.ndarray' to torch tensor
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

'''
    this method returns dataset for training
'''
def load_train_dataset():
    training_dataset = CIFAR10(root='data/processed', train=True, download=True, transform=transforms)
    return training_dataset

def load_test_dataset():
    test_dataset = CIFAR10(root='data/processed', train=False, download=True, transform=transforms)
    return test_dataset

if __name__ == '__main__':
    train_data = load_train_dataset()
    test_data = load_test_dataset()