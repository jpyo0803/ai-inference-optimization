from dataset import load_train_dataset
from config import Config
from model import ResNet_CIFAR

import torch
import torch.optim as optim

cfg = Config()

def train():
    train_dataset = load_train_dataset()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    
    model = ResNet_CIFAR(depth=cfg.depth)
    model = model.to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss() # This takes in 'logits' and do softmax internally

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    '''
        Divide learning rate by 10 at 32k and 48k interations.
        
        The number of training samples is 50,000.
        The number of batches per epoch is 50000 / 128 = 390 batches
        After 82 (32k/390) and 123 (48k/390) epochs, divide lr by 10
    '''
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[82, 123], gamma=0.1
    )

    device = cfg.device

    max_accuracy = None

    for epoch in range(cfg.num_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            _, predicted = outputs.max(dim=1)
            total += inputs.shape[0]
            correct += (predicted == targets).sum().item()

        train_acc = correct / total * 100
        train_loss = running_loss / total
        scheduler.step()

        print(f'Epoch: {epoch + 1} / {cfg.num_epoch}, Loss: {train_loss:.4f} Acc: {train_acc:.2f} %')

        if max_accuracy is None or train_acc > max_accuracy:
            max_accuracy = train_acc
        else:
            break

    torch.save(model.state_dict(), f'weight/resnet_{cfg.depth}.pth')
    print("Training complete")


if __name__ == "__main__":
    train()