from dataset import load_train_dataset
from config import Config
from model import ResNet50

import torch
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

cfg = Config()

def train():
    train_dataset = load_train_dataset()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = ResNet50(num_classes=10).to(cfg.device)
    scaler = GradScaler()

    criterion = torch.nn.CrossEntropyLoss() # This takes in 'logits' and do softmax internally

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch)

    device = cfg.device

    max_accuracy = None

    model.train()

    for epoch in range(cfg.num_epoch):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

    torch.save(model.state_dict(), f'weight/resnet50.pth')
    print("Training complete")


if __name__ == "__main__":
    train()