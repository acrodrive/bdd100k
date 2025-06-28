


""" outdated
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

class MySimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = MyConvLayer(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MyConvLayer(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # x: [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # x: [B, 32, 32, 32]
        x = x.view(x.size(0), -1)             # x: [B, 32*32*32]
        return self.fc(x)
    
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, boxes, class_ids in dataloader:
            if any(cls.numel() == 0 or (cls == -1).any().item() for cls in class_ids):
                continue
            imgs = imgs.to(device)
            labels = torch.tensor([cls[0].item() for cls in class_ids], device=device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy"""