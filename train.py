import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import time

class_to_id = {
	"person": 0,
	"rider": 1,
	"car": 2,
	"truck": 3,
	"bus": 4,
	"train": 5,
	"motor": 6,
	"bike": 7,
	"traffic light": 8,
	"traffic sign": 9,
	"banner": 10,
	"billboard": 11,
	"lane divider": 12,
	"parking sign": 13,
	"pole": 14,
	"pole group": 15,
	"street light": 16,
	"traffic cone": 17,
	"traffic device": 18,
	"sign frame": 19,
	"caravan": 20,
	"motorcycle": 21,
	"trailer": 22,
	"area/alternative": 23,
	"area/drivable": 24,
	"lane/crosswalk": 25,
	"lane/double other": 26,
	"lane/double white": 27,
	"lane/double yellow": 28,
	"lane/road curb": 29,
	"lane/single other": 30,
	"lane/single white": 31,
	"lane/single yellow": 32
}


def collate_fn(batch):
	images = []
	boxes = []
	class_ids = []

	for img, box, cls in batch:
		images.append(img)
		boxes.append(box)
		class_ids.append(cls)

	images = torch.stack(images, dim=0)

	for i, cls in enumerate(class_ids):
		if cls.numel() == 0:
			print(f"Warning: sample {i} has empty class_ids")

	return images, boxes, class_ids

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
    return avg_loss, accuracy

class BDD100KDataset(Dataset):
	def __init__(self, image_dir, label_dir, class_to_id, transform=None):
		self.image_dir = image_dir
		self.label_dir = label_dir
		self.transform = transform
		self.class_to_id = class_to_id
		self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
		self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_path = os.path.join(self.image_dir, self.image_files[idx])
		label_path = os.path.join(self.label_dir, self.label_files[idx])

		img = Image.open(img_path).convert("RGB")

		with open(label_path, 'r') as f:
			data = json.load(f)

		boxes = []
		class_ids = []

		frame = data["frames"][0]

		for obj in frame["objects"]:
			if "box2d" in obj:
				box = obj["box2d"]
				x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
				boxes.append([x1, y1, x2, y2])

				category = obj["category"]
				if category in self.class_to_id:
					class_ids.append(self.class_to_id[category])
				else:
					class_ids.append(-1)

		boxes = torch.tensor(boxes, dtype=torch.float32)
		class_ids = torch.tensor(class_ids, dtype=torch.long)

		if self.transform:
			img = self.transform(img)

		return img, boxes, class_ids


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


def main():
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = BDD100KDataset(
        image_dir='bdd100k/images/100k/train',
        label_dir='bdd100k/labels/100k/train',
        class_to_id=class_to_id,
        transform=transform
    )

    val_dataset = BDD100KDataset(
        image_dir='bdd100k/images/100k/val',
        label_dir='bdd100k/labels/100k/val',
        class_to_id=class_to_id,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True) #개선사항
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MySimpleCNN(num_classes=len(class_to_id)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 3
    no_improve_epochs = 0
    max_epochs = 50

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()

        for i, (imgs, boxes, class_ids) in enumerate(train_loader):
            start_time = time.time()

            if any(cls.numel() == 0 or (cls == -1).any().item() for cls in class_ids):
                continue

            imgs = imgs.to(device)
            labels = torch.tensor([cls[0].item() for cls in class_ids], device=device) #개선사항

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
			
            if i % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Time: {time.time() - start_time:.4f}")

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - start_epoch_time

        #print(f"Epoch {epoch+1} Loss: {running_loss / train_loader.batch_size:.4f}")
        #print(f"[Epoch {epoch+1} DONE] ⏱ Time: {epoch_duration:.2f}s | 🏋️ Train Loss: {avg_train_loss:.4f} | 🔍 Val Loss: {val_loss:.4f} | ✅ Val Acc: {val_acc*100:.2f}%")
        print(f"\n✅ Epoch {epoch+1} Summary:")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Val Loss  : {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
        print(f"  - Epoch Time: {time.time() - start_epoch_time:.2f} sec")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
            print("   Validation loss improved. Saving model.")
        else:
            no_improve_epochs += 1
            print(f"   No improvement for {no_improve_epochs} epoch(s).")

            if no_improve_epochs >= patience:
                print(f"\n Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'best_model.pth')
        print("\n📦 Best model saved as 'best_model.pth'")

    # 저장
    torch.save(model.state_dict(), 'my_simple_cnn.pth')

    # 불러오기
    model = MySimpleCNN(num_classes=len(class_to_id)).to(device)
    model.load_state_dict(torch.load('my_simple_cnn.pth'))
    model.eval()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Windows에서 freeze된 실행 파일 생성 시 유용 (선택적)
    main()