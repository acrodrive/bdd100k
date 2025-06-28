import torch
from torchvision import transforms
import time
from dataset import BDD100KDataset, class_to_id, collate_fn
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
import copy
import torchvision.models.detection
from engine import evaluate, train_one_epoch
from torch.utils.data import Subset

def train_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

    # 1) val 데이터 일부 샘플링 (500개)
    val_subset = Subset(val_dataset, list(range(500)))

    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # model = MySimpleCNN(num_classes=len(class_to_id)).to(device)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    best_val_loss = float('inf')
    best_model_state = None
    best_map = None
    patience = 3
    no_improve_count = 0
    num_epochs = 5

    for epoch in range(num_epochs):
        start_epoch_time = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)

        coco_eval = evaluate(model, val_loader, device=device)
        current_map = coco_eval.coco_eval['bbox'].stats[0]

        print(f"\n✅ Epoch {epoch+1} Summary:")
        print(f"  - mAP@[0.50:0.95]: {current_map:.4f}")
        print(f"  - Epoch Time: {time.time() - start_epoch_time:.2f} sec")

        if best_map is None or current_map > best_map:
            best_map = current_map
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"    🥇 New best model saved (mAP: {best_map:.4f})")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"    No improvement for {no_improve_count} epoch(s)")
    
            if no_improve_count >= patience:
                print(f"\n    ⛔ Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'best_model.pth')
        print("\n    Best model saved as 'best_model.pth'")

    # 저장
    torch.save(model.state_dict(), 'my_simple_cnn.pth')

    # 불러오기

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows에서 freeze된 실행 파일 생성 시 유용 (선택적)
    train_model()