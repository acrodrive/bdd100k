import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from model import MySimpleCNN, evaluate
from dataset import BDD100KDataset, class_to_id, collate_fn
import torch.nn as nn

def test_model():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = BDD100KDataset(
        image_dir='bdd100k/images/100k/test',
        label_dir='bdd100k/labels/100k/test',
        class_to_id=class_to_id,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = MySimpleCNN(num_classes=len(class_to_id)).to(device)
    
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_id))
    model = model.to(device)
    
    # 저장된 모델 로드
    model.load_state_dict(torch.load('my_simple_cnn.pth', map_location=device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    avg_loss, accuracy = evaluate(model, test_loader, criterion, device)

    print("\n    Test 결과")
    print(f"  - Test Loss     : {avg_loss:.4f}")
    print(f"  - Test Accuracy : {accuracy*100:.2f}%")

if __name__ == '__main__':
    test_model()