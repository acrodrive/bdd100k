import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

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

"""def collate_fn(batch):
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
"""
"""
def collate_fn(batch):
    return tuple(zip(*batch))
"""
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def collate_fn(batch):
    if len(batch) == 0:
        return [], []
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets
    """images, boxes, class_ids = tuple(zip(*batch))
    images = torch.stack(images, dim=0)  # 이미지 배치 텐서로 변환
    return images, boxes, class_ids"""

class BDD100KDataset(Dataset):
	def __init__(self, image_dir, label_dir, class_to_id, transform=None):
		self.image_dir = '/content/' + image_dir if in_colab() else image_dir
		self.label_dir = '/content/' + label_dir if in_colab() else label_dir
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

	            # ✅ 유효성 검사 추가
				if x2 > x1 and y2 > y1:
					boxes.append([x1, y1, x2, y2])
					category = obj["category"]
					if category in self.class_to_id:
						class_ids.append(self.class_to_id[category])
					else:
						class_ids.append(-1)
				else:
					print(f"⚠️ Skipped invalid box: {x1, y1, x2, y2}")

		# 박스가 하나도 없는 경우를 대비해 최소 하나 넣기 (선택 사항)
		if len(boxes) == 0:
			boxes.append([0, 0, 1, 1])
			class_ids.append(0)  # 또는 background 클래스 ID

		boxes = torch.tensor(boxes, dtype=torch.float32)
		class_ids = torch.tensor(class_ids, dtype=torch.long)

		target = {
	        "boxes": boxes,
	        "labels": class_ids
	    }

		if self.transform:
			img = self.transform(img)

		return img, target
