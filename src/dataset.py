from torch.utils.data import Dataset
from collections import defaultdict
import os
import random
from PIL import Image

class Market1501(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_list = []
        self.labels_list = []
        self.ids = []

        for image in os.listdir(root):
            if not image.endswith(".jpg"):
                continue

            img_id = image.split('_')[0]

            if img_id in ['0000', '-1']:
                continue

            image_path = os.path.join(root, image)
            self.images_list.append(image_path)
            self.ids.append(img_id)

        unique_ids = sorted(list(set(self.ids)))
        self.id2label = {
            id_: i for i, id_ in enumerate(unique_ids)
        }
        self.labels_list = [
            self.id2label[id_] for id_ in self.ids
        ]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.labels_list[idx]

        return img, label

class Evaldataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for img in os.listdir(root):
            if not img.endswith(".jpg"):
                    continue

            parts = img.split('_')
            pid = parts[0]
            camid = parts[1][1]

            img_path = os.path.join(self.root, img)

            self.samples.append((pid, img_path, camid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, img_path, camid = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, pid, camid




