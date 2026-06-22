import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """Image folder dataset used by the legacy STC trainer.

    Expected layout:
        root_dir/
          split_or_source_name/
            class_folder/
              *.jpg

    Labels default to the first character of the class folder, matching the
    original CWRU/BJTU/HUST naming convention: N, I, O, B.
    """

    def __init__(self, root_dir, transform=None, label_func=lambda foldername: foldername[0], scale=int(1e20)):
        self.root_dir = str(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = set()
        self.class_name_folders = {}
        scale = int(math.ceil(scale))

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        manifest_path = Path(self.root_dir) / "manifest.json"
        if manifest_path.exists():
            self._load_manifest(manifest_path, label_func, scale)
            self._load_or_create_class_index()
            return

        for data_name in sorted(os.listdir(self.root_dir)):
            data_dir = os.path.join(self.root_dir, data_name)
            if not os.path.isdir(data_dir):
                continue
            for class_name in sorted(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    label = label_func(class_name)
                    self.class_name_folders[label] = self.class_name_folders.get(label, 0) + 1
                    self.class_names.add(label)

        for class_name in self.class_name_folders:
            folder_num = self.class_name_folders[class_name]
            q, r = divmod(scale, int(folder_num))
            self.class_name_folders[class_name] = [q + 1] * r + [q] * (folder_num - r)

        for data_name in sorted(os.listdir(self.root_dir)):
            data_dir = os.path.join(self.root_dir, data_name)
            if not os.path.isdir(data_dir):
                continue
            for class_name in sorted(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                label = label_func(class_name)
                num = self.class_name_folders[label].pop()
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(class_dir, img_name)
                        if os.path.isfile(img_path):
                            self.samples.append((img_path, label))
                            num -= 1
                    if num == 0:
                        break

        self._load_or_create_class_index()

    def _load_manifest(self, manifest_path, label_func, scale):
        with Path(manifest_path).open("r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        entries = manifest.get("entries", [])
        by_label = defaultdict(list)
        for entry in entries:
            img_path = entry["path"]
            class_name = entry.get("class_name") or Path(img_path).parent.name
            label = entry.get("label") or label_func(class_name)
            if os.path.isfile(img_path):
                by_label[label].append((img_path, label))
                self.class_names.add(label)

        for label in sorted(by_label):
            selected = by_label[label][:scale]
            self.samples.extend(selected)
            self.class_name_folders[label] = 1

    def _load_or_create_class_index(self):
        class_index_path = Path("class_to_idx.json")
        if not class_index_path.exists():
            self.class_names = sorted(self.class_names)
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
            with class_index_path.open("w", encoding="utf-8") as f:
                json.dump(self.class_to_idx, f, ensure_ascii=False, indent=2)
        else:
            with class_index_path.open("r", encoding="utf-8") as f:
                self.class_to_idx = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __add__(self, other):
        assert isinstance(other, CustomImageDataset), "Can only add CustomImageDataset instances"
        assert self.transform == other.transform, "Transforms must match"
        assert self.class_to_idx == other.class_to_idx, "Class indices must match"
        assert self.class_names == other.class_names, "Class names must match"
        new_dataset = deepcopy(self)
        new_dataset.samples.extend(other.samples)
        return new_dataset

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx

    def random_samples(self, num_samples):
        random_samples = []
        for img_path, _ in random.sample(self.samples, num_samples):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            random_samples.append(image)
        return random_samples

    def each_classes_samples(self):
        class_ids = self.class_to_idx.values()
        class_indexes = []
        classes_samples = []
        for class_id in class_ids:
            for image, label_idx in self:
                if (label_idx == class_id) and (label_idx not in class_indexes):
                    classes_samples.append(image)
                    class_indexes.append(label_idx)
                    break
        return classes_samples

    def class_samples(self, class_id, num_samples=1e23):
        class_samples = []
        for img_path, label in self.samples:
            if (self.class_to_idx[label] == class_id) and (len(class_samples) < num_samples):
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                class_samples.append((image, img_path, label))
        return class_samples

    def name(self):
        return "CustomImageDataset"
