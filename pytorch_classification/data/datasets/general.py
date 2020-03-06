import os
import torch
import PIL.Image as Image

from torch.utils.data.dataset import Dataset


class GeneralDataset(Dataset):
    def __init__(self, img_dir, anno_file, transforms=None):
        self.img_dir = img_dir
        self.anno_file = anno_file
        with open(self.anno_file, "r") as f:
            anno = [x.strip() for x in f]
        self.img_names = [x.split()[0] for x in anno]
        self.labels = [x.split()[1] for x in anno]
        self.transforms = transforms

        assert len(self.img_names) == len(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(int(self.labels[index]))

        return img, label
