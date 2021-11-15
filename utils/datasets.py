import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os.path as osp
import json
import torchvision.transforms.functional as FT
# train.py 文件要加. 表示同级目录下的utils模块 但是想运行datasets.py 不要.
from .utils import transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_folder = 'data/VOC2012/json'


class PasCalDataset(Dataset):
    def __init__(self, data_folder, split='train', keep_difficult=True) -> None:
        super(PasCalDataset, self).__init__()
        self.keep_difficult = keep_difficult
        self.split = split

        assert split in {'train', 'val'}

        j = open(osp.join(data_folder, split+'_images.json'), 'r')  # 图片的绝对路径
        self.path = json.load(j)
        object = open(osp.join(data_folder, split+'_objects.json'), 'r')
        self.objects = json.load(object)

    def __getitem__(self, index):
        path = self.path[index]
        image = Image.open(path)  # PIL对象
        image.convert('RGB')

        object = self.objects[index]
        boxes = torch.FloatTensor(object['boxes'])
        labels = torch.LongTensor(object['labels'])
        difficulties = torch.ByteTensor(object['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, split=self.split)

        # transform
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.path)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties
        #tensor(b, h, w), list(b, n_objects, 4), list(b, n_objects), list(b, n_objects)

# dataset = PasCalDataset(data_folder, keep_difficult=False)
# dataloader = DataLoader(dataset=dataset, batch_size=32, collate_fn=dataset.collate_fn, drop_last=True)
# print(next(iter(dataloader)))
# print(len(dataloader))
