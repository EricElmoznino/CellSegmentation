from torch.utils.data import Dataset
from torchvision.transforms import functional as tr
from PIL import Image
import os
import numpy as np
import torch

from PIL import ImageDraw
import random


def mask_to_bounding_box(mask_path):
    mask = Image.open(mask_path).convert('L')
    return np.array(mask.getbbox(), dtype=np.float32)


class YoloDataset(Dataset):

    def __init__(self, dir, augment=False):
        samples = os.listdir(dir)
        samples = [os.path.join(dir, s) for s in samples]

        self.data = []
        for s in samples:
            image = os.listdir(os.path.join(s, 'images'))
            image = [os.path.join(s, 'images', im) for im in image if '.png' in im][0]

            masks = os.listdir(os.path.join(s, 'masks'))
            masks = [os.path.join(s, 'masks', m) for m in masks if '.png' in m]
            bounding_boxes = []
            for mask in masks:
                bounding_boxes.append(mask_to_bounding_box(mask))
            bounding_boxes = np.array(bounding_boxes)

            self.data.append({'image': image, 'bounding_boxes': bounding_boxes})


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image, bounding_boxes = self.data[item]['image'], self.data[item]['bounding_boxes']

        image = Image.open(image).convert('RGB')
        scale_x = 320.0 / image.size[0]
        scale_y = 320.0 / image.size[1]
        image = tr.resize(image, [320, 320])
        bounding_boxes[:, np.array([0, 2])] *= scale_x
        bounding_boxes[:, np.array([1, 3])] *= scale_y

        draw = ImageDraw.Draw(image)
        for b in bounding_boxes:
            b = b.tolist()
            draw.rectangle(b, outline='blue')
        image.save('/Users/eric/Desktop/images/'+str(random.randint(0, 1000))+'.jpg')

        image = tr.to_tensor(image)

        return {'image': image, 'bounding_boxes': bounding_boxes,
                'classes': np.zeros([len(bounding_boxes), 1], dtype=np.int)}

    @staticmethod
    def collate_fn(batch):
        images = [s['image'] for s in batch]
        bounding_boxes = [s['bounding_boxes'] for s in batch]
        classes = [s['classes'] for s in batch]
        images = torch.stack(images)
        return {'image': images, 'bounding_boxes': bounding_boxes, 'classes': classes}

