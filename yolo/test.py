import os
import cv2
import numpy as np
import pickle
import argparse

from torch.utils.data import DataLoader
from datasets.yolo_dataset import YoloDataset
from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
import cfgs.config as cfg
import torch
from torch.autograd import Variable
from torchvision.transforms import functional as tr
from PIL import Image, ImageDraw
import random


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test(image, cfg.inp_size), 0)  # noqa
    return image, im_data


parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--image_size_index', type=int, default=0,
                    metavar='image_size_index',
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
args = parser.parse_args()


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
trained_model = 'yolo-cells.h5'
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.01
vis = False
# ------------


def test_net(net, dataloader, max_per_image=300, thresh=0.5, vis=False):
    num_images = len(dataloader.dataset)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(1)]

    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index

    for i, batch in enumerate(dataloader):
        orig_image = np.array(tr.to_pil_image(batch['image'][0]))
        image = batch['image']
        image = Variable(image)
        if torch.cuda.is_available():
            image = image.cuda()

        bbox_pred, iou_pred, prob_pred = net(image)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,
                                                          iou_pred,
                                                          prob_pred,
                                                          orig_image.shape,
                                                          cfg,
                                                          thresh,
                                                          0
                                                          )

        orig_image = Image.fromarray(orig_image)
        draw = ImageDraw.Draw(orig_image)
        for box in bboxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='blue')
        orig_image.save('outputs/'+str(random.randint(0, 10000))+'.jpg')


if __name__ == '__main__':
    test_dataset = YoloDataset(os.path.abspath('../data/stage1_validation'))
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                                  num_workers=2, collate_fn=YoloDataset.collate_fn)
    print('load data succ...')

    net = Darknet19()
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()

    test_net(net, test_dataloader, max_per_image, thresh, vis)
