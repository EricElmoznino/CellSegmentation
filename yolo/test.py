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
trained_model = os.path.join(cfg.train_output_dir,
                             'darknet19_voc07trainval_exp3_118.h5')
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
                                                          np.array(tr.to_pil_image(batch['image'])).shape,
                                                          cfg,
                                                          thresh,
                                                          0
                                                          )

        for j in range(1):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, 1):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if vis:
            im2show = yolo_utils.draw_detection(np.array(tr.to_pil_image(batch['image'])),
                                                bboxes,
                                                scores,
                                                cls_inds,
                                                cfg,
                                                thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show,
                                     (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            cv2.imshow('test', im2show)
            cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


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
