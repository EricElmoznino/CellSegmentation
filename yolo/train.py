import os
import torch
import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable

from darknet import Darknet19

from datasets.yolo_dataset import YoloDataset
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


# data loader
train_dataset = YoloDataset(os.path.abspath('../data/stage1_train'))
train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                              num_workers=2, collate_fn=YoloDataset.collate_fn)
print('load data succ...')

net = Darknet19()
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

t = Timer()
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
for epoch in range(0, cfg.max_epoch):

    for i, data in enumerate(train_dataloader):
        t.tic()

        image, bouding_boxes, classes = data['image'], data['bounding_boxes'], data['classes']
        image = Variable(image)
        if torch.cuda.is_available():
            image = image.cuda()

        net(image, bouding_boxes, classes, [[] for _ in range(cfg.train_batch_size)], 0)

        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        duration = t.toc()

        if i % cfg.disp_interval == cfg.disp_interval - 1:
            train_loss /= cfg.disp_interval
            bbox_loss /= cfg.disp_interval
            iou_loss /= cfg.disp_interval
            cls_loss /= cfg.disp_interval
            print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                   'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                   (epoch, i+1, len(train_dataset)//cfg.train_batch_size, train_loss, bbox_loss,
                    iou_loss, cls_loss, duration,
                    str(datetime.timedelta(seconds=int((len(train_dataset)//cfg.train_batch_size - (i+1)) * duration))))))

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            t.clear()

    if epoch in cfg.lr_decay_epochs:
        lr *= cfg.lr_decay
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)

    save_name = os.path.join(cfg.train_output_dir,
                             '{}_{}.h5'.format(cfg.exp_name, epoch))
    net_utils.save_net(save_name, net)
    print(('save model: {}'.format(save_name)))
