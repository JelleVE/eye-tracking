import cv2, os
import sys
sys.path.insert(0, 'PIPNet/FaceBoxesV2')
# sys.path.insert(0, '..')
import numpy as np
import pickle
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .lib.networks import *
from .lib.functions import *
from .lib.mobilenetv3 import mobilenetv3_large


class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 98
        self.save_interval = self.num_epochs
        self.num_nb = 10
        self.use_gpu = False
        self.gpu_id = 2



class PIPNet():
    def __init__(self):
        network_path = 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py'

        experiment_name = network_path.split('/')[-1][:-3]
        data_name = network_path.split('/')[-2]
        config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

        # my_config = importlib.import_module(config_path, package='PIPNet')
        # Config = getattr(my_config, 'Config')
        cfg = Config()
        cfg.experiment_name = experiment_name
        cfg.data_name = data_name

        save_dir = os.path.join('PIPNet/snapshots', cfg.data_name, cfg.experiment_name)
        meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('PIPNet/data', cfg.data_name, 'meanface.txt'), cfg.num_nb)


        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=cfg.pretrained)
            net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=cfg.pretrained)
            net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=cfg.pretrained)
            net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'mobilenet_v2':
            mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
            net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        elif cfg.backbone == 'mobilenet_v3':
            mbnet = mobilenetv3_large()
            if cfg.pretrained:
                mbnet.load_state_dict(torch.load('PIPNet/lib/mobilenetv3-large-1cd25616.pth'))
            net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        else:
            print('No such backbone!')
            exit(0)

        if cfg.use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        net = net.to(device)

        weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
        state_dict = torch.load(weight_file, map_location=device)
        net.load_state_dict(state_dict)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

        detector = FaceBoxesDetector('FaceBoxes', 'PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth', cfg.use_gpu, device)

        self.cfg = cfg
        self.preprocess = preprocess
        self.net = net
        self.device = device
        self.detector = detector
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2
        self.max_len = max_len



    def detectFaces(self, image):
        my_thresh = 0.6
        
        self.net.eval()
        image_height, image_width, _ = image.shape
        detections, _ = self.detector.detect(image, my_thresh, 1)
        return detections



    def detectLandmarks(self, image, detection):
        det_box_scale = 1.2
        self.net.eval()

        image_height, image_width, _ = image.shape

        det_xmin = detection[2]
        det_ymin = detection[3]
        det_width = detection[4]
        det_height = detection[5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale-1)/2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale-1)/2)
        det_xmax += int(det_width * (det_box_scale-1)/2)
        det_ymax += int(det_height * (det_box_scale-1)/2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width-1)
        det_ymax = min(det_ymax, image_height-1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1
        cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (self.cfg.input_size, self.cfg.input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = self.preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(self.device)
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(self.net, inputs, self.preprocess, self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()

        coords = np.zeros((self.cfg.num_lms,2))
        landmarks = dict()
        for i in range(self.cfg.num_lms):
            x_pred = lms_pred_merge[i*2] * det_width
            y_pred = lms_pred_merge[i*2+1] * det_height

            coords[i,0] = int(x_pred)+det_xmin
            coords[i,1] = int(y_pred)+det_ymin

            cv2.circle(image, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), 2)

        landmarks['jaw'] = coords[0:33]
        landmarks['eyebrow_left'] = coords[33:42]
        landmarks['eyebrow_right'] = coords[42:51]
        landmarks['eye_left'] = coords[60:68]
        landmarks['eye_right'] = coords[68:76]
        landmarks['pupil_left'] = coords[96:97]
        landmarks['pupil_right'] = coords[97:98]
        landmarks['nose'] = coords[51:60]
        landmarks['lips'] = coords[76:96]

        return landmarks
        
    #cv2.imwrite('images/1_out.jpg', image)
    # cv2.imshow('1', image)
    # cv2.waitKey(0)