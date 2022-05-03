"""
Run a Flask REST API exposing a YOLOv5s model
"""

import argparse
import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, request, jsonify

from yolo_utils.torch_utils import select_device
from models.experimental import attempt_load
from yolo_utils.datasets import letterbox
from yolo_utils.utils import check_img_size, scale_coords, non_max_suppression

from utils.args import read_config, parse_arg_cfg
from utils.common import load_checkpoint
from utils.models import MODELS
from utils.transforms import TRANSFORMS

app = Flask(__name__)

DETECTION_URL = "/image"


yoloConfig = {
    'device': 'cpu',
    'weights': './yolo_weights/best_yolov5s_bdd.pt',
    'imgsz': 640,
    'confThres': 0.4,
    'iouThres': 0.5,
    'classes': None,
    'augment': False,
    'agnosticNMS': False,

    'model': None,
    'half': False,
    'names': None,
    'colors': None
}

laneNetConfig = {
    'model': None,
    'trans': None
}

retain_args = [
    'mixed_precision', 'pred', 'metric', 'mask_path', 'keypoint_path', 
    'gt_keypoint_path', 'checkpoint', 'device', 'image_suffix', 'keypoint_suffix', 
    'gt_keypoint_suffix', 'mask_suffix', 'use_color_pool', 'style'
]

lane_cfg = { }

laneNetArgs = argparse.Namespace(
    batch_size=None, cfg_options=None,
    checkpoint='./checkpoints/trained_model/resnet18_bezierlanenet_culane_aug1b_20211109.pt',
    config='./configs/lane_detection/bezierlanenet/resnet18_culane-aug1b.py',
    continue_from=None, device='cpu', dist_url=None, epochs=None, exp_name=None, gt_keypoint_path=None, gt_keypoint_suffix='.lines.txt',
    image_path='', image_suffix='',
    keypoint_path=None, keypoint_suffix='.lines.txt', lr=None, mask_path=None, mask_suffix='.png', metric='culane',
    mixed_precision=False,
    pred= True,
    save_dir=None,
    save_path='',
    style='point', thresh=None, use_color_pool=False, val_num_steps=None, warmup_steps=None, weight_decay=None,
    workers=None, world_size=None
)


def laneNetInit():

    global lane_cfg
    global laneNetArgs
    global laneNetConfig

    # Init lane_cfg 
    cfg = read_config(laneNetArgs.config)
    laneNetArgs, cfg = parse_arg_cfg(laneNetArgs, cfg)
    lane_cfg = cfg
    for k in retain_args:
        lane_cfg['test'][k] = vars(laneNetArgs)[k]

    # Load model
    laneNetConfig['model'] = MODELS.from_dict(lane_cfg['model'])
    load_checkpoint(net=laneNetConfig['model'], lr_scheduler=None, optimizer=None, filename=laneNetArgs.checkpoint)
    laneNetConfig['model'].to(laneNetArgs.device)
    laneNetConfig['trans'] = TRANSFORMS.from_dict(lane_cfg['test_augmentation']) # 未使用


def yoloInit():

    # Initialize
    yoloConfig['device'] = select_device(yoloConfig['device'])
    yoloConfig['half'] = yoloConfig['device'].type != 'cpu'

    # Load model
    yoloConfig['model'] = attempt_load(yoloConfig['weights'], map_location=yoloConfig['device'])
    yoloConfig['imgsz'] = check_img_size(yoloConfig['imgsz'], s=yoloConfig['model'].stride.max())
    if yoloConfig['half']:
        yoloConfig['model'].half()  # to FP16

    # Get names and colors
    yoloConfig['names'] = yoloConfig['model'].module.names if hasattr(yoloConfig['model'], 'module') else yoloConfig['model'].names
    yoloConfig['colors'] = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yoloConfig['names']))]



def laneNet(img):

    # Record origin image shape
    originShape = img.shape
    # img = Image.open(io.BytesIO(img)).convert('RGB')
    # originImg = transforms.Compose([
    #     transforms.PILToTensor()
    # ])(img)

    # Convert
    # img = laneNetConfig['trans'](img)
    img = cv2.resize( img, tuple(reversed(lane_cfg['test']['input_size'])) ) # resize
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # BGR to RGB
    img = torch.from_numpy(img).float().to(laneNetArgs.device)
    img = transforms.Compose([
        transforms.Normalize(
            mean = lane_cfg['test_augmentation']['transforms'][2]['mean'],
            std = lane_cfg['test_augmentation']['transforms'][2]['std']
        )
    ])(img)
    img = img.unsqueeze(0)
    img = img.to(laneNetArgs.device)
    
    # Inference
    keypoints = laneNetConfig['model'].inference(
        img,
        [ lane_cfg['test']['input_size'], originShape[0:2] ],
        lane_cfg['test']['gap'],
        lane_cfg['test']['ppl'],
        lane_cfg['test']['dataset_name'],
        lane_cfg['test']['max_lane']
    )
    keypoints = np.array(keypoints)
    keypoints = np.round(keypoints).astype(np.int16)

    return keypoints.tolist()



def yolov5(img):

    # Record origin image shape
    originShape = img.shape

    # Padded resize
    img = letterbox(img, new_shape=yoloConfig['imgsz'])[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(yoloConfig['device'])
    img = img.half() if yoloConfig['half'] else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    # Inference
    pred = yoloConfig['model'](img, augment=yoloConfig['augment'])[0]

    # Apply NMS
    det = non_max_suppression(pred, yoloConfig['confThres'], 
        yoloConfig['iouThres'], classes=yoloConfig['classes'], agnostic=yoloConfig['agnosticNMS'])[0]

    # Convert result
    result = []
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], originShape).round()
        for *xyxy, conf, cls in det:
            cxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
            obj = {
                'class': yoloConfig['names'][int(cls.item())],
                'color': yoloConfig['colors'][int(cls.item())],
                'coords': [int(i) for i in cxy],
                'confidence': conf.item()
            }
            result.append(obj)
        return result
    else:
        return False


@app.route(DETECTION_URL, methods=["POST"])
def predictImage():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        img = cv2.imdecode(np.frombuffer(im_bytes, np.uint8), cv2.IMREAD_COLOR)

        # yolov5s
        result = { 'yolo': None, 'lane': None }
        yoloResult = yolov5(img)
        result['yolo'] = yoloResult

        # laneNet
        laneNetResult = laneNet(img)
        result['lane'] = laneNetResult

        return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    yoloInit()
    laneNetInit()

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat