# ******************************************************
# Author           : liuyang && jnulzl
# Contributed by   : José Carlos Reyes Hernández
# Last modified    : 2024-19-03 13:02
# Filename         : inference.py
# ******************************************************
from __future__ import absolute_import
import argparse
import time

import numpy as np
import torch
import os
import glob
import cv2
from core.workspace import create, load_config
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from data.transform.image_util import normalize_img
from data.anchors_opr.generate_anchors import GeneartePriorBoxes

parser = argparse.ArgumentParser(description="Test Details")
parser.add_argument(
    "--weight_path",
    default="snapshots/MogFace_Ali-AMS/model_70000.pth",
    type=str,
    help="The weight path.",
)
parser.add_argument("--source", default="", type=str, help="img path")
parser.add_argument("--nms_th", default=0.3, type=float, help="nms threshold.")
parser.add_argument(
    "--pre_nms_top_k", default=5000, type=int, help="number of max score image."
)
parser.add_argument("--score_th", default=0.9, type=float, help="score threshold.")
parser.add_argument(
    "--max_bbox_per_img", default=750, type=int, help="max number of det bbox."
)
parser.add_argument(
    "--config", "-c", default="./configs/mogface/MogFace_E.yml", type=str, help="config yml."
)
parser.add_argument("--test_idx", default=None, type=int)
parser.add_argument(
    "--output_dir",
    default="./predictions",
    type=str,
    help="Directory where to save the images and labels",
)


class DataSetting(object):
    def __init__(self):
        pass


def detect_face(net, image, shrink, generate_anchors_fn):
    with torch.inference_mode():
        x = image
        if shrink != 1:
            x = cv2.resize(
                image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR
            )

        # print('shrink:{}'.format(shrink))

        width = x.shape[1]
        height = x.shape[0]
        print("width: {}, height: {}".format(width, height))

        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0)
        x = x.cuda()

        out = net(x)

        anchors = anchor_utils.transform_anchor((generate_anchors_fn(height, width)))
        anchors = torch.FloatTensor(anchors).cuda()
        decode_bbox = anchor_utils.decode(out[1].squeeze(0), anchors)
        boxes = decode_bbox
        scores = out[0].squeeze(0)  # [N,1] o [N,2] normalmente

        # ---- convertir a probabilidades en [0,1] ----
        if scores.dim() == 1:  # raro, pero por si viene [N]
            probs = torch.sigmoid(scores)
        elif scores.shape[1] == 1:
            probs = torch.sigmoid(scores[:, 0])          # binario 1 canal
        elif scores.shape[1] >= 2:
            probs = torch.softmax(scores, dim=1)[:, 1]   # binario 2 canales -> clase "face" = idx 1
        else:
            raise RuntimeError(f"Shape de cls inesperado: {scores.shape}")

        scores = probs 

        select_idx_list = []
        tmp_height = height
        tmp_width = width

        test_idx = args.test_idx
        if test_idx is not None:
            for i in range(2):
                tmp_height = (tmp_height + 1) // 2
                tmp_width = (tmp_width + 1) // 2

            for i in range(6):
                if i == 0:
                    select_idx_list.append(tmp_height * tmp_width)
                else:
                    select_idx_list.append(
                        tmp_height * tmp_width + select_idx_list[i - 1]
                    )
                tmp_height = (tmp_height + 1) // 2
                tmp_width = (tmp_width + 1) // 2

            if test_idx == 2:
                boxes = boxes[: select_idx_list[(test_idx - 2)]]
                scores = scores[: select_idx_list[(test_idx - 2)]]
            else:
                boxes = boxes[
                    select_idx_list[test_idx - 3] : select_idx_list[test_idx - 2]
                ]
                scores = scores[
                    select_idx_list[test_idx - 3] : select_idx_list[test_idx - 2]
                ]

        # print('scores shape', scores.shape)
        # print('boxes shape', boxes.shape)
        top_k = args.pre_nms_top_k
        v, idx = scores.sort(0)
        idx = idx[-top_k:]
        boxes = boxes[idx]
        scores = scores[idx]

        # [11620, 4]
        boxes = boxes.cpu().numpy()
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        boxes[:, 0] /= shrink
        boxes[:, 1] /= shrink
        boxes[:, 2] = boxes[:, 0] + w / shrink - 1
        boxes[:, 3] = boxes[:, 1] + h / shrink - 1
        # boxes = boxes / shrink
        # [11620, 2]
        scores = scores.cpu().numpy()

        inds = np.where(scores > args.score_th)[0]
        if len(inds) == 0:
            det = np.empty([0, 5], dtype=np.float32)
            return det
        c_bboxes = boxes[inds]
        # [5,]
        c_scores = scores[inds]
        # [5, 5]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False
        )

        keep = nms(c_dets, args.nms_th)
        c_dets = c_dets[keep, :]

        max_bbox_per_img = args.max_bbox_per_img
        if max_bbox_per_img > 0:
            image_scores = c_dets[:, -1]
            if len(image_scores) > max_bbox_per_img:
                image_thresh = np.sort(image_scores)[-max_bbox_per_img]
                keep = np.where(c_dets[:, -1] >= image_thresh)[0]
                c_dets = c_dets[keep, :]

        for i in range(c_dets.shape[0]):
            if c_dets[i][0] < 0.0:
                c_dets[i][0] = 0.0

            if c_dets[i][1] < 0.0:
                c_dets[i][1] = 0.0

            if c_dets[i][2] > width - 1:
                c_dets[i][2] = width - 1

            if c_dets[i][3] > height - 1:
                c_dets[i][3] = height - 1

        return c_dets


def process_img(img, net, generate_anchors_fn, normalize_setting):
    """
    Process a single image to detect faces.

    Args:
        - img (numpy.ndarray): Input image.
        - net (torch.nn.Module): Neural network model.
        - generate_anchors_fn (function): Function to generate anchors.
        - normalize_setting: Normalization setting.

    Returns:
        numpy.ndarray: Detected faces with bounding boxes and scores.
    """
    with torch.inference_mode():
        img = normalize_img(img.astype(np.float32), normalize_setting)
        max_im_shrink = (0x7FFFFFFF / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5
        max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        boxes = detect_face(net, img, shrink, generate_anchors_fn)
        return boxes


def process_images_in_directory(
    input_dir, output_dir, net, generate_anchors_fn, normalize_setting
):
    """
    Process images in a directory to detect faces and save results.

    Args:
        - input_dir (str): Input directory containing images.
        - output_dir (str): Output directory to save processed images and labels.
        - net (torch.nn.Module): Neural network model.
        - generate_anchors_fn (function): Function to generate anchors.
        - normalize_setting: Normalization setting.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    for img_path in image_files:
        with torch.inference_mode():
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Unable to load image {img_path}")
                continue

            img_show = img.copy()
            t1 = time.time()
            boxes = process_img(img, net, generate_anchors_fn, normalize_setting)
            t2 = time.time()
            print("Inference time: %d ms" % ((t2 - t1) * 1000))

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    img_show = cv2.rectangle(
                        img_show,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255),
                        4,
                    )

                img_filename = os.path.basename(img_path)
                cv2.imwrite(os.path.join(output_dir, "images", img_filename), img_show)

                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                with open(os.path.join(output_dir, "labels", label_filename), "w") as f:
                    for box in boxes:
                        if boxes is not None and len(boxes) > 0:
                            # box = [x1, y1, x2, y2, score] (c_dets tras NMS)
                            x1, y1, x2, y2, score = map(float, box[:5])
                            # Guardar en formato: x1 y1 x2 y2 score
                            f.write(f"{score:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")
                            print(f"Faces detected in image {img_path}.")
                        else:
                            print(f"No faces detected in image {img_path}.")   


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)

    generate_anchors_fn = GeneartePriorBoxes(
        scale_list=cfg["GeneartePriorBoxes"]["scale_list"],
        aspect_ratio_list=cfg["GeneartePriorBoxes"]["aspect_ratio_list"],
        stride_list=cfg["GeneartePriorBoxes"]["stride_list"],
        anchor_size_list=cfg["GeneartePriorBoxes"]["anchor_size_list"],
    )

    normalize_setting = DataSetting()
    normalize_setting.use_rgb = cfg["BasePreprocess"]["use_rgb"]
    normalize_setting.img_mean = (
        np.array(cfg["BasePreprocess"]["img_mean"]).astype("float32") * 255
    )[::-1]
    normalize_setting.img_std = (
        np.array(cfg["BasePreprocess"]["img_std"]).astype("float32") * 255
    )[::-1]
    normalize_setting.normalize_pixel = cfg["BasePreprocess"]["normalize_pixel"]

    # Create and load the model onto CUDA
    net = create(cfg["architecture"])
    print("Load model from {}".format(args.weight_path))
    net.load_state_dict(torch.load(args.weight_path))
    net = net.cuda()  # Ensure the model is on CUDA
    net.eval()

    onnx_path = args.weight_path.replace(".pth", ".onnx")
    if not os.path.exists(onnx_path):
        img = torch.rand(1, 3, 640, 640)
        model = net.cpu()
        torch.onnx.export(
            model,
            img,
            onnx_path,
            verbose=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output1", "output2"],
        )
        
        net = net.cuda() # Move the model back to CUDA after exporting

    input_dir = args.source
    output_dir = args.output_dir

    print("Finish load model.")

    process_images_in_directory(
        input_dir, output_dir, net, generate_anchors_fn, normalize_setting
    )
