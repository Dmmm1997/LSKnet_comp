# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import csv
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np
import sys
from mmrotate.datasets.data30 import Data30


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--data-dir',
        default="/mnt/data/DM_Data/rs_rotatedet_comp/test_fs",
        help='Image file')
    parser.add_argument(
        '--out-dir',
        default=
        "work_dirs/lsk_s_ema_fpn_1x_data30_le90_aug-noise-blur-brightness_tta2flip_finetune_again/ep12_tta3flip/visual_csv",
        help='Path to output file')
    parser.add_argument(
        '--csvfile',
        default="work_dirs/lsk_s_ema_fpn_1x_data30_le90_aug-noise-blur-brightness_tta2flip_finetune_again/ep12_tta3flip/results.csv",
        help='Path to output file')
    parser.add_argument(
        '--nums', default=200, type=int, help='Path to output file')
    parser.add_argument(
        '--score-thr', type=float, default=0.05, help='bbox score threshold')
    args = parser.parse_args()
    return args


color_dict = {}
for name, color in zip(Data30.CLASSES, Data30.PALETTE):
    color_dict[name] = color


def read_from_csv(file):
    info_dict = defaultdict(list)
    with open(file, 'r') as F:
        reader = csv.reader(F)
        next(reader)
        for row in reader:
            img_name, cls_name, X1, Y1, X2, Y2, X3, Y3, X4, Y4, score = row
            info_dict[img_name].append([
                cls_name,
                int(float(X1)),
                int(float(Y1)),
                int(float(X2)),
                int(float(Y2)),
                int(float(X3)),
                int(float(Y3)),
                int(float(X4)),
                int(float(Y4)),
                float(score)
            ])
    return info_dict


def visulization(image, qboxes, cls_names, scores, score_thr):
    for qbox, cls_name, score in zip(qboxes, cls_names, scores):
        if score < score_thr:
            continue
        cv2.line(
            image,
            pt1=(qbox[0], qbox[1]),
            pt2=(qbox[2], qbox[3]),
            color=color_dict[cls_name],
            thickness=2)
        cv2.line(
            image,
            pt1=(qbox[2], qbox[3]),
            pt2=(qbox[4], qbox[5]),
            color=color_dict[cls_name],
            thickness=2)
        cv2.line(
            image,
            pt1=(qbox[4], qbox[5]),
            pt2=(qbox[6], qbox[7]),
            color=color_dict[cls_name],
            thickness=2)
        cv2.line(
            image,
            pt1=(qbox[6], qbox[7]),
            pt2=(qbox[0], qbox[1]),
            color=color_dict[cls_name],
            thickness=2)

        # text = "{}|{:.2f}".format(cls_name,score)
        ind_random = np.random.choice(range(0, 8, 2), 1)[0]
        text = "{:.2f}".format(score)
        cv2.putText(image, text, (qbox[ind_random], qbox[ind_random + 1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dict[cls_name], 2)
    return image


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    info_dict = read_from_csv(args.csvfile)
    ind = 0

    for img_name, info_list in tqdm(info_dict.items()):
        if ind > args.nums:
            sys.exit(0)

        image_path = os.path.join(args.data_dir, img_name)
        image = cv2.imread(image_path)
        # if image.shape[0]==1024 and image.shape[1]==1024:
        #     continue

        cls_names = [info[0] for info in info_list]
        qboxes = [info[1:-1] for info in info_list]
        scores = [info[-1] for info in info_list]
        image = visulization(image, qboxes, cls_names, scores, args.score_thr)

        save_path = os.path.join(args.out_dir, img_name)
        save_path = save_path.replace(".bmp", ".jpg")
        cv2.imwrite(save_path, image)
        ind += 1


main()