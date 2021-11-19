import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import pathlib as Path
import os
import numpy as np
import random
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='KAIST Displayer')
    parser.add_argument('--dataset_root', default='/home/lishangjie/data/KAIST/kaist-paired', type=str,
                        help='Root directory path to dataset.')
    parser.add_argument('--save_folder1', default=None, type=str,
                        help='An output folder to save images.')
    parser.add_argument('--save_folder2', default=None, type=str,
                        help='An output folder to save images.')
    parser.add_argument('--save_split', default='selected.txt', type=str,
                        help='The output file for dataset ids.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--show_annotations', default=False, action='store_true',
                        help='Whether or not to show annotations.')
    parser.add_argument('--display', default=False, action='store_true',
                        help='Whether or not to dispaly.')
    parser.add_argument('--split', default='trainval', type=str,
                        help='The dataset split to consider. Only trainval and test are supported.')
    parser.add_argument('--name_length', default=6, type=int,
                        help='The length of name.')

    global args
    args = parser.parse_args(argv)

kaist_classes = ['person', 'cyclist', 'people', 'person?', 'unpaired']

def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b

def draw_annotation(img, boxes, labels, colors=None, classes=kaist_classes):
    """
    Inputs:
        img (ndarray, [h, w, 3]): image of BGR
        boxes (ndarray, [n, 4]): coordinates of boxes
        labels (ndarray, [n,]): labels
        colors (list[(b, g, r)]): colors of boxes and labels
        classes (list[str]): names of all classes
    Outputs:
        img (ndarray, [h, w, 3]): image of BGR
    """
    dim = len(boxes.shape)
    if dim != 2:
        raise ValueError('len(boxes.shape) must be 2.')
    
    white = (255, 255, 255)
    face = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.4
    thick = 1
    
    num = boxes.shape[0]
    if colors is None:
        colors = []
        for i in range(num):
            colors.append((create_random_color()))
    
    for i in range(num):
        c = colors[i]
        x1, y1, x2, y2 = boxes[i][:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness=1)
        
        text = classes[int(labels[i])]
        tw, th = cv2.getTextSize(text, face, scale, thick)[0]
        cv2.rectangle(img, (x1, y1), (x1 + tw, y1 + th + 3), c, -1)
        cv2.putText(img, text, (x1, y1 + th), face, scale, white, thick, cv2.LINE_AA)
    
    return img

if __name__ == '__main__':
    parse_args()
    data_root = args.dataset_root
    
    save_mode = False
    if args.save_folder1 or args.save_folder2:
        if args.save_folder1 is None or args.save_folder2 is None:
            print('Error: args.save_folder1 and args.save_folder2 must be set at the same time!')
            exit()
        save_mode = True
        if not os.path.exists(args.save_folder1):
            os.mkdir(args.save_folder1)
        if not os.path.exists(args.save_folder2):
            os.mkdir(args.save_folder2)
    
    dataset_ids = []
    for line in open(os.path.join(data_root, 'splits', args.split + '.txt')):
        split = line.strip().split('/')
        p = split[0] + '/' + split[1]
        i = split[2]
        # dataset_ids: [('set00/V000', 'I01217'), ...]
        dataset_ids.append((p, i))
    
    if save_mode and args.save_split is not None:
        with open(args.save_split, 'w') as f:
            f.seek(0)
            f.truncate()
    
    dataset_size = len(dataset_ids) if args.max_images < 0 else min(args.max_images, len(dataset_ids))
    for idx in range(dataset_size):
        print('\n--------[%d/%d]--------' % (idx + 1, dataset_size))
        img_id = dataset_ids[idx]
        img_path_c = os.path.join(data_root, 'images', img_id[0], 'visible', img_id[1] + '.jpg')
        img_path_t = os.path.join(data_root, 'images', img_id[0], 'lwir', img_id[1] + '.jpg')
        
        print('img_c: %s' % img_path_c)
        print('img_t: %s' % img_path_t)
        
        if save_mode:
            save_path_c = os.path.join(args.save_folder1, str(idx).zfill(args.name_length) + '.jpg')
            save_path_t = os.path.join(args.save_folder2, str(idx).zfill(args.name_length) + '.jpg')
        
        if save_mode and args.save_split is not None:
            with open(args.save_split, 'a') as f:
                f.write(img_id[0] + '/' + img_id[1] + '\n')
        
        img_c = cv2.imread(img_path_c)
        img_t = cv2.imread(img_path_t)
        
        if save_mode and not args.show_annotations:
            cv2.imwrite(save_path_c, img_c)
            cv2.imwrite(save_path_t, img_t)
            print('Saving image to %s.' % save_path_c)
            print('Saving image to %s.' % save_path_t)
        
        ann_path = os.path.join(data_root, 'annotations', img_id[0], 'visible', img_id[1] + '.txt')
        names, boxes, labels = [], [], []
        for line in open(ann_path):
            split = line.strip().split(' ')
            if split[0] != '%':
                x, y, w, h = int(split[1]), int(split[2]), int(split[3]), int(split[4])
                boxes.append([x, y, x + w, y + h])
                names.append([split[0]])
                try:
                    labels.append([kaist_classes.index(split[0])])
                except:
                    continue
        if len(names) > 0:
            names, boxes, labels = np.array(names), np.array(boxes), np.array(labels)
            img_c_annotated = draw_annotation(img_c.copy(), boxes, labels)
            img_t_annotated = draw_annotation(img_t.copy(), boxes, labels)
        else:
            img_c_annotated = img_c.copy()
            img_t_annotated = img_t.copy()
        
        if save_mode and args.show_annotations:
            cv2.imwrite(save_path_c, img_c_annotated)
            cv2.imwrite(save_path_t, img_t_annotated)
            print('Saving image to %s.' % save_path_c)
            print('Saving image to %s.' % save_path_t)
        
        if args.display:
            if args.show_annotations:
                cv2.imshow('img_c', img_c_annotated)
                cv2.imshow('img_t', img_t_annotated)
            else:
                cv2.imshow('img_c', img_c)
                cv2.imshow('img_t', img_t)
            key = cv2.waitKey(0)
            if key == 27:
                break
            else:
                continue

