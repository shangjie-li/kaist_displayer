import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import pathlib as Path
import os
import argparse
import glob

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='DISPLAY')
    parser.add_argument('--image', default=None, type=str,
                        help='Path to image.')
    parser.add_argument('--images', default=None, type=str,
                        help='Root path to images.')

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parse_args()
    
    if args.image is None and args.images is None:
        print('Error: args.image or args.images must be set firstly.')
        exit()
    
    if args.image is not None:
        img = cv2.imread(args.image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        print('Press any key to stop.')
        key = cv2.waitKey(0)
        if key:
            exit()
    
    images_list = os.listdir(args.images)
    images_list.sort(key = lambda x: int(x.split('.')[0]))
    
    dataset_size = len(images_list)
    
    idx = 0
    while idx < dataset_size:
        print('Processing: %d / %d (%s)...' % (idx + 1, dataset_size, images_list[idx]))
        path = os.path.join(args.images, images_list[idx])
        
        img = cv2.imread(path)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        
        print('Press `Esc` to stop, `Space` to next one, `Q` to last one.')
        key = cv2.waitKey(0)
        if key == 27: # Esc
            break
        elif key == 32: # Space
            idx += 1
            idx = idx if idx < dataset_size else dataset_size - 1
            continue
        elif key == 113: # Q
            idx -= 1
            idx = idx if idx >= 0 else 0
            continue
        else:
            continue
