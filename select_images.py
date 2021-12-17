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
        description='SELECT')
    parser.add_argument('--input_folder1', default=None, type=str,
                        help='An input folder to read images.')
    parser.add_argument('--input_folder2', default=None, type=str,
                        help='An input folder to read images.')
    parser.add_argument('--output_folder1', default=None, type=str,
                        help='An output folder to save images.')
    parser.add_argument('--output_folder2', default=None, type=str,
                        help='An output folder to save images.')
    parser.add_argument('--name_length', default=6, type=int,
                        help='The length of name of saving image.')
    parser.add_argument('--start_id_for_saving', default=0, type=int,
                        help='The id of saving image to start.')

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parse_args()
    
    if args.input_folder1 is None or args.input_folder2 is None:
        print('Error: args.input_folder1 and args.input_folder2 must be set firstly.')
        exit()
    if args.output_folder1 is None or args.output_folder2 is None:
        print('Error: args.output_folder1 and args.output_folder2 must be set firstly.')
        exit()
    
    if not os.path.exists(args.output_folder1):
        os.mkdir(args.output_folder1)
    if not os.path.exists(args.output_folder2):
        os.mkdir(args.output_folder2)
    
    images1_list = os.listdir(args.input_folder1)
    images1_list.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))
    
    images2_list = os.listdir(args.input_folder2)
    images2_list.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))
    
    assert len(images1_list) == len(images2_list), 'The number of files in %s is not equal to %s.' % \
        (args.input_folder1, args.input_folder2)
    dataset_size = len(images1_list)
    
    idx = 0
    save_idx = 0
    while idx < dataset_size:
        print('Processing: %d / %d...' % (idx + 1, dataset_size))
        path1 = os.path.join(args.input_folder1, images1_list[idx])
        path2 = os.path.join(args.input_folder2, images2_list[idx])
        
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        save_path1 = os.path.join(args.output_folder1, str(save_idx + args.start_id_for_saving).zfill(args.name_length) + '.jpg')
        save_path2 = os.path.join(args.output_folder2, str(save_idx + args.start_id_for_saving).zfill(args.name_length) + '.jpg')
        
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        
        print('Press `Esc` to stop, `Enter` to save and continue, `Space` to ignore and continue.')
        key = cv2.waitKey(0)
        if key == 27: # Esc
            break
        elif key == 13: # Enter
            cv2.imwrite(save_path1, img1)
            cv2.imwrite(save_path2, img2)
            idx += 1
            save_idx += 1
            continue
        elif key == 32: # Space
            idx += 1
            continue
        else:
            continue


