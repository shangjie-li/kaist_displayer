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
        description='JPG2PNG')
    parser.add_argument('--input_folder', default=None, type=str,
                        help='An input folder to images.')
    parser.add_argument('--output_folder', default=None, type=str,
                        help='An output folder to images.')

    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parse_args()
    
    if args.input_folder or args.output_folder:
        if args.input_folder is None or args.output_folder is None:
            print('Error: args.input_folder and args.output_folder must be set at the same time!')
            exit()
        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)
    
    files = glob.glob(os.path.join(args.input_folder, '*.jpg'))
    num = len(files)
    
    for idx in range(num):
        print('Processing: %d / %d...' % (idx + 1, num))
        
        input_name = os.path.basename(files[idx])
        print(input_name)
        img = cv2.imread(os.path.join(args.input_folder, input_name))
        
        output_name = '.'.join(input_name.split('.')[:-1]) + '.png'
        cv2.imwrite(os.path.join(args.output_folder, output_name), img)


