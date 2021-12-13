import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

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
import time

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='KAIST2ROSmsg')
    parser.add_argument('--images1', default=None, type=str,
                        help='An input folder to images.')
    parser.add_argument('--images2', default=None, type=str,
                        help='An input folder to images.')
    parser.add_argument('--start_index', default=0, type=int,
                        help='The start index of image.')
    parser.add_argument('--end_index', default=-1, type=int,
                        help='The end index of image.')
    parser.add_argument('--frame_rate', default=30, type=int,
                        help='The frame rate to play.')
    parser.add_argument('--pub_topic1', default='image1', type=str,
                        help='The name of ROS topic.')
    parser.add_argument('--pub_topic2', default='image2', type=str,
                        help='The name of ROS topic.')

    global args
    args = parser.parse_args(argv)

def publish_image(pub, data, frame_id='base_link'):
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id
    
    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tostring()
    msg.header = header
    msg.step = msg.width * 1 * 3
    
    pub.publish(msg)

if __name__ == '__main__':
    parse_args()
    
    if args.images1 is None or args.images2 is None:
        print('Error: args.images1 and args.images2 must be set firstly.')
        exit()
    
    images1_list = os.listdir(args.images1)
    images1_list.sort(key = lambda x: int(x.split('.')[0]))
    
    images2_list = os.listdir(args.images2)
    images2_list.sort(key = lambda x: int(x.split('.')[0]))
    
    assert len(images1_list) == len(images2_list), \
        'The number of files in %s is not equal to %s.' % (args.images1, args.images2)
    dataset_size = len(images1_list)
    
    assert dataset_size > 0, 'The dataset is empty.'
    assert args.start_index >= 0 and args.start_index < dataset_size, \
        'args.start_index must be between [0, %d).' % dataset_size
    
    if args.end_index == -1: args.end_index = dataset_size
    assert args.end_index > 0 and args.end_index <= dataset_size, \
        'args.end_index must be between (0, %d].' % dataset_size
    
    total_time = dataset_size / args.frame_rate
    interval = 1 / args.frame_rate
    
    rospy.init_node('kaist2rosmsg', anonymous=True, disable_signals=True) # `disable_signals=True` allows ctrl+c to stop
    pub1 = rospy.Publisher(args.pub_topic1, Image, queue_size=1)
    pub2 = rospy.Publisher(args.pub_topic2, Image, queue_size=1)
    
    try:
        for idx in range(args.start_index, args.end_index):
            path1 = os.path.join(args.images1, images1_list[idx])
            path2 = os.path.join(args.images2, images2_list[idx])
            
            img1 = cv2.imread(path1)[:, :, ::-1] # To RGB
            img2 = cv2.imread(path2)[:, :, ::-1] # To RGB
            
            publish_image(pub1, img1)
            publish_image(pub2, img2)
            
            current_time = time.time()
            elapsed_time = (idx + 1) * interval
            print(' [RUNNING]  Current Time: %f  Duration: %f / %f' % (current_time, elapsed_time, total_time))
            time.sleep(interval)
            
    except KeyboardInterrupt:
        rospy.signal_shutdown("Everything is over now.")

