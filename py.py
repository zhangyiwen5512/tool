#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os, cv2
from timer import Timer
import argparse
import numpy as np
import torch

print (time.strftime('End Time: %Y.%m.%d %H:%M:%S', time.localtime(time.time())))

learning_rate = 0.00025
learning_rate /= (10 ** (35000 // 450000))
print(learning_rate)
exit(0)


datasets = {
            'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
            'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
            'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
            'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
            'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
            'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
            'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
            'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
            'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
            'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
            'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}

timer = time.time()

def CDnet(name):
    # Load the demo image
    #im_file = '/media/zhangyiwen/zhangyiwen/CDW2014/dataset/' + name + '/' + 'input'
    im_file = '/home/zhangyiwen/Desktop/CornerNet/data/coco/images/minval2014'
    imgs = os.listdir(im_file)
    lens = len(imgs)
    for i in range(lens):
        if i == 0 :
            continue

        l = str(i+549323)
        #if l != 6:
         # number = '0' * (6 - len(l)) + l
        if l != 12:
          number = '0' * (12 - len(l)) + l
        #image = im_file + '/in' + number + '.jpg'
        image = im_file + '/COCO_val2014_' + number + '.jpg'#COCO_val2014_000000000042
        print(image)
        now = time.time()
        im = cv2.imread(image)
  #      im = im[:, :, (2, 1, 0)]
        print(im.shape)
        image_height, image_width = im.shape[0:2]
        print(image_height, image_width)
        bbox = [10,10, 40,40]

        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cv2.rectangle(im, (x,y), (x + w, y + h), (0, 0, 255), 1)
   #     font = cv2.InitFont(cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,1,0,3,8)
        cv2.putText(im, str(30), (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,255),1)




        im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_LINEAR)
        total_time = time.time() - now
        cv2.putText(im, "FPS: %.2f" % (1.0 / total_time), (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),2)
        cv2.imshow("image", im)
        cv2.waitKey(0)



for category, scene_list in datasets.items():
    for scene in scene_list:
        print('demo in ' + category + '/' + scene)
        CDnet(category + '/' + scene)
