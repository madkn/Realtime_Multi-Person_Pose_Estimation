import cv2 as cv
import numpy as np
import pickle


from torchreid.datasets import init_imgreid_dataset
import os
import sys

#set the srouce and target paths here:
#data_root - root folder where the folder names 'market1501' is
from ..paths import data_preparation_root, data_root, market_pose_label_path

dataset_name = "market1501"

#use data parsers for standard datasets from https://github.com/madkn/deep-person-reid
dataset = init_imgreid_dataset(
    root=data_root, name=dataset_name, relabel_train = False
)

all_set = dataset.train + dataset.query + dataset.gallery

def ensure_dir(file_path):
    #directory = os.path.dirname(file_path)
    # print person_folder_target, os.path.exists(file_path)
    if not os.path.exists(file_path):

        os.makedirs(file_path)


target_root = '/'.join(market_pose_label_path.split('/')[:-1])

import time
start = time.time()
count_person = 0

from pose_calculation import calculate_pose

pose_data = dict()



for triple in all_set:
    im_path = triple[0]
    im = cv.imread(im_path)
    fn = im_path.split('/')[-1]
    connection_all, candidate, subset, person, all_peaks, heatmap_avg, paf_avg = calculate_pose(im)
    if candidate is None:
        continue


    pose_data[fn.split('.jpg')[0]] = dict()
    pose_data[fn.split('.jpg')[0]]['subset'] = subset
    pose_data[fn.split('.jpg')[0]]['points'] = dict()
    for i in xrange(len(candidate)):
        pose_data[fn.split('.jpg')[0]]['points'][candidate[i][-1]] = (candidate[i][0], candidate[i][1])

    if count_person % 100 == 0:
        print count_person
        sys.stdout.flush()

    count_person+=1




print "seen ",  count_person
print time.time() - start


pickle.dump(pose_data, open(market_pose_label_path, 'wb'))


#generates hitmap from the previously extracted data (dict : filename -> 'subset' dict, 'points' dict)
def get_gt_maps_const(subset, points, output_size, radius = 5., initial_size=(128, 64)):
    if not output_size:
        hitmaps_gt = np.zeros((18,initial_size[0], initial_size[1]))
    else:
        hitmaps_gt = np.zeros((18,output_size[0], output_size[1]))

    # take the person with largest number of points
    person = max(subset, key=lambda x: x[-1])

    H = output_size[0] if output_size else initial_size[0]
    W = output_size[1] if output_size else initial_size[1]
    if output_size:
        radius = (radius / initial_size[0]  * output_size[0]  +
                  radius / initial_size[1]  * output_size[1]) / 2

    #print radius
    for i in xrange(len(person) - 2):
        if person[i] == -1:
            continue
        anchor = points[person[i]][1], points[person[i]][0]

        if output_size:
            anchor = anchor[0] / initial_size[0] * output_size[0], anchor[1] / initial_size[1] * output_size[1]

        #print anchor, int(round(anchor[0] - radius)), int(round(anchor[0] + radius))
        for h in xrange(int(round(anchor[0] - radius)), int(round(anchor[0] + radius))+1):
            for w in xrange(int(round(anchor[1] - radius)), int(round(anchor[1] + radius))+1):
                if h < 0 or h >= H or w < 0 or w >= W:
                    continue
                dist_sq = np.sqrt((h - anchor[0]) ** 2 + (w - anchor[1]) ** 2)
                if dist_sq <radius+1 :
                    hitmaps_gt[i, h, w] = 1

    return hitmaps_gt


import cv2


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def draw_pose(subset, points, output_size, stickwidth = 3, radius = 5., initial_size=(128, 64)):
    if not output_size:
        canvas = np.zeros((initial_size[0], initial_size[1], 3), dtype = np.uint8)
    else:
        canvas = np.zeros((output_size[0], output_size[1], 3), dtype = np.uint8)

    # take the person with largest number of points
    person = max(subset, key=lambda x: x[-1])
    if output_size:
        radius = (radius / initial_size[0] * output_size[0]  +
                  radius / initial_size[1] * output_size[1]) / 2


    for i in range(17):
        index = person[np.array(limbSeq[i])-1]

        if -1 in index:
            continue

        Y = [points[index[0]][0], points[index[1]][0]]
        X = [points[index[0]][1], points[index[1]][1]]

        if output_size:
            Y = Y[0] / initial_size[0] * output_size[0], Y[1] / initial_size[1] * output_size[1]
            X = X[0] / initial_size[0] * output_size[0], X[1] / initial_size[1] * output_size[1]

        cv2.line(canvas, pt1 = (int(Y[0]), int(X[0])), pt2 = (int(Y[1]), int(X[1])), color = colors[i], thickness = stickwidth)



    for i in range(17):
        j1,j2 = limbSeq[i]

        index = person[np.array(limbSeq[i])-1]

        if -1 in index:
            continue
        Y = [points[index[0]][0], points[index[1]][0]]
        X = [points[index[0]][1], points[index[1]][1]]

        if output_size:
            Y = Y[0] / initial_size[0] * output_size[0], Y[1] / initial_size[1] * output_size[1]
            X = X[0] / initial_size[0] * output_size[0], X[1] / initial_size[1] * output_size[1]


        cv2.circle(canvas, (int(Y[0]), int(X[0])), int(radius), colors[j1-1], thickness=-1)
        cv2.circle(canvas, (int(Y[1]), int(X[1])), int(radius), colors[j2-1], thickness=-1)

    return canvas






pose_data = pickle.load(open(market_pose_label_path, 'rb'))
print pose_data.keys()[0]

print pose_data['0816_c6s2_095218_02']['subset']
print pose_data['0816_c6s2_095218_02']['points']

canvas = draw_pose(pose_data['0816_c6s2_095218_02']['subset'],
               pose_data['0816_c6s2_095218_02']['points'],
               output_size = (128, 64), stickwidth = 3, radius = 5., initial_size=(128, 64))

import matplotlib.pyplot as plt
plt.imshow(canvas)
plt.show()
