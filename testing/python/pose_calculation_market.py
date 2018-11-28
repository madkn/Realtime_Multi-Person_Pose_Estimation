import cv2 as cv
import numpy as np
import pickle


from torchreid.datasets import init_imgreid_dataset
import os
import sys

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
