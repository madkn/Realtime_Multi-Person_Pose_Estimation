import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import pickle

param, model = config_reader()
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

if param['use_gpu']: 
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

import os
caffe.set_device(param['GPUdeviceNumber']) # set to your device!

curr_folder = '/'.join(os.path.abspath(__file__).split('/')[:-1])
net = caffe.Net(os.path.join(curr_folder, model['deployFile']),os.path.join(curr_folder, model['caffemodel']), caffe.TEST)


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

def calculate_heatmaps_pafs(oriImg):
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
#     # first figure shows padded images
#     f, axarr = plt.subplots(1, len(multiplier))
#     f.set_size_inches((20, 5))
#     # second figure shows heatmaps
#     f2, axarr2 = plt.subplots(1, len(multiplier))
#     f2.set_size_inches((20, 5))
#     # third figure shows PAFs
#     f3, axarr3 = plt.subplots(2, len(multiplier))
#     f3.set_size_inches((20, 10))

    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
        #print imageToTest_padded.shape

#         axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
#         axarr[m].set_title('Input image: scale %d' % m)

        net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        #net.forward() # dry run
        net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        start_time = time.time()
        output_blobs = net.forward()
       #print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        # visualization
#         axarr2[m].imshow(oriImg[:,:,[2,1,0]])
#         ax2 = axarr2[m].imshow(heatmap[:,:,3], alpha=.5) # right wrist
#         axarr2[m].set_title('Heatmaps (Rwri): scale %d' % m)

#         axarr3.flat[m].imshow(oriImg[:,:,[2,1,0]])
#         ax3x = axarr3.flat[m].imshow(paf[:,:,16], alpha=.5) # right elbow
#         axarr3.flat[m].set_title('PAFs (x comp. of Rwri to Relb): scale %d' % m)
#         axarr3.flat[len(multiplier) + m].imshow(oriImg[:,:,[2,1,0]])
#         ax3y = axarr3.flat[len(multiplier) + m].imshow(paf[:,:,17], alpha=.5) # right wrist
#         axarr3.flat[len(multiplier) + m].set_title('PAFs (y comp. of Relb to Rwri): scale %d' % m)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)
        
    return heatmap_avg, paf_avg
    

# f2.subplots_adjust(right=0.93)
# cbar_ax = f2.add_axes([0.95, 0.15, 0.01, 0.7])
# _ = f2.colorbar(ax2, cax=cbar_ax)

# f3.subplots_adjust(right=0.93)
# cbar_axx = f3.add_axes([0.95, 0.57, 0.01, 0.3])
# _ = f3.colorbar(ax3x, cax=cbar_axx)
# cbar_axy = f3.add_axes([0.95, 0.15, 0.01, 0.3])
# _ = f3.colorbar(ax3y, cax=cbar_axy)


import scipy

#plt.imshow(heatmap_avg[:,:,2])
from scipy.ndimage.filters import gaussian_filter

def calculate_all_peaks(heatmap_avg):
    all_peaks = []
    peak_counter = 0

    for part in range(19-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
#        print "id ", id
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks

def calculate_connection_all(paf_avg, all_peaks):
    connection_all = []
    special_k = []
    special_non_zero_index = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
#        print "-=========limb ", names[limbSeq[k][0]], names[limbSeq[k][1]]
        
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
#                    print candA, candB
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*paf_avg.shape[1]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))

            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break
            #print len(connection)
            connection_all.append(connection)
            
        elif(nA != 0 or nB != 0):
            special_k.append(k)
            special_non_zero_index.append(indexA if nA != 0 else indexB)
            connection_all.append([])
        else :
            connection_all.append([])

        
    return connection_all, special_k


def calculate_subset(all_peaks, connection_all, special_k):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:

            if len(connection_all) <= k or len(connection_all[k]) == 0:
                continue
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                   
                    subset = np.vstack([subset, row])
                    
    return subset, candidate
# # delete some rows of subset which has few parts occur
# deleteIdx = [];
# for i in range(len(subset)):
#     if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
#         deleteIdx.append(i)
# subset = np.delete(subset, deleteIdx, axis=0)



def calculate_pose(oriImg):
    heatmap_avg, paf_avg = calculate_heatmaps_pafs(oriImg)
    all_peaks = calculate_all_peaks(heatmap_avg)
    connection_all, special_k = calculate_connection_all(paf_avg, all_peaks)
    subset, candidate = calculate_subset(all_peaks, connection_all, special_k)
    if len(subset) == 0:
        return None, None, None, None, None, None, None
    person = max(subset, key = lambda x: x[-1])
    
    return connection_all, candidate, subset, person, all_peaks, heatmap_avg, paf_avg

#generates hitmap from the previously extracted data
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
