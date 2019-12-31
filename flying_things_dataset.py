'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', npoints=2048, train=True):
        self.npoints = npoints
        self.train = train
        self.root = root
        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1']
                pos2 = data['points2']
                color1 = data['color1'] / 255
                color2 = data['color2'] / 255
                flow = data['flow']
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1_ = np.copy(pos1[sample_idx1, :])
            pos2_ = np.copy(pos2[sample_idx2, :])
            color1_ = np.copy(color1[sample_idx1, :])
            color2_ = np.copy(color2[sample_idx2, :])
            flow_ = np.copy(flow[sample_idx1, :])
            mask1_ = np.copy(mask1[sample_idx1])
        else:
            pos1_ = np.copy(pos1[:self.npoints, :])
            pos2_ = np.copy(pos2[:self.npoints, :])
            color1_ = np.copy(color1[:self.npoints, :])
            color2_ = np.copy(color2[:self.npoints, :])
            flow_ = np.copy(flow[:self.npoints, :])
            mask1_ = np.copy(mask1[:self.npoints])

        return pos1_, pos2_, color1_, color2_, flow_, mask1_

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048)
    print(len(d))
    import time
    tic = time.time()
    for i in range(100):
        pc1, pc2, c1, c2, flow, m1, m2 = d[i]

        print(pc1.shape)
        print(pc2.shape)
        print(flow.shape)
        print(np.sum(m1))
        print(np.sum(m2))
        pc1_m1 = pc1[m1==1,:]
        pc1_m1_n = pc1[m1==0,:]
        print(pc1_m1.shape)
        print(pc1_m1_n.shape)
        mlab.points3d(pc1_m1[:,0], pc1_m1[:,1], pc1_m1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc1_m1_n[:,0], pc1_m1_n[:,1], pc1_m1_n[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()

        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()
        mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1)
        raw_input()

    print(time.time() - tic)
    print(pc1.shape, type(pc1))


