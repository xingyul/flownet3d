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
    def __init__(self, root='kitti_rm_ground', npoints=16384, train=True):
        self.npoints = npoints
        self.root = root
        self.train = train
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1']
                pos2 = data['pos2']
                flow = data['gt']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

            n1 = pos1.shape[0]
            n2 = pos2.shape[0]
            if n1 >= self.npoints:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
            if n2 >= self.npoints:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

            pos1_ = np.copy(pos1)[sample_idx1, :]
            pos2_ = np.copy(pos2)[sample_idx2, :]
            flow_ = np.copy(flow)[sample_idx1, :]

        color1 = np.zeros([self.npoints, 3])
        color2 = np.zeros([self.npoints, 3])
        mask = np.ones([self.npoints])

        return pos1_, pos2_, color1, color2, flow_, mask

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    import mayavi.mlab as mlab
    d = SceneflowDataset(root='kitti_rm_ground', npoints=16384)
    print(len(d))
    import time
    tic = time.time()
    for i in range(1, 100):
        pc1, pc2, color1, color2, flow, mask = d[i]
        print(pc1.shape, pc2.shape)
        continue

        mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        input()

        mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1, color=(0,0,1), line_width=0.2)
        input()

    print(time.time() - tic)
    print(pc1.shape, type(pc1))


