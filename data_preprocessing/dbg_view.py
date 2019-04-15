





import numpy as np
import os

data_dir = '.'
filename = 'test.npz'
# data_dir = '/scr-ssd/Projects/flownet3d_private/data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k_12288'
# filename = 'TRAIN_A_0376_right_0009-0.npz'
# data_dir = '/scr-ssd/Projects/flownet3d_private/data_processed_maxcut_35_color_both_mask_20k_2k'
# filename = 'TRAIN_A_0376_right_0009-2.npz'

data = np.load(os.path.join(data_dir, filename), filename)

points1 = data['points1']
points2 = data['points2']
color1 = data['color1']
color2 = data['color2']
flow = data['flow']
valid_mask1 = data['valid_mask1']

points1_valid = points1[valid_mask1]
points1_nonvalid = points1[np.logical_not(valid_mask1)]

n1_valid = points1_valid.shape[0]
n1_nonvalid = points1_nonvalid.shape[0]
n2 = points2.shape[0]

f = open('view.pts', 'w')

for i in range(n1_valid):
    # f.write('{} {} {} {} {} {}\n'.format(points1_valid[i, 0], points1_valid[i, 1], points1_valid[i, 2], 2*color1[i, 0]-1, 2*color1[i, 1]-1, 2*color1[i, 2]-1))
    f.write('{} {} {} {} {} {}\n'.format(points1_valid[i, 0], points1_valid[i, 1], points1_valid[i, 2], 1, -1, -1))
for i in range(n1_nonvalid):
    f.write('{} {} {} {} {} {}\n'.format(points1_nonvalid[i, 0], points1_nonvalid[i, 1], points1_nonvalid[i, 2], -1, -1, -1))

for i in range(n1_valid + n1_nonvalid):
    f.write('{} {} {} {} {} {}\n'.format((points1 + flow)[i, 0], (points1 + flow)[i, 1], (points1 + flow)[i, 2], -1, -1, 1))
    pass

for i in range(n2):
    # f.write('{} {} {} {} {} {}\n'.format(points2[i, 0], points2[i, 1], points2[i, 2], 2*color2[i, 0]-1, 2*color2[i, 1]-1, 2*color2[i, 2]-1))
    f.write('{} {} {} {} {} {}\n'.format(points2[i, 0], points2[i, 1], points2[i, 2], -1, 1, -1))
    pass

