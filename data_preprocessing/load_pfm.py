#!/usr/bin/python
# Adapted from https://gist.github.com/chpatrick/8935738


import numpy as np
import re

def load_pfm(filename):
     file = open(filename, 'r', newline='', encoding='latin-1')
     color = None
     width = None
     height = None
     scale = None
     endian = None

     header = file.readline().rstrip()
     if header == 'PF':
         color = True
     elif header == 'Pf':
         color = False
     else:
         raise Exception('Not a PFM file.')

     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
     if dim_match:
         width, height = map(int, dim_match.groups())
     else:
         raise Exception('Malformed PFM header.')

     scale = float(file.readline().rstrip())
     if scale < 0: # little-endian
         endian = '<'
         scale = -scale
     else:
         endian = '>' # big-endian

     data = np.fromfile(file, endian + 'f')
     shape = (height, width, 3) if color else (height, width)

     file.close()

     return np.reshape(data, shape), scale

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    img, _ = load_pfm(sys.argv[1])
    img = img[::-1, :]
    imgplot = plt.imshow( 1050 / img)
    plt.colorbar()
    plt.show()
