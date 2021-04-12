import os
from os import listdir

import cv2
import numpy as np

namelist = listdir('train')
print(namelist)
for item in namelist:
    os.rename('train/' + item, 'train/' + item.replace('.', ''))
namelist = listdir('train')
print(namelist)