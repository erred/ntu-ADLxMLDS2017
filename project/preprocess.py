from skimage.io import imread, imsave
from skimage.transform import resize
import os

counter = 0
SHAPE = (256, 256)
for fname in os.listdir('data/images'):
    imsave('data/resized/' + fname,
           resize(imread('data/images/' + fname, as_grey=True), SHAPE))
    counter += 1
    if counter % 1000 == 0:
        print(counter)
