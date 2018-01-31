import csv
import numpy as np
import pickle

LABELS = {
    'No Finding': 0,
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Consolidation': 3,
    'Edema': 4,
    'Effusion': 5,
    'Emphysema': 6,
    'Fibrosis': 7,
    'Hernia': 8,
    'Infiltration': 9,
    'Infiltrate': 9,
    'Mass': 10,
    'Nodule': 11,
    'Pleural_Thickening': 12,
    'Pneumonia': 13,
    'Pneumothorax': 14
}

PRIORS = np.array([[113.26, 94.50], [673.04, 494.74], [216.05, 218.86],
                   [495.50, 343.66], [239.74, 490.62]])

# reduce to X by X
TARGET_GRID = 8
CLASSES = 15
ORIGINAL = 1024
BLOCKSIZE = ORIGINAL // TARGET_GRID
BOXES = 5

data = {}
with open('data/BBox_List_2017.csv') as f:
    re = csv.reader(f)
    next(re)
    for r in re:
        id = r[0]
        if id not in data:
            data[id] = np.zeros([TARGET_GRID, TARGET_GRID, BOXES, 5 + CLASSES])
        obs = LABELS[r[1]] + 5

        x = float(r[2])
        x_block = int(x // BLOCKSIZE)
        x_offset = (x % BLOCKSIZE) / BLOCKSIZE

        y = float(r[3])
        y_block = int(y // BLOCKSIZE)
        y_offset = (y % BLOCKSIZE) / BLOCKSIZE

        # exp(pred_w) * prior = w
        w = float(r[4])
        w = w / PRIORS[:, 0]
        h = float(r[5])
        h = h / PRIORS[:, 1]

        # print(x, y)
        # print(x_block, y_block)
        # print(x_offset, y_offset)

        # conf
        data[id][x_block, y_block, :, 0] = 1
        # x y
        data[id][x_block, y_block, :, 1] = x_offset
        data[id][x_block, y_block, :, 2] = y_offset
        # w h
        data[id][x_block, y_block, :, 3] = w
        data[id][x_block, y_block, :, 4] = h
        # classes
        data[id][x_block, y_block, :, obs] = 1


with open('data/labels_boxed.pkl', 'wb') as f:
    pickle.dump(data, f)
