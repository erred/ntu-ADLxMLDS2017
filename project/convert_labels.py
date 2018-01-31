import csv
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

testset = set()
validset = set()
trainset = set()

with open('data/train.txt') as f:
    re = csv.reader(f)
    for r in re:
        trainset.add(r[0])

with open('data/valid.txt') as f:
    re = csv.reader(f)
    for r in re:
        validset.add(r[0])

with open('data/test.txt') as f:
    re = csv.reader(f)
    for r in re:
        testset.add(r[0])

traindata = {}
validdata = {}
testdata = {}

with open('data/Data_Entry_2017_v2.csv') as f:
    re = csv.reader(f)
    next(re)
    for r in re:
        id = r[0]
        obs = []
        for observe in r[1].split('|'):
            ob = LABELS[observe]
            obs.append(ob)

        if id in trainset:
            traindata[id] = obs
        elif id in validset:
            validdata[id] = obs
        elif id in testset:
            testdata[id] = obs

with open('data/pickles/labels_train.pkl', 'wb') as f:
    pickle.dump(traindata, f)

with open('data/pickles/labels_valid.pkl', 'wb') as f:
    pickle.dump(validdata, f)

with open('data/pickles/labels_test.pkl', 'wb') as f:
    pickle.dump(testdata, f)
