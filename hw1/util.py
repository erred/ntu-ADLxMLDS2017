import csv
import os
import itertools

def phone248char(inputDir):
    p2cp = os.path.join(inputDir, "48phone_char.map")
    p2cm = {}
    with open(p2cp, 'r') as p2cf:
        p2cr = csv.reader(p2cf, delimiter='\t')
        for row in p2cr:
            p2cm[row[0]] = row[2]
    return p2cm

def phone239char(inputDir):
    p2cm = phone248char(inputDir)
    c2cp = os.path.join(inputDir, "phones/48_39.map")
    c2cm = {}
    with open(c2cp,'r') as c2cf:
        c2cr = csv.reader(c2cf, delimiter='\t')
        for row in c2cr:
            c2cm[row[0]] = row[1]
    mapping = {}
    for key,item in c2cm.items():
        mapping[key] = p2cm[item]
    return mapping

def trimOutStr(string):
    trimmed = string.strip('L')
    return ''.join(i for i, _ in itertools.groupby(trimmed))

def trimOutList(lis):
    trimmed = trimOutStr(''.join(lis))
    return [x for x in trimmed]
