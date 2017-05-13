#!/usr/bin/python3

import os
import argparse
import multiprocessing
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dir", metavar='dir', help="model dirs")
args = parser.parse_args()

result = []


def getResult(path):
    sub = path[0]
    i = path[1]
    with open(os.path.join(args.dir, sub, i, 'cell.lua')) as fd:
        performance = fd.readline().split()[1].strip()
        try:
            performance = float(performance)
            if performance < 0:
                return
            return ((int(i), performance))
        except:
            pass


dirs = [('live', i) for i in os.listdir(os.path.join(args.dir, 'live'))] + \
    [('dead', i) for i in os.listdir(os.path.join(args.dir, 'dead'))]

pool = multiprocessing.Pool(16)
result = [i for i in pool.imap_unordered(getResult, dirs) if i]
pool.close()
pool.join()
result = sorted(result, key=lambda x: x[0])
print(max([i[1] for i in result]))
print(sorted(result, key=lambda x: x[1])[-1])

plt.plot([i[1] for i in result], '.')
plt.show()

plt.plot([i[0] for i in result], [i[1] for i in result])
plt.show()
