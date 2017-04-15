#!/usr/bin/python3

import os
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dir", metavar='dir', help="model dirs")
args = parser.parse_args()

while True:
    fig = plt.figure()
    result = []
    for sub in ['live', 'dead']:
        d = os.listdir(os.path.join(args.dir, sub))
        for i in d:
            with open(os.path.join(args.dir, sub, i, 'cell.lua')) as fd:
                performance = fd.readline().split()[1].strip()
                try:
                    performance = float(performance)
                    if performance < 0:
                        continue
                    result.append((int(i), performance))
                except:
                    pass
                    # print(i, 'error')
    result = sorted(result, key=lambda x: x[0])
    plt.plot([i[1] for i in result], '.')
    print(max([i[1] for i in result]))
    print(sorted(result, key=lambda x: x[1])[-1])

    timer = fig.canvas.new_timer(interval=60000)
    timer.add_callback(plt.close)
    timer.start()
    plt.show()
