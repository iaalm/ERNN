#! /usr/bin/python3

import os
import random
import argparse
from ruler import simpleFileSystemRuler

parser = argparse.ArgumentParser()
parser.add_argument("dir", metavar='dir', help="model dirs")
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(123)
    ruler = simpleFileSystemRuler(args.dir, 10, 2)
    while True:
        path = ruler.born()
        os.system('cd %s/code ; th train.lua -seed %d' %
                  (path, random.randint(1, 1000)))
        ruler.fight(path)
