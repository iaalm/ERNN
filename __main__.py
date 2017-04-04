#! /usr/bin/python3

import os
import random
from ruler import simpleFileSystemRuler

if __name__ == '__main__':
    random.seed(123)
    ruler = simpleFileSystemRuler('workdir', 2, 2)
    while True:
        path = ruler.born()
        os.system('cd %s/code ; th train.lua' % path)
        ruler.fight(path)
