#! /usr/bin/python3

import os
from ruler import simpleFileSystemRuler

if __name__ == '__main__':
    ruler = simpleFileSystemRuler('workdir', 2, 2)
    while True:
        path = ruler.born()
        os.system('cd %s/code ; th train.lua')
        ruler.fight(path)
