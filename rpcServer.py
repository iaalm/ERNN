#! /usr/bin/python3

import random
import argparse
from ruler import rpcFileSystemRuler

parser = argparse.ArgumentParser()
parser.add_argument("dir", metavar='dir', help="model dirs")
parser.add_argument("--port", default=8080, help="server port")
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(123)
    ruler = rpcFileSystemRuler(args.dir, 10, 2)
    ruler.listen(int(args.port))
