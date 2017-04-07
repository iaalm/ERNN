#!/bin/bash

cd coco-caption
flock lock python2 myeval.py $2 $1 > /dev/null 2>&1 
cd ../
