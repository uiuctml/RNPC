#!/usr/bin/env bash

cd ../src

./train_reference.py -m "cbm" -b 256 -e 100 -s 42
./train_reference.py -m "dcr" -b 256 -e 100 -s 42