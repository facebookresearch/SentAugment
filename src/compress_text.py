#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
import sys
import os
import argparse

import numpy as np
import torch
DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(DIR + '/../src/lib')
from indexing import CompressText




def main():
    parser = argparse.ArgumentParser(description="Generating ref file to support fetching text from memmap")
    parser.add_argument("--input", type=str, help="input text file")
    args = parser.parse_args()
    CompressText(args.input)



if __name__ == "__main__":
    main()
