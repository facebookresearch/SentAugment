#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
Script for retrieving nearest neighbors of sentences from the bank using a given faiss index
Example: python src/faiss_retrieve.py --input $input --bank $bank --index $index --K $K
"""

import argparse
import faiss
import os
import sys
import time
import torch

from indexing import IndexLoad, IndexSearchKNN, IndexTextOpen 

parser = argparse.ArgumentParser(description="retrieve nearest neighbors of sentences")
parser.add_argument("--input", type=str, required=True , help="input pytorch embeddings")
parser.add_argument("--bank", type=str, required=True, help="compressed text file")
parser.add_argument("--index", type=str, required=True, help="faiss index")
parser.add_argument("--K", type=int, default=100, help="number of nearest neighbors per sentence")
parser.add_argument("--nprobe", type=int, default=1024, help="number of probes for the FAISS index")
parser.add_argument("--gpu", type=str, default="True", help="use gpu")

args = parser.parse_args()
assert args.gpu in ["True", "False"]
args.gpu = eval(args.gpu)

# load query embeddings
query_emb = torch.load(args.input).numpy()

# normalize embeddings
faiss.normalize_L2(query_emb)

# load the index
index = IndexLoad(args.index, args.nprobe, args.gpu)

# query the index and print retrieved neighbors
txt_mmap, ref_mmap = IndexTextOpen(args.bank)
nns = IndexSearchKNN(index, query_emb, txt_mmap, ref_mmap, args.K)
for nn in nns:
    print(nn)


