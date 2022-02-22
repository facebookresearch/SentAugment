#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#

# indexing and search with FAISS

import faiss
import os.path
import sys
import numpy as np
import torch


###############################################################################
# create an FAISS index on the given data

def IndexCreate(input_path, idx_type, output_path, normalize=True, dim=512):

    assert idx_type == 'FlatL2', 'only FlatL2 index is currently supported'
    x = torch.load(input_path).numpy()
    print(' - creating FAISS index')
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
    idx.add(x)
    print(' - saving index into ' + output_path)
    faiss.write_index(idx, output_path)
    return x, idx


def LoadTextSimple(text_fname):
    """
    Naive version of loading text into python list
    used for retrieve text using sentence idx from faiss
    NOTE: inefficient, will be replaced with mmap
    """
    with open(text_fname, 'r', encoding='utf-8', errors='ignore') as fin:
        sentences = [s.strip() for s in fin]
    return sentences


def CompressText(txt_fname):
    """
    generate ref binary file storing starting offset for each sentence
    """
    fname = txt_fname.replace('.txt', '.ref.bin64')
    offsets = [0]
    with open(txt_fname, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            offsets.append(offsets[-1] + len(bytes(line, encoding='utf-8', errors='ignore')))
    offsets = np.array(offsets[:-1], dtype=np.int64)  # discard last one
    offsets.tofile(fname)


###############################################################################
# Opens a text file with the sentences corresponding to the indices used
# by an FAISS index
# We also need the reference files with the byte offsets to the beginning
# of each sentence
# optionnally:  array with number of words per sentence
# All arrays are memory mapped

def IndexTextOpen(txt_fname):
    # print('Reading text corpus')
    # print(' - texts: {:s}'.format(txt_fname))
    txt_mmap = np.memmap(txt_fname, mode='r', dtype=np.uint8)
    fname = txt_fname.replace('.txt', '.ref.bin32')
    if os.path.isfile(fname):
        # print(' - sentence start offsets (32 bit): {}'.format(fname))
        ref_mmap = np.memmap(fname, mode='r', dtype=np.uint32)
    else:
        fname = txt_fname.replace('.txt', '.ref.bin64')
        if os.path.isfile(fname):
            # print(' - sentence start offsets (64 bit): {}'.format(fname))
            ref_mmap = np.memmap(fname, mode='r', dtype=np.uint64)
        else:
            # print('ERROR: no file with sentence start offsets found')
            sys.exit(1)
    # print(' - found {:d} sentences'.format(ref_mmap.shape[0]))
    return txt_mmap, ref_mmap


###############################################################################
# Return the text for the given index

def IndexTextQuery(txt_mmap, ref_mmap, idx):
    p = int(ref_mmap[idx])  # get starting byte position
    i = 0
    dim = 10000  # max sentence length in bytes
    b = bytearray(dim)
    #  find EOL
    while txt_mmap[p+i] != 10 and i < dim:
        b[i] = txt_mmap[p+i]
        i += 1
    return b[0:i].decode('utf-8')



###############################################################################
# Load an FAISS index

def IndexLoad(idx_path, nprobe=0, gpu=False):
    print('Reading FAISS index', file=sys.stderr)
    print(' - index: {:s}'.format(idx_path), file=sys.stderr)
    index = faiss.read_index(idx_path)
    print(' - found {:d} sentences of dim {:d}'.format(index.ntotal, index.d), file=sys.stderr)
    print(' - setting nbprobe to {:d}'.format(nprobe), file=sys.stderr)
    if gpu:
        print(' - transfer index to %d GPUs ' % faiss.get_num_gpus(), file=sys.stderr)
        index = faiss.index_cpu_to_all_gpus(index) # co=co
        faiss.GpuParameterSpace().set_index_parameter(index, 'nprobe', nprobe)
    return index


###############################################################################
# Search the [k] nearest vectors of [x] in the given index
# and return the text lines

def IndexSearchKNN(index, x, T, R, kmax=1, dedup=True):
    D, I = index.search(x, kmax)
    all_res = []
    for n in range(x.shape[0]):
        prev = set()  # for depuplication
        res = []
        for i in range(kmax):
            txt = IndexTextQuery(T, R, I[n, i])
            # txt = T[I[n, i]]
            if dedup and txt not in prev:
                prev.add(txt)
                res.append((txt, D[n, i]))
        all_res.append(res)
    return all_res
