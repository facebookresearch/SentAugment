#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#

"""
Script that takes text as input and output SASE sentence embeddings
Example: python src/sase.py --input $input --model $modelpath --spm_model $spmmodel --batch_size 64 --cuda "True" --output $output
"""

import os
import sys
import torch
import argparse
from collections import OrderedDict
import sentencepiece as spm

sys.path.insert(0, 'XLM/')

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

parser = argparse.ArgumentParser(description="SASE encoding")


def main():
    parser.add_argument("--input", type=str, default="", help="input file")
    parser.add_argument("--model", type=str, default="", help="model path")
    parser.add_argument("--spm_model", type=str, default="", help="spm model path")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--max_words", type=int, default=100, help="max words")
    parser.add_argument("--cuda", type=str, default="True", help="use cuda")
    parser.add_argument("--output", type=str, default="", help="output file")
    args = parser.parse_args()

    # Reload a pretrained model
    reloaded = torch.load(args.model)
    params = AttrDict(reloaded['params'])

    # Reload the SPM model
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(args.spm_model)

    # cuda
    assert args.cuda in ["True", "False"]
    args.cuda = eval(args.cuda)

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)


    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    reloaded['model'] = OrderedDict({key.replace('module.', ''):reloaded['model'][key] for key in reloaded['model']})
    model.load_state_dict(reloaded['model'])
    model.eval()

    if args.cuda:
        model.cuda()

    # load sentences
    sentences = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            line = spm_model.EncodeAsPieces(line.rstrip())
            line = line[:args.max_words - 1]
            sentences.append(line)
            if i % 10_000 == 0:
                print(f"loading sentences line {i + 1}...")

    # encode sentences
    embs = []
    for i in range(0, len(sentences), args.batch_size):
        if i % 100 == 0:
            print(f"encoding sentences batch {i + 1}...")

        batch = sentences[i:i+args.batch_size]
        lengths = torch.LongTensor([len(s) + 1 for s in batch])
        bs, slen = len(batch), lengths.max().item()
        assert slen <= args.max_words

        x = torch.LongTensor(slen, bs).fill_(params.pad_index)
        for k in range(bs):
            sent = torch.LongTensor([params.eos_index] + [dico.index(w) for w in batch[k]])
            x[:len(sent), k] = sent

        if args.cuda:
            x = x.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            embedding = model('fwd', x=x, lengths=lengths, langs=None, causal=False).contiguous()[0].cpu()

        embs.append(embedding)

    # save embeddings
    torch.save(torch.cat(embs, dim=0).squeeze(0), args.output)


if __name__ == "__main__":
    main()
