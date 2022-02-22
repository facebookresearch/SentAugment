
# SentAugment

SentAugment is a data augmentation technique for semi-supervised learning in NLP. It uses state-of-the-art sentence embeddings to structure the information of a very large bank of sentences. The large-scale sentence embedding space is then used to retrieve in-domain unannotated sentences for any language understanding task such that semi-supervised learning techniques like self-training and knowledge-distillation can be leveraged. This means you do not need to assume the presence of unannotated sentences to use semi-supervised learning techniques. In our paper [Self-training Improves Pre-training for Natural Language Understanding](https://arxiv.org/abs/2010.02194), we show that SentAugment provides strong gains on multiple language understanding tasks when used in combination with self-training or knowledge distillation.

![Model](sentaugment_figure.png)

## Dependencies

*  [PyTorch](https://pytorch.org/)
*  [FAISS](https://github.com/facebookresearch/faiss)
*  [XLM](https://github.com/facebookresearch/XLM)

## I. The large-scale bank of sentences
Our approach is based on a large bank of CommonCrawl web sentences. We use SentAugment to filter domain-specific unannotated data for semi-supervised learning NLP methods. This data can be found [here](http://www.statmt.org/cc-english/) and can be recovered from CommonCrawl by the [ccnet](https://github.com/facebookresearch/CC_Net) repository. It consists of 5 billion sentences, each file containing 100M sentences. As an example, we are going to use 100M sentences from the first file:

```bash
mkdir data && cd data
wget http://www.statmt.org/cc-english/x01.cc.5b.tar.gz
```
Then untar files and put all sentences into a single file:
```bash
tar -xvf *.tar.gz
cat *.5b > keys.txt
```

Then, for fast indexing, create a memory map (mmap) of this text file:
```bash
python src/compress_text.py --input data/keys.txt &
```
We will use this data as the bank of sentences.

## II. The SentAugment sentence embedding space (SASE)
Our sentence encoder is based on the Transformer implementation of XLM. It obtains state-of-the-art performance on several STS benchmarks. To use it, first clone XLM:
```bash
git clone https://github.com/facebookresearch/XLM
```

Then, download the SentAugment sentence encoder (SASE), and its sentencepiece model:
```bash
cd data
wget https://dl.fbaipublicfiles.com/sentaugment/sase.pth
wget https://dl.fbaipublicfiles.com/sentaugment/sase.spm
```


Then to embed sentences, you can run for instance:
```bash
input=data/keys.txt  # input text file
output=data/keys.pt  # output pytorch file

# Encode sentence from $input file and save it to $output
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output
```

This will output a torch file containing sentence embeddings (dim=256).

## III. Retrieving nearest neighbor sentences from a query
Now that you have constructed a sentence embedding space by encoding many sentences from CommonCrawl, you can leverage that "bank of sentences" with similarity search.
From an input query sentence, you can retrieve nearest neighbors from the bank by running:

```bash
bank=data/keys.txt  # compressed text file (bank)
emb=data/keys.pt  # embeddings of sentences (keys)
K=10000  # number of sentences to retrieve per query

## encode input sentences as sase embedding
input=sentence.txt  # input file containing a few (query) sentences
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $input.pt

## use embedding to retrieve nearest neighbors
input=sentence.txt  # input file containing a few (query) sentences
python src/flat_retrieve.py --input $input.pt --bank $bank --emb data/keys.pt --K $K > nn.txt &
```

Sentences in nn.txt can be used for semi-supervised learning as unannotated in-domain data. They also provide good paraphrases (use the cosine similarity score to filter good paraphrase pairs).

In the next part, we provide fast nearest-neighbor indexes for faster retrieval of similar sentences.

## IV. Fast K-nearest neighbor search
Fast K-nearest neighbor search is particularly important when considering a large bank of sentences. We use [FAISS](https://github.com/facebookresearch/faiss) indexes to optimize the memory usage and query time.

### IV.1 - The KNN index bestiary
For fast nearest-neighbor search, we provide pretrained [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/The-index-factory) (see Table below). Each index enables fast NN search based on different compression schemes. The embeddings are compressed using for instance scalar quantization (SQ4 or SQ8), PCA reduction (PCAR: 14, 40, 256), and search is sped up with k-means clustering (32k or 262k). Please consider looking at the [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU) for more information on indexes and  [how to train them](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

FAISS index | \#Sentences | \#Clusters | Quantization | #PCAR | Machine | Size
|:---: |:---: |:---: | :---: |:---: | :---: | :------: |
[`100M_1GPU_16GB`](https://dl.fbaipublicfiles.com/sentaugment/100M_1GPU_16GB.faiss.idx) | 100M | 32768 | SQ4 | 256 | 1GPU16 | 14GiB
[`100M_1GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/100M_1GPU_32GB.faiss.idx) | 100M | 32768 | SQ8 | 256 | 1GPU32 | 26GiB
[`1B_1GPU_16GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_1GPU_16GB.faiss.idx) | 1B | 262144 | SQ4 | 14 | 1GPU16 | 15GiB
[`1B_1GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_1GPU_32GB.faiss.idx) | 1B | 262144 | SQ4 | 40 | 1GPU32 | 28GiB
[`1B_8GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_8GPU_32GB.faiss.idx) | 1B | 262144 | SQ4 | 256 | 8GPU32 | 136GiB

We provide indexes that fit either on 1 GPU with 16GiB memory (1GPU16) up to a larger index that fits on 1 GPU with 32 GiB memory (1GPU32) and one that fits on 8 GPUs (32GB). Indexes that use 100M sentences are built from the first file "x01.cc.5b.tar.gz", and 1B indexes use the first ten files. All indexes are based on SASE embeddings.

  ### IV.2 - How to use an index to query nearest neighbors
You can get K nearest neighbors for each sentence of an input text file by running:

```bash
## encode input sentences as sase embedding
input=sentence.txt  # input file containing a few (query) sentences
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $input.pt

index=data/100M_1GPU_16GB.faiss.idx  # FAISS index path
input=sentences.pt  # embeddings of input sentences
bank=data/keys.txt  # text file with all the data (the compressed file keys.ref.bin64 should also be present in the same folder)
K=10  # number of sentences to retrieve per query
NPROBE=1024 # number of probes for querying the index

python src/faiss_retrieve.py --input $input --bank $bank --index $index --K $K --nprobe $NPROBE --gpu "True" > nn.txt &
```
This can also be used for paraphrase mining.


## Reference
If you found the resources here useful, please consider citing our paper:

```
@article{du2020self,
  title={Self-training Improves Pre-training for Natural Language Understanding},
  author={Du, Jingfei and Grave, Edouard and Gunel, Beliz and Chaudhary, Vishrav and Celebi, Onur and Auli, Michael and Stoyanov, Ves and Conneau, Alexis},
  journal={arXiv preprint arXiv:2010.02194},
  year={2020}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
The majority of SentAugment is licensed under CC-BY-NC. However, license information for PyTorch code is available at https://github.com/pytorch/pytorch/blob/master/LICENSE
