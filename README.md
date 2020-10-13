
# SentAugment

SentAugment is a data augmentation technique for NLP that retrieves similar sentences from a large bank of sentences. It can be used in combination with self-training and knowledge-distillation, or for retrieving paraphrases. See [our paper](arxiv.org/abs/) for more information.
  

## Dependencies

*  [FAISS](https://github.com/facebookresearch/faiss)
*  [XLM](https://github.com/facebookresearch/XLM)  

## I. Downloading a large bank of sentences
Our approach is based on a large bank of web sentences which can be used as unannotated data for semi-supervised learning NLP methods. This data can be found [here](http://www.statmt.org/cc-english/) and can be recovered from CommonCrawl by the [ccnet](https://github.com/facebookresearch/CC_Net) repository. This data consists of 5 billion sentences, each file consists of 100M sentences; to download the first 100 million sentences, run:

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

  ## II. Getting sentence embeddings with our encoder
  Our sentence encoder is based on the Transformer implementation of XLM. First, clone XLM:
```bash
git clone https://github.com/facebookresearch/XLM
```

Then, download the SentAugment sentence encoder (SASE), and its sentencepiece model:
```bash
cd data
wget https://dl.fbaipublicfiles.com/sentaugment/sase.pth
wget https://dl.fbaipublicfiles.com/sentaugment/sase.spm
```


Then to embed sentences, you can run:
```bash
input=data/keys.txt  # input text file
output=data/keys.pt  # output pytorch file

# Encode sentence from $input file and save it to $output
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $output
```

This will output a torch file containing sentence embeddings (dim=256).

  ## III. Retrieving nearest neighbor sentences from a query
From an input sentence, you can retrieve nearest neighbors from the bank by running:

```bash
bank=data/keys.txt.ref.bin64  # compressed text file (bank)
emb=data/keys.pt  # embeddings of sentences (keys)
K=10000  # number of sentences to retrieve per query

## encode input sentences as sase embedding
input=sentence.txt  # input file containing a few (query) sentences
python src/sase.py --input $input --model data/sase.pth --spm_model data/sase.spm --batch_size 64 --cuda "True" --output $input.pt

## use embedding to retrieve nearest neighbors
input=sentence.txt  # input file containing a few (query) sentences
python src/flat_retrieve.py --input $input.pt --bank $bank --emb data/keys.pt --K $K > nn.txt &
```

Sentences in nn.txt can be used for semi-supervised learning as unannotated in-domain data.

  ## IV. Fast K-nearest neighbor search (paraphrase retrieval)
Fast K-nearest neighbor search is particularly important when considering a large bank of sentences. We use [FAISS](https://github.com/facebookresearch/faiss) indexes to optimize the memory usage and query time.

### IV.1 - The KNN index bestiary
For fast nearest-neighbor search, we provide pretrained [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/The-index-factory). Each index enables fast NN search with various compression schemes. The embeddings are compressed using scalar quantization (SQ4 or SQ8), PCA reduction (PCA: 14, 40, 256), and search is sped up with k-means clustering (32k or 262k). See [FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU) for more information on [how to train indexes](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

FAISS index | \#Sentences | \#Clusters | Quantization | #PCAR | Machine | Size
|:---: |:---: |:---: | :---: |:---: | :---: | :------: |
[`100M_1GPU_16GB`](https://dl.fbaipublicfiles.com/sentaugment/100M_1GPU_16GB.faiss.idx) | 100M | 32768 | SQ8 | 256 | 1GPU16 | 14GiB 
[`100M_1GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/100M_1GPU_32GB.faiss.idx) | 100M | 32768 | SQ4 | 256 | 1GPU32 | 27GiB 
[`1B_1GPU_16GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_1GPU_16GB.faiss.idx) | 1B | 262144 | SQ4 | 14 | 1GPU16 | 15GiB
[`1B_1GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_1GPU_32GB.faiss.idx) | 1B | 262144 | SQ4 | 40 | 1GPU32 | 29GiB
[`1B_8GPU_32GB`](https://dl.fbaipublicfiles.com/sentaugment/1B_8GPU_32GB.faiss.idx) | 1B | 262144 | SQ4 | 256 | 8GPU32 | 136GiB 

We provide indexes that fit either on 1 GPU with 16GiB memory (1GPU16) up to a larger index that fits on 8 GPUs with 32 GiB memory (8GPU32). Indexes that use 100M sentences are built from the first file "x01.cc.5b.tar.gz", and 1B indexes use the first ten files. All indexes are based on SASE embeddings.
  
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

## License

See the [LICENSE](LICENSE) file for more details.