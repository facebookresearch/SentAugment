Create Conda environment.

`conda create -n sent_augment python=3.8`

`conda activate sent_augment`

Pytorch version for Unity cluster. Check right one [here](https://pytorch.org/).

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

Faiss download for CPU or GPU:

`conda install -c conda-forge faiss-cpu`

`conda install -c conda-forge faiss-gpu`

Download sentencepiece.

`conda install -c conda-forge sentencepiece`


Downloading CC data. First access CPU resource.

`srun --time 2:00:00 --mem 40G --pty /usr/bin/bash`

Run script to download data.

`sbatch unity_scripts/download_cc_data.sh`

Download XML package inside SentAugment workspace.

`git clone https://github.com/facebookresearch/XLM`