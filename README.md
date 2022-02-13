# Translatotron(without wavenet and auxilary decoder networks)

PyTorch implementation of [Direct speech-to-speech translation with a sequence-to-sequence model](https://arxiv.org/abs/1904.06037). 

This implementation was on a private Telugu-Hindi dataset which cannot be published but has been tested and verified.

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
2. CD into this repo: `cd tacotron2`
3. Initialize submodule: `git submodule init; git submodule update`
4. Modify the dataset module according to the dataset of your preference. The dataset used was noisy so corresponding filtering was applied to make it fit for training. 
5. Install [PyTorch]
6. Install [Apex]
7. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  

1. Download pretrained published [Tacotron 2](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view) model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`


## Acknowledgements
This implementation uses code from the following repos: [Tacotron 2](https://github.com/NVIDIA/tacotron2)

We are thankful to the Transalotron paper authors and Tacotron 2 paper authors.

[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
