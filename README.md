# TargetGAN

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![TensorFlow-GPU Version](https://img.shields.io/badge/tensorflow--gpu-2.5-orange.svg)](https://www.tensorflow.org/install/gpu)

## 1. Environment 🚀

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Install 

```bash
git clone https://github.com/xlxianglei/TargetGAN.git
cd TargetGAN
conda env create -f targetgan.yml
conda activate targetGAN
```

## 2. Training WGAN-GP ✒️

```bash
python main.py --work wgan-gp
```

## 3. Generate promoters ✏️
```bash
python main.py --work generate
```

## 4. Training targetgan ⚡
```bash
python main.py --work targetgan
```

Feel free to contact us if you have any questions or suggestions regarding the code and models.