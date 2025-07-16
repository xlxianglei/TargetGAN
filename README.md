# TargetGAN

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3.8/library/index.html)
[![TensorFlow-GPU Version](https://img.shields.io/badge/tensorflow--gpu-2.5-orange.svg)](https://www.tensorflow.org/install/gpu)

## 1. Install 🚀

```bash
git clone https://github.com/xlxianglei/TargetGAN.git
cd TargetGAN
conda env create -f targetgan.yml
conda activate targetGAN
```

## 2. Training WGAN-GP ✒️

```bash
python main.py --work wgan-gp --wgan_gp_log_dir {WGAN_GP_path}
```

## 3. Generate promoters ✏️
```bash
python main.py --work generate --generated_seqs_save_path {Generated_promoters_path}
```

## 4. Training targetgan ⚡
```bash
python main.py --work targetgan --targetgan_log_dir {TargetGAN_path}
```

Feel free to contact us if you have any questions or suggestions regarding the code and models.