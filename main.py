import os
import pandas as pd
import functools
import argparse
from src.utils import *
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import math
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=str,   
                    help= "wgan-gp, Generate, targetgan")
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent space')
    parser.add_argument('--gen_dim', type=int, default=200, help='Generator dimension parameter')
    parser.add_argument('--disc_dim', type=int, default=200, help='Discriminator dimension parameter')
    parser.add_argument('--gen_layers', type=int, default=5, help='How many layers for generator')
    parser.add_argument('--disc_layers', type=int, default=5, help='How many layers for discriminator')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument('--lmbda', type=float, default=10., help='Lipschitz penalty hyperparameter')
    parser.add_argument('--train_iters', type=int, default=100000, help='Number of iterations to train GAN for')
    parser.add_argument('--max_seq_len', type=int, default=170, help="Maximum sequence length of data")
    parser.add_argument('--checkpoint_iters', type=int, default=100, help='Number of iterations before saving checkpoint')
    parser.add_argument('--disc_iters', type=int, default=5, help='Number of iterations to train discriminator for at each training step')
    parser.add_argument('--data_loc', type=str, default='./data/Natural promoters.xlsx', help='Data location')
    parser.add_argument('--wgan_gp_log_dir', type=str, default='./wgan-gp',help='Base log folder')

    parser.add_argument('--generate_batch_size', type=int, default=9, help='Generate batch size')
    parser.add_argument('--generate_num_seqs', type=int, default=76851, help='Number of sequences generated at once')
    parser.add_argument('--generated_seqs_save_path', type=str, default='./', help='Generate sequence save path')
    parser.add_argument('--generator', type=str, default="./wgan-gp/z_dim_100_gen_dim_200_disc_dim_200/checkpoints/checkpoint_54000/generator.h5",\
                                                                         help='The final WGAN-GP model save path')
    
    parser.add_argument('--targetgan_batch_size', type=int, default=2048, help='Targetgan batch size')
    parser.add_argument('--predictor', type=str, default='./data/predictor.h5', help="Location of predictor model")
    parser.add_argument('--target', default="max", help="Optimization target. Can be either 'max', 'min', or a target score number given as a float")
    parser.add_argument('--step_size', type=float, default=1e-2, help='Step-size for optimization.')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations to run the optimization for')
    parser.add_argument('--targetgan_log_dir', type=str, default='./targetgan',help='Base log folder')
    parser.add_argument('--device', type=str, default='0', help='Which GPU to use, 0 or 1.')


    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)  # set random seed for numpy
        tf.random.set_seed(args.seed) # set random seed for tensorflow-cpu
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    if args.work == "wgan-gp":
        from src.wgan_gp import *
        """checking arguments"""
        check_folder(args.data_loc)
        check_folder(args.wgan_gp_log_dir)

        model = WGAN_GP(args)
        model.train()

    elif args.work == "generate":
        generator = tf.keras.models.load_model(args.generator)
        nSampleBatches = math.ceil(args.generate_num_seqs / args.generate_batch_size)
        fixed_latents = []
        for nBaches in range(nSampleBatches):
            fixed_latents.append(np.random.normal(size=[args.generate_batch_size, 100]))
        samples = []
        for nBaches in range(nSampleBatches):  
            samples.append(generator(fixed_latents[nBaches]))
        samples = np.concatenate(samples, axis=0)
        if len(samples) > args.generate_num_seqs:
            samples = samples[:args.generate_num_seqs]
        save_samples(args.generated_seqs_save_path,samples,54000)
        print("%s sequences have been generated "%(samples.shape[0]))
        
    elif args.work == "targetgan":
        from src.targetgan import *
        """checking arguments"""
        check_folder(args.targetgan_log_dir)
        model = TargetGAN(args)
        model.train()