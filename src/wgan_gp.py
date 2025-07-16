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

def resblock(inputs, num_channels): 
    res = inputs 
    for _ in range(2):  
        res = layers.ReLU()(res) #shape=[5, num_channels, num_channels],
        res = layers.Conv1D(num_channels, kernel_size=5, strides=1, padding='same', kernel_initializer=keras.initializers.RandomUniform(  
                                minval=-math.sqrt(3.) * math.sqrt(4. / (5 * num_channels + 5 * num_channels)),
                                maxval=math.sqrt(3.) * math.sqrt(4. / (5 * num_channels + 5 * num_channels))),
                                bias_initializer='zeros')(res)
    return inputs + (0.3 * res)

def generator_model(latent_dim=50, num_channels=100,res_layers=5):
    inputs = layers.Input(shape=(latent_dim,))
    output_size = 170*num_channels
    x = layers.Dense(output_size,kernel_initializer=keras.initializers.RandomUniform( 
                    minval=-math.sqrt(3.) * math.sqrt(2. / (latent_dim + output_size)), 
                    maxval= math.sqrt(3.) * math.sqrt(2. / (latent_dim + output_size))),
                    bias_initializer='zeros')(inputs)
    x = layers.Reshape((170, num_channels))(x)
    for _ in range(res_layers):
        outputs = resblock(x, num_channels)
        x = outputs
    x = layers.Conv1D(4,kernel_size=1,strides=1, padding="same",kernel_initializer=keras.initializers.RandomUniform(
                            minval=-math.sqrt(3.) * math.sqrt(4. / (1 * num_channels + 1 * 4)),
                            maxval= math.sqrt(3.) * math.sqrt(4. / (1 * num_channels + 1 * 4))),
                            bias_initializer=keras.initializers.Constant(0.0),activation='softmax')(x)#shape=[vocab_size]

    model = keras.Model(inputs=inputs, outputs=x)  
    
    return model

def discriminator_model(res_layers=5, num_channels=100):
    inputs = layers.Input(shape=(170,4))
    x = layers.Conv1D(num_channels, kernel_size=1, strides=1, padding="same",kernel_initializer=keras.initializers.RandomUniform(
                            minval=-math.sqrt(3.) * math.sqrt(4. / (1 * num_channels + 1 *4)),
                            maxval= math.sqrt(3.) * math.sqrt(4. / (1 * num_channels + 1 * 4))),
                            bias_initializer=keras.initializers.Constant(0.0))(inputs)  
    for _ in range(res_layers):
        outputs = resblock(x, num_channels)
        x = outputs
    x = layers.Flatten()(x)
    # x = layers.Reshape((,170*num_channels))(x)
    input_size = 170*num_channels
    score = layers.Dense(1, kernel_initializer=keras.initializers.RandomUniform(
                            minval=-math.sqrt(3.) * math.sqrt(2. / (input_size + 1)), 
                            maxval= math.sqrt(3.) * math.sqrt(2. / (input_size + 1))),
                            bias_initializer = keras.initializers.Constant(0.0))(x)
    model = tf.keras.Model(inputs=inputs, outputs=score)
    
    return model

class WGAN_GP():
    def __init__(self,args):
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.gen_dim = args.gen_dim
        self.disc_dim = args.disc_dim
        self.gen_layers = args.gen_layers
        self.disc_layers = args.disc_layers
        self.learning_rate = args.learning_rate
        self.lmbda = args.lmbda
        self.train_iters = args.train_iters
        self.max_seq_len = args.max_seq_len
        self.checkpoint_iters = args.checkpoint_iters
        self.disc_iters = args.disc_iters
        self.data_loc = args.data_loc
        self.log_dir = args.wgan_gp_log_dir
        self.g = generator_model(latent_dim=self.latent_dim, num_channels=self.gen_dim,res_layers=self.gen_layers)
        self.d = discriminator_model(res_layers=self.disc_layers, num_channels=self.disc_dim)
        self.g_optimizer = Lion(learning_rate=self.learning_rate)
        self.d_optimizer = Lion(learning_rate=self.learning_rate)

    def gradient_penalty(self, f, real, fake):
        shape = [tf.shape(real)[0]] + [1] * (real.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        x = real + alpha * (fake - real)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp

    @tf.function
    def train_G(self):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batch_size, self.latent_dim))
            x_fake = self.g(z, training=True)
            gen_score = self.d(x_fake, training=True)
            mean_gen_score = tf.reduce_mean(gen_score)
            G_loss = - mean_gen_score
        G_grad = t.gradient(G_loss, self.g.trainable_variables)
        self.g_optimizer.apply_gradients(zip(G_grad, self.g.trainable_variables))
        return G_loss, mean_gen_score

    @tf.function
    def train_D(self, x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batch_size, self.latent_dim))
            x_fake = self.g(z, training=True)
            x_real_d_logit = self.d(x_real, training=True)
            x_fake_d_logit = self.d(x_fake, training=True)
            mean_real_score = tf.reduce_mean(x_real_d_logit)
            disc_diff = tf.reduce_mean(x_fake_d_logit) - mean_real_score
            gp = self.gradient_penalty(functools.partial(self.d, training=True), x_real, x_fake)
            D_loss = disc_diff+gp*self.lmbda
        D_grad = t.gradient(D_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(D_grad, self.d.trainable_variables))
        return D_loss, mean_real_score

    @tf.function
    def sample(self, z):
        return self.g(z, training=False)

    def train(self):
        stamp = "z_dim_%s_gen_dim_%s_disc_dim_%s"%(self.latent_dim, self.gen_dim, self.disc_dim)
        logdir = os.path.join(self.log_dir, stamp)
        check_folder(os.path.join(self.log_dir, stamp))
        check_folder(os.path.join(self.log_dir, stamp, "samples"))
        # load_dataset
        print("Loading seqs data!")
        print("================================================")  
        data = pd.read_excel(self.data_loc)
        train_data = data[data.dataset == 'training set'].sequence.values[:200]
        train_data = [one_hot_encode(i) for i in train_data]
        valid_data = data[data.dataset == 'validation set'].sequence.values[:200]
        valid_data = [one_hot_encode(i) for i in valid_data]
        train_label = data[data.dataset == 'training set'].enrichment.values[:200]
        valid_seqs = feed(valid_data, self.batch_size, reuse=False)
        train_seqs = balanced_batch(train_data, train_label, self.batch_size)
        print("Training GAN!")
        print("================================================")
        fixed_latents = []
        nSampleBatches = 10
        for nBaches in range (nSampleBatches):
            fixed_latents.append(np.random.normal(size=[self.batch_size, self.latent_dim]))
        train_cost = []
        gen_costs = []
        gen_scores = []
        real_scores = []
        gen_counts = []
        train_counts = []
        valid_cost = []
        valid_counts = []
        for idx in range(self.train_iters):
            t1 = time.time()
            true_count = idx + 1 
            train_counts.append(true_count)
            # train generator
            if idx > 0:
                gen_counts.append(true_count)
                gen_cost_itr,mean_gen_score_itr = self.train_G()
                gen_costs.append(gen_cost_itr)
                gen_scores.append(mean_gen_score_itr)

            # train discriminator "to optimality"
            for d in range(self.disc_iters):
                data = next(train_seqs)
                cost, mean_real_score_itr = self.train_D(data)
            train_cost.append(cost)
            real_scores.append(mean_real_score_itr)
            t2 = time.time() 
            if  true_count % 100 == 0:
                #validation
                cost_vals = []
                data = next(valid_seqs)
                while data is not None:
                    z = tf.random.normal(shape=(self.batch_size, self.latent_dim), seed=self.seed)
                    x_fake = self.g(z, training=False)
                    x_real_d_logit = self.d(data, training=False)
                    x_fake_d_logit = self.d(x_fake, training=False)
                    disc_diff = tf.reduce_mean(x_fake_d_logit) - tf.reduce_mean(x_real_d_logit)
                    cost_vals.append(disc_diff)
                    data = next(valid_seqs)
                valid_cost.append(np.mean(cost_vals))
                valid_counts.append(true_count)
                print("Iteration {}: train_disc_cost={:.5f}, valid_disc_diff={:.5f}".format(true_count, cost, np.mean(cost_vals))) 
                if true_count <= 5000:
                    self.checkpoint_iters = 100
                elif true_count <= 10000:
                    self.checkpoint_iters = 250
                elif true_count <= 25000:
                    self.checkpoint_iters = 500
                elif true_count <= 50000:
                    self.checkpoint_iters = 1000
                else:
                    self.checkpoint_iters = 2000

            if true_count % self.checkpoint_iters == 0:
                samples = []
                for nBaches in range (nSampleBatches):  
                    samples.append(self.sample(fixed_latents[nBaches]))    
                samples = np.concatenate(samples, axis=0)
                
                save_samples(logdir, samples, true_count)
                ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_count))
                check_folder(ckpt_dir)
                self.g.save(os.path.join(ckpt_dir, "generator.h5"))
                self.d.save(os.path.join(ckpt_dir, "discrimator.h5"))

                name = "valid_disc_diff"
                plot(valid_counts, valid_cost, logdir, name, xlabel="Iteration", ylabel="Discriminator Valid Cost (mean(gen_score) - mean(real_score))")
                np.savetxt(os.path.join(logdir, "{}".format(name + ".csv")), np.c_[valid_counts, valid_cost], delimiter=",", fmt='%.5f')

                name = "train_disc_cost"
                plot(train_counts, train_cost, logdir,name, xlabel="Iteration", ylabel="Discriminator Train Cost (disc_diff + grad_penalty)")
                np.savetxt(os.path.join(logdir, "{}".format(name + ".csv")), np.c_[train_counts,train_cost], delimiter=",", fmt='%.5f')

                name = "mean_real_score"
                plot(train_counts, real_scores, logdir,name, xlabel="Iteration", ylabel="Discriminator Mean Score Real Seqs")
                np.savetxt(os.path.join(logdir, "{}".format(name + ".csv")), np.c_[train_counts,real_scores], delimiter=",", fmt='%.5f')

                name = "gen_cost"
                plot(gen_counts, gen_costs, logdir,name, xlabel="Iteration", ylabel="Generator Train Cost (-mean(gen_score))")
                np.savetxt(os.path.join(logdir, "{}".format(name + ".csv")), np.c_[gen_counts, gen_costs], delimiter=",", fmt='%.5f')

                name = "mean_gen_score"
                plot(gen_counts, gen_scores, logdir,name, xlabel="Iteration", ylabel="Discriminator Mean Score Generated Seqs")
                np.savetxt(os.path.join(logdir, "{}".format(name + ".csv")), np.c_[gen_counts, gen_scores], delimiter=",", fmt='%.5f')
                print('Program running : %s ms' % ((t2 - t1)*1000))
        print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training WGAN-GP!')
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
    parser.add_argument('--data_loc', type=str, default='../data/Natural promoters.xlsx', help='Data location')
    parser.add_argument('--log_dir', type=str, default='../wgan-gp',help='Base log folder')
    parser.add_argument('--device', type=str, default='0', help='Which GPU to use, 0 or 1.')
    args = parser.parse_args()

    """checking arguments"""
    check_folder(args.data_loc)
    check_folder(args.log_dir)
    if args.seed is not None:
        np.random.seed(args.seed)  # set random seed for numpy
        tf.random.set_seed(args.seed) # set random seed for tensorflow-cpu
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    model = WGAN_GP(args)
    model.train()