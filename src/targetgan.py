import os
import numpy as np
import pandas as pd
import argparse
from utils import *
import tensorflow as tf

class TargetGAN():
    def __init__(self,args):
        super(TargetGAN, self).__init__()
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.target = args.target
        self.step_size = args.step_size
        self.iterations = args.iterations
        self.g = tf.keras.models.load_model(args.generator)
        self.p = tf.keras.models.load_model(args.predictor)
        self.optimizer = Lion(learning_rate=args.step_size)
        self.log_dir = args.log_dir
    
    def opti_p(self, x, target):
        with tf.GradientTape() as t:
            gen_output = self.g(x)
            preds = self.p(gen_output)
            pred_onehot = self.p(np.eye(4)[np.argmax(gen_output, -1)])
            if target=="max":
                cost = tf.reduce_mean(-preds)
            elif target=="min":
                cost = tf.reduce_mean(preds)
            else: 
                tar = eval(target)
                mean, var = tf.nn.moments(preds, axes=[0])
                cost = 0.5 * (mean - tf.cast(tar, tf.float32)) ** 2 + 0.5 * (var - 0.0) ** 2
        grad_cost_latent = t.gradient(cost, x)
        self.optimizer.apply_gradients([(grad_cost_latent, x)])
        return gen_output, preds, pred_onehot, x
    
    def train(self):
        stamp = 'target_%s_seed_%s'%(self.target,self.seed)
        logdir = os.path.join(self.log_dir, stamp)
        check_folder(logdir)
        dist = []
        dist_preonehot = []
        iters = []
        means_onehot = []
        z= tf.Variable(tf.random.normal(shape=(self.batch_size, self.latent_dim)))
        for ctr in range(self.iterations):
            true_ctr = ctr  + 1
            if true_ctr==self.iterations :
                np.savetxt(os.path.join(logdir,"last_z.txt"),z,delimiter=",")
            gen_outputs, preds, pred_onehot, z = self.opti_p(z,self.target)
            preds = np.squeeze(preds)
            pred_onehot = np.squeeze(pred_onehot)
            mean_pred = np.mean(preds)
            mean_pred_onehot = np.mean(pred_onehot)
            best_idx = np.argmax(preds, 0)
            min_idx = np.argmin(preds, 0) 
            best_idx_onehot = np.argmax(pred_onehot, 0)
            min_idx_onehot = np.argmin(pred_onehot, 0)            
            best_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx], -1))
            best_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx_onehot], -1))
            min_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[min_idx], -1))
            min_seq_onehot = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[min_idx_onehot], -1))
            dist.append(preds)
            dist_preonehot.append(pred_onehot)
            iters.append(true_ctr)
            means_onehot.append(mean_pred_onehot)
            
            if true_ctr%200 ==0:
                print("\nIter {}\nBatch mean DNA score: {:.6f};\nBatch mean DNA score (one-hot predictor input): {:.6f};".format(true_ctr, mean_pred, mean_pred_onehot))
                print("Min DNA score: {:.6f}\n Min score Seq: {}".format(preds[min_idx], min_seq))
                print("Best DNA score Seq : {:.6f}\n Best Seq: {}".format(preds[best_idx], best_seq))
                print("Min DNA onehot score: {:.6f}\n Min onehot score Seq: {}".format(pred_onehot[min_idx_onehot], min_seq_onehot))
                print("Best DNA onehot score Seq : {:.6f}\n Best onehot Seq: {}\n".format(pred_onehot[best_idx_onehot], best_seq_onehot))
                if true_ctr==self.iterations:
                    plt.cla()
                    #plt.ylim([0., 1.])
                    plt.xlabel("iteration")
                    plt.ylabel("the predicted enrichment of the generated sequences")
                    dist_x = np.reshape([[c] * self.batch_size for c in np.linspace(0, true_ctr, len(dist))], [-1])
                    plt.scatter(dist_x, np.reshape(dist_preonehot,[-1]), color='C0', s=0.5, alpha=0.01)
                    plt.plot(np.linspace(0, true_ctr, len(means_onehot)), means_onehot, color='C1', ls='--', label='Mean score of one-hot re-encoded seqs')
                    plt.title('target = %s'%(self.target))
                    ax = plt.gca()
                    handles, labels = ax.get_legend_handles_labels()
                    
                    # sort both labels and handles by labels
                    def key(label):
                        if "one-hot" in label:
                            return 0
                        elif "Mean" in label:
                            return 1
                        elif "max" in label:
                            return 2
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: key(t[0])))
                    if self.target=="max":
                        ax.legend(handles, labels, loc='lower right')
                    elif self.target=="min":
                        ax.legend(handles, labels, loc='upper right')
                    else:
                        ax.legend(handles, labels, )
                    name = "scores_opt"
                    plt.savefig(os.path.join(logdir, name + ".png"), dpi=200)
                    plt.close()
                    iter = np.reshape([[c] * self.batch_size for c in np.linspace(1, true_ctr, len(dist),dtype=int)], [-1])
                    score = np.reshape(dist,[-1])
                    score_onehot = np.reshape(dist_preonehot,[-1])
                    optimization = ['%s'%self.target for i in range(len(score))]
                    all_score = pd.DataFrame({'iter':iter,'score':score,'optimization':optimization})
                    all_score_onehot = pd.DataFrame({'iter':iter,'score onehot':score_onehot,'optimization':optimization})
                    all_score.to_csv(os.path.join(logdir, name + ".csv"))
                    all_score_onehot.to_csv(os.path.join(logdir, name + "onehot.csv"))
                    last_seq = []
                    for i in gen_outputs:
                        seq = "".join(rev_charmap[n] for n in np.argmax(i, -1))
                        last_seq.append(seq)
                    with open(os.path.join(logdir,'last_seq.txt'),'w') as f:
                        f.write("\n".join(str(seq) for seq in last_seq))
        print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training TargetGAN!')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--generator', type=str, default='../wgan-gp/z_dim_100_gen_dim_200_disc_dim_200/checkpoints/checkpoint_54000/generator.h5', help="Location of generator model")
    parser.add_argument('--predictor', type=str, default='../data/predictor.h5', help="Location of predictor model")
    parser.add_argument('--target', default="max", help="Optimization target. Can be either 'max', 'min', or a target score number given as a float")
    parser.add_argument('--step_size', type=float, default=1e-2, help='Step-size for optimization.')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations to run the optimization for')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent dim')
    parser.add_argument('--log_dir', type=str, default='../targetgan',help='Base log folder')
    parser.add_argument('--device', type=str, default='0', help='Which GPU to use, 0 or 1.')
    args = parser.parse_args()
    """checking arguments"""
    check_folder(args.log_dir)
    if args.seed is not None:
        np.random.seed(args.seed)  # set random seed for numpy
        tf.random.set_seed(args.seed) # set random seed for tensorflow-cpu

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    model = TargetGAN(args)
    model.train()