import os
import json
import random
import numpy as np
import torch
import argparse
from argparse import ArgumentDefaultsHelpFormatter

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class Parameter_config:
    def __init__(self):
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="distributed learning", formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument("--show_str", default="distLearning", type=str, help="string for prints and saving names")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument("--run_index", default=0, type=int, help="Run Index")
        # communication network configuration
        parser.add_argument("--user_num", default=24, type=int, help="Number of users")
        parser.add_argument("--channel_num", default=1, type=int, help="Number of channels or tones")
        parser.add_argument("--noise_level", default=1, type=float, help="Noise level (Shannon)")
        parser.add_argument("--p_max", default=10, type=int, help="maximum power for each user, for each user")
        parser.add_argument("--channel_thr", default=0.01, type=int, help="a threshold to consider the significance of the channel and to define neighbors")
        parser.add_argument("--ch_coherence", default=0.3, type=int, help="the coherence factor for the fast fading part")
        parser.add_argument("--K_hop", default=5, type=int, help="the number of considered neighbor hops")
        parser.add_argument("--state_mode", default="ones", type=str, help="what is the state x in the aggregation sequence of GNN approach")
        # train-test setup
        parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning Rate for main training")
        parser.add_argument("--sample_num_test", default=200, type=int, help="Number of training samples for unsupervised training")
        parser.add_argument("--epochs", default=2000, type=int, help="Number of Epochs for main training")
        parser.add_argument("--batch_size", default=10, type=int, help="Batch size for rate constraints optimization/ test unconstrained")
        parser.add_argument("--optimizer", default="ADAM", type=str, help="optimizer in the centralized training")
        # DNN hyperparameters
        parser.add_argument("--dropout_factor", default=0.0, type=float, help="Dropout Factor, the smaller number, the less effective")
        parser.add_argument("--activation_fun", default="relu", type=str, help="Activation Function")
        # proposed optimizer parameters
        parser.add_argument("--scenario", default="dist", type=str, help="distributed or centralized RA")
        parser.add_argument("--synchrony", default="asynchron", type=str, help="if the users are synchron or not")
        parser.add_argument("--oracle_order", default="zeroth", type=str, help="oracle order for the optimization")
        parser.add_argument("--epsilon", default=2, type=float, help="Epsilon for gradient estimation in zeroth-order optimization")
        parser.add_argument("--max_delay", default=10, type=float, help="maximum number of intermediate nodes between two points of the graph")
        parser.add_argument("--max_buffer_size", default=40, type=float, help="maximum buffer size in each user")
        parser.add_argument("--comm_delay", default=1, type=float, help="communication delay between two direct points")
        parser.add_argument("--lr_decay_factor", default=0.25, type=float, help="the factor for decaying lr in SGD - lr = learning_rate/(iter ** lr_decay_factor)")
        parser.add_argument("--random_connection", default=False, type=bool, help="if the graph contains delays or only connectivity")
        parser.add_argument("--up_p", default=0.9, type=float, help="probability of parameter update in each user")
        parser.add_argument("--tx_p", default=0.9, type=float, help="probability of transmitting updates by each user")
        parser.add_argument("--rx_p", default=0.9, type=float, help="probability of receiving updates by each user")
        parser.add_argument("--kernel_size", default=6, type=int, help="size of kernel generating the connected graph")
        parser.add_argument("--stride", default=3, type=int, help="stride for generating the connected graph")       

        self.default_args = parser.parse_args()
        self.default_args.path = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(self.default_args.path + 'commandline_args.txt', 'w') as f:
            json.dump(self.default_args.__dict__, f, indent=2)

        self.default_args.network_hidden_config = np.array([30, 30])  # the default architecture of the DNN
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.default_args.n_gpu = 1
        self.default_args.device = "cpu"  # torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.default_args)

        print("\nProject: " + parser.description)
        print("Using Device: ", self.default_args.device)
        # print("GPU Count: ", torch.cuda.device_count())
        # print("Visible Device Names: ", os.environ["CUDA_VISIBLE_DEVICES"], flush=True)

        self.args = self.default_args
        return self.default_args