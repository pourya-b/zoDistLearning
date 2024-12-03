import os
import time
from copy import deepcopy
import torch
import numpy as np
from scipy.io import savemat
from MyOptimizer_async import MyOptimizer_async
from MyOptimizer_async_pro import MyOptimizer_async_pro
from gen_channel import gen_rect_channel


class DeepNet_dist_myopt_async:
    def __init__(self, args, ra_scheme, problem, model_list):

        self.args = args
        self.problem = problem
        self.model_list = deepcopy(model_list)
        self.ra_scheme = ra_scheme

        self.channel_generator = gen_rect_channel(args, L=args.user_num, l=args.user_num/4, ch_cor=args.ch_coherence, ch_var=1, path_loss_fact=2.2)

    def train(self):
        print("\nTraining device: ", self.args.device)
        print("Training ...")
        start_time = time.time()
        loss_log = torch.zeros((self.args.epochs, 1))
        running_loss = []

        if self.args.random_connection:
            idx = np.arange(self.args.user_num)
            if self.args.max_delay > self.args.user_num:
                self.args.max_delay = self.args.user_num  # the maximum delay is user_num! As the underlying graph is connected by assumption.
                Warning("The maximum delay cannot exceed the number of users (nodes) in a connected graph! max_delay=user_num is considered instead.")

            # graph matrix containing the delays between the users. Set max_delay = 0 in synchronous scenarios.
            if self.args.max_delay == 0:
                graph = torch.zeros(self.args.user_num, self.args.user_num, dtype=torch.int64)
            else:
                graph = torch.randint(0, self.args.max_delay, (self.args.user_num, self.args.user_num))  # delays due to connection via intermediate nodes in the graph
                graph += self.args.comm_delay  # delays due to communication delay between two nodes (even neighbors)
                graph[idx, idx] = torch.zeros(self.args.user_num, dtype=torch.int64)  # each user has a delay=0 with itself!
        else:
            graph = torch.zeros(self.args.user_num, self.args.user_num, dtype=torch.int64)  # ones for test
            # kernel_size = 6
            for k in np.arange(start=0, stop=self.args.user_num - self.args.kernel_size + 1, step=self.args.stride):
                graph[k:k+self.args.kernel_size, k:k+self.args.kernel_size] = torch.ones(self.args.kernel_size, self.args.kernel_size, dtype=torch.int64)

        if self.args.random_connection:
            self.myOpt = MyOptimizer_async()
            self.myOpt.config(graph=graph, problem=self.problem, ra_scheme=self.ra_scheme, args=self.args, model_list=self.model_list, epsilon=self.args.epsilon, lr=self.args.learning_rate, lr_decay_factor=self.args.lr_decay_factor)
        else:
            self.myOpt = MyOptimizer_async_pro()
            self.myOpt.config(graph=graph, problem=self.problem, ra_scheme=self.ra_scheme, args=self.args, model_list=self.model_list, epsilon=self.args.epsilon, lr=self.args.learning_rate, lr_decay_factor=self.args.lr_decay_factor)

        for epoch in range(self.args.epochs):

            samples = self.channel_generator.gen_channel(2*self.args.batch_size, self.args.user_num, ch_type="grid", new_path_loss=False)
            samples = torch.abs(samples)

            _, loss = self.myOpt.backward(samples)

            running_loss.append(loss)
            running_loss_mean = float(sum(running_loss) / len(running_loss))
            loss_log[epoch] = running_loss_mean
            print(
                f" {epoch + 1}/{self.args.epochs}   "
                f"Loss (Train): {running_loss_mean:.4f}    ",
                flush=True
            )

        path = self.args.path + 'mat_logs/'
        if not(os.path.exists(path)):
            os.makedirs(path)
        mdic_loss = {'loss': loss_log.detach().numpy()}
        savemat(path + self.args.show_str + '_opt_' + str(self.args.run_index) + '.mat', mdic_loss)

        # path = self.args.path + 'models/'
        # if not(os.path.exists(path)):
        #     os.makedirs(path)
        # torch.save(self.model, path + "model_run_" + str(self.args.run_index))

        end_time = time.time()
        # for k in range(self.args.user_num):
        #     torch.save(self.model_list[k], self.args.path + "model_" + str(k+1) + "_run_" + str(self.args.run_index))

        print(f" Training time: {end_time - start_time:.4f}")
