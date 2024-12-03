import os
import time
from copy import deepcopy
from scipy.io import savemat
import torch
from MyOptimizer import MyOptimizer
from gen_channel import gen_rect_channel


class DeepNet_myopt:
    def __init__(self, args, ra_scheme, problem, model_list):

        self.args = args
        self.problem = problem
        self.myOpt = MyOptimizer()
        self.model_list = deepcopy(model_list)
        self.ra_scheme = ra_scheme

        self.channel_generator = gen_rect_channel(args, L=args.user_num, l=args.user_num/4, ch_cor=args.ch_coherence, ch_var=1, path_loss_fact=2.2)

    def define_scenario(self, scenario="dist"):
        if scenario == "cntr":  # keeping only one DNN in the model list
            model_list = []
            model_list.append(self.model_list[0])
            self.model_list = model_list

    def train(self):
        print("\nTraining device: ", self.args.device)
        print("Training ...")
        start_time = time.time()
        loss_log = torch.zeros((self.args.epochs, 1))
        running_loss = []

        self.myOpt.config(problem=self.problem, ra_scheme=self.ra_scheme, args=self.args, model_list=self.model_list, epsilon=self.args.epsilon, lr=self.args.learning_rate, lr_decay_factor=self.args.lr_decay_factor)
        for epoch in range(self.args.epochs):

            samples = self.channel_generator.gen_channel(2*self.args.batch_size, self.args.user_num, ch_type="grid", new_path_loss=False)
            samples = torch.abs(samples)

            _, loss = self.myOpt.backward(samples)

            running_loss.append(loss)
            running_loss_mean = float(sum(running_loss) / len(running_loss))
            loss_log[epoch] = loss
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
