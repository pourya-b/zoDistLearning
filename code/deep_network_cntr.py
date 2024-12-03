import torch
import os
import time
from copy import deepcopy
from MyOptimizer import MyOptimizer
from scipy.io import savemat
from gen_channel import gen_rect_channel

class DeepNet_cntr:
    def __init__(self, args, ra_scheme, problem, model):

        self.args = args
        self.problem = problem
        self.myOpt = MyOptimizer()
        self.model = deepcopy(model)
        self.ra_scheme = ra_scheme

        self.channel_generator = gen_rect_channel(args, L=args.user_num, l=args.user_num/4, ch_cor=args.ch_coherence, ch_var=1, path_loss_fact=2.2)

    def train(self):
        print("\nTraining device: ", self.args.device)
        print("Training ...")
        start_time = time.time()
        loss_log = torch.zeros((self.args.epochs, 1))
        running_loss = []

        params = self.model.parameters()
        self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            samples = self.channel_generator.gen_channel(self.args.batch_size, self.args.user_num, ch_type="grid", new_path_loss=False)
            samples = torch.abs(samples)

            loss_mean = 0
            for ll in range(self.args.batch_size):  # it is the only way to address batchsize as states my be some quantities related to the output of the model, e.g., data rate.
                states = torch.ones(self.args.user_num)
                self.channel = samples[ll].unsqueeze(0)
                net_input = self.ra_scheme.brew_input(self.channel, states)  # net_input dim: (batch*user_num, model inputDim)

                net_output = self.model(net_input).flatten()

                loss, user_rates = self.problem.loss_fun(net_output.unsqueeze(0), self.channel)
                self.ra_scheme.update(net_output.detach(), user_rates.detach(), self.channel)
                loss_mean += loss/self.args.batch_size

            loss_mean.backward(retain_graph=False)
            self.optimizer.step()

            running_loss.append(loss_mean.detach())
            running_loss_mean = float(sum(running_loss) / len(running_loss))
            loss_log[epoch] = loss_mean.detach()
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

        path = self.args.path + 'models/'
        if not(os.path.exists(path)):
            os.makedirs(path)
        torch.save(self.model, path + "model_run_" + str(self.args.run_index))

        end_time = time.time()
        # for k in range(self.args.user_num):
        #     torch.save(self.model_list[k], self.args.path + "model_" + str(k+1) + "_run_" + str(self.args.run_index))

        print(f" Training time: {end_time - start_time:.4f}")
