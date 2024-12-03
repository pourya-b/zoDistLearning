import os
import time
from scipy.io import savemat
import torch
from MyOptimizer import MyOptimizer
from gen_channel import gen_rect_channel


class DeepNet_dist:
    def __init__(self, args, ra_scheme, problem, model_list):

        self.ra_scheme = ra_scheme
        self.model_list = model_list
        self.args = args
        self.problem = problem
        self.myOpt = MyOptimizer()

        self.channel_generator = gen_rect_channel(args, L=args.user_num, l=args.user_num/4, ch_cor=args.ch_coherence, ch_var=1, path_loss_fact=2.2)

    def train(self):
        print("\nTraining device: ", self.args.device)
        print("Training ...")
        start_time = time.time()
        loss_log = torch.zeros((self.args.epochs, 1))
        running_loss = []
        running_loss_equal = []

        params = []
        for i in range(self.args.user_num):
            params += list(self.model_list[i].parameters())

        if self.args.optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
        elif self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=self.args.learning_rate)

        for epoch in range(self.args.epochs):
            self.optimizer.zero_grad()
            samples = self.channel_generator.gen_channel(self.args.batch_size, self.args.user_num, ch_type="grid", new_path_loss=False)
            samples = torch.abs(samples)
            # ch_ave_test += samples # to calculate ch_thr in brew

            states = torch.ones(self.args.user_num)
            net_output = torch.zeros(self.args.user_num)
            user_rates_mean = 0
            loss_mean = 0
            for ll in range(self.args.batch_size):
                channel = samples[ll]
                if self.args.state_mode == "power":
                    states = net_output.detach()
                net_input = self.ra_scheme.brew_input(channel, states)

                for k in range(self.args.user_num):
                    net_output[k] = self.model_list[k](net_input[k].unsqueeze(0)).flatten()

                loss, user_rates = self.problem.loss_fun(net_output.unsqueeze(0), channel.unsqueeze(0))
                loss_mean += loss/self.args.batch_size
                user_rates_mean += user_rates/self.args.batch_size
                self.ra_scheme.update(net_output, user_rates, channel)

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

        # path = self.args.path + 'models/'
        # if not(os.path.exists(path)):
        #     os.makedirs(path)
        # torch.save(self.model, path + "model_run_" + str(self.args.run_index))

        end_time = time.time()
        # for k in range(self.args.user_num):
        #     torch.save(self.model_list[k], self.args.path + "model_" + str(k+1) + "_run_" + str(self.args.run_index))

        print(f" Training time: {end_time - start_time:.4f}")
