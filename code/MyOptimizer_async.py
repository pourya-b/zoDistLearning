import copy
import random
import numpy as np
import torch

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class MyOptimizer_async:
    def __init__(self):
        self.iter = 1
        self.U_weight_buffer = []
        self.U_bias_buffer = []
        self.slope_buffer = []

    def config(self, graph, problem, ra_scheme, args, model_list, epsilon, lr, lr_decay_factor=0.25, polution_factor=0.0):

        self.model_list = model_list
        self.epsilon = epsilon
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.polution_factor = polution_factor  # not implemented in this version
        self.args = args
        self.problem = problem
        self.ra_scheme = ra_scheme
        self.graph = graph

        set_seed(args)

    def reset(self):
        self.iter = 1

    def backward(self, samples):
        agent_parameter = self.model_list[0]  # parameter of the only one model
        self.layer_num = len(agent_parameter.fc)  # we assume all the models have the same shape (structure), we picked the first one
        self.model_list_modefied = copy.deepcopy(self.model_list)
        U_weight_list = []  # U vector is the random distortion in the zeroth-order optimization by terminology
        U_bias_list = []
        model_num = len(self.model_list)

        for l in range(self.layer_num):  # generating random vectors U
            weight_shape = agent_parameter.fc[l].weight.shape  # the same shape (structure) by assumption
            bias_shape = agent_parameter.fc[l].bias.shape
            U_weight_list.append(torch.randn(model_num, weight_shape[0], weight_shape[1]))  # list dim: layer_num length, each with model_num by weight_shape
            U_bias_list.append(torch.randn(model_num, bias_shape[0]))  # list dim: layer_num length, each with model_num by bias_shape

        if len(self.U_weight_buffer) == self.args.user_num:  # avoid buffer overflow
            del self.U_weight_buffer[0]  # deleting the oldest entry
            del self.U_bias_buffer[0]

        self.U_weight_buffer.append(U_weight_list)
        self.U_bias_buffer.append(U_bias_list)

        # if self.iter == 1:  # initialization
        #     self.U_dic = {"U_w_pre": U_weight_buffer, "U_b_pre": U_bias_buffer}

        for k in range(model_num):  # generating modified DNNs with the ranodm vectors U
            for l in range(self.layer_num):
                weight_temp = copy.deepcopy(self.model_list[k].fc[l].weight).detach()
                bias_temp = copy.deepcopy(self.model_list[k].fc[l].bias).detach()

                weight_temp += self.epsilon * U_weight_list[l][k]
                bias_temp += self.epsilon * U_bias_list[l][k]

                self.model_list_modefied[k].fc[l].weight = copy.deepcopy(torch.nn.Parameter(weight_temp))
                self.model_list_modefied[k].fc[l].bias = copy.deepcopy(torch.nn.Parameter(bias_temp))

        batch_size = self.args.batch_size
        net_output = torch.zeros(1, self.args.user_num)

        user_rates_mean = 0
        loss_mean = 0
        for ll in range(batch_size):  # it is the only way to address batchsize as states my be some quantities related to the output of the model, e.g., data rate.
            states = torch.ones(self.args.user_num)
            self.channel = samples[ll].unsqueeze(0)
            net_input = self.ra_scheme.brew_input(self.channel, states)  # net_input dim: (batch*user_num, model inputDim)
            # net_input = net_input.reshape((batch_size, self.args.user_num, -1))  # net_input dim: (batch, user_num, model inputDim)

            if model_num == 1:
                net_output[0, :] = self.model_list[0](net_input).flatten()
            else:
                for k in range(model_num):
                    net_output[0, k] = self.model_list[k](net_input[k, :]).flatten()  # each user has its own model # net_output dim: (batch, user_num)
            net_output = net_output.detach()
            loss, user_rates = self.problem.loss_fun(net_output, self.channel)
            self.ra_scheme.update(net_output, user_rates, self.channel)
            loss_mean += loss/batch_size
            user_rates_mean += user_rates/batch_size

        net_output_modified = torch.zeros(1, self.args.user_num)

        user_rates_modified_mean = 0
        for ll in range(batch_size):  # it is the only way to address batchsize as states my be some quantities related to the output of the model, e.g., data rate.
            self.channel = samples[batch_size+ll].unsqueeze(0)
            net_input = self.ra_scheme.brew_input(self.channel, states)  # net_input dim: (batch*user_num, model inputDim)
            # net_input = net_input.reshape((batch_size, self.args.user_num, -1))  # net_input dim: (batch, user_num, model inputDim)

            if model_num == 1:
                net_output_modified[0, :] = self.model_list_modefied[0](net_input).flatten()
            else:
                for k in range(model_num):
                    net_output_modified[0, k] = self.model_list_modefied[k](net_input[k, :]).flatten()  # each user has its own model # net_output dim: (batch, user_num)
            net_output_modified = net_output_modified.detach()
            _, user_rates_modified = self.problem.loss_fun(net_output_modified, self.channel)
            self.ra_scheme.update(net_output_modified, user_rates_modified, self.channel)
            user_rates_modified_mean += user_rates_modified/batch_size

        # all the users calculate the (change in) their average data rate
        mean_rate = torch.mean(user_rates_mean, dim=0)
        mean_rate_modified = torch.mean(user_rates_modified_mean, dim=0)

        # all the users estimate the slope for their own dara rate
        self.slope = (mean_rate_modified - mean_rate)/self.epsilon  # note that the slope is a vector, each user reports its own slope to the others
        if len(self.slope_buffer) == self.args.user_num:  # avoid buffer overflow
            del self.slope_buffer[0]  # deleting the oldest entry
        self.slope_buffer.append(self.slope)

        lr = self.lr / (self.iter ** (self.lr_decay_factor))  # decaying lr
        # lr = self.lr

        # all the users update their parameters (zeroth-order SGD)
        # all the users do the update asynchronously
        for k in range(model_num):  # updating user kth model
            for l in range(self.layer_num):
                weight_temp = copy.deepcopy(self.model_list[k].fc[l].weight)
                bias_temp = copy.deepcopy(self.model_list[k].fc[l].bias)

                weight_grad_temp = 0
                bias_grad_temp = 0
                for kk in range(model_num):  # user k sums up all slopes from all the other users (with possible delays)
                    delay = self.graph[k, kk]  # the delay user k sees to get user kk's slope! if delay=0, it means k and kk are neighbors and have a direct link.
                    delay_idx = len(self.slope_buffer) - delay - 1  # the index in the buffer corresponding to delay (eg delay=0 is the last (newest) entry in the buffer)
                    if delay_idx >= 0:  # at the beginning, the buffer does not have the entries for large delays
                        weight_grad_temp += self.slope_buffer[delay_idx][kk] * self.U_weight_buffer[delay_idx][l][k]
                        bias_grad_temp += self.slope_buffer[delay_idx][kk] * self.U_bias_buffer[delay_idx][l][k]

                self.model_list[k].fc[l].weight = copy.deepcopy(torch.nn.Parameter(weight_temp + lr * weight_grad_temp))  # gradient ascend
                self.model_list[k].fc[l].bias = copy.deepcopy(torch.nn.Parameter(bias_temp + lr * bias_grad_temp))

        self.iter += 1
        loss_mean = loss
        # self.U_dic = {"U_w_pre": U_weight_list, "U_b_pre": U_bias_list}
        return self.model_list, loss_mean
