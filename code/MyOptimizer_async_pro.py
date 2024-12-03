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


class MyOptimizer_async_pro:
    def __init__(self):
        self.iter = 0
        self.U_weight_buffer = []
        self.U_bias_buffer = []

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

        self.slope_buffer = -1*torch.ones((args.user_num, args.user_num, 2))  # first column is time, second is value
        self.slope_buffer_rx = -1*torch.ones((2, self.args.user_num, self.args.user_num, 2))  # a temporary buffer for receiving the new quantities from the neighbors (two channels for simulating communication delay, each user's buffer, for each user ID, for time and value)

        set_seed(args)

    def reset(self):
        self.iter = 0

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

        for k in range(model_num):  # generating modified DNNs with the ranodm vectors U
            up = np.random.rand(1)  # user k throughs a coin to decide if to update or not
            temp_coef = 0
            if up <= self.args.up_p:
                temp_coef = 1
            # else:
            #     print("exception")

            for l in range(self.layer_num):
                U_weight_list[l][k] *= temp_coef
                U_bias_list[l][k] *= temp_coef

                weight_temp = copy.deepcopy(self.model_list[k].fc[l].weight).detach()
                bias_temp = copy.deepcopy(self.model_list[k].fc[l].bias).detach()

                weight_temp += self.epsilon * U_weight_list[l][k]
                bias_temp += self.epsilon * U_bias_list[l][k]

                self.model_list_modefied[k].fc[l].weight = copy.deepcopy(torch.nn.Parameter(weight_temp))
                self.model_list_modefied[k].fc[l].bias = copy.deepcopy(torch.nn.Parameter(bias_temp))

        if len(self.U_weight_buffer) == self.args.max_buffer_size:  # avoid buffer overflow
            del self.U_weight_buffer[0]  # deleting the oldest entry
            del self.U_bias_buffer[0]

        self.U_weight_buffer.append(U_weight_list)  # each user saves its own perturbation in its second buffer
        self.U_bias_buffer.append(U_bias_list)

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
        for ll in range(batch_size):  # it is the only way to address batchsize as states may be some quantities related to the output of the model, e.g., data rate.
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
        slope = (mean_rate_modified - mean_rate)/self.epsilon  # note that the slope is a vector, each user reports its own slope to the others
        self.slope_buffer_rx[1] = self.slope_buffer_rx[0]  # simulating channel/communication delay of 1 time slot
        for i in range(self.args.user_num):  # user i transmits its values to their neighbors j
            self.slope_buffer[i, i, 0] = self.iter  # updating self buffer by the recently estimated value 'slope'
            self.slope_buffer[i, i, 1] = slope[i]
            self.slope_buffer_rx[1, i, i, 0] = self.iter
            self.slope_buffer_rx[1, i, i, 1] = slope[i]

            # self.slope_buffer_rx[0, i] = self.slope_buffer[i]  # simulates the communication delay of 0 timeslot!

            neighbor_idx = np.where(self.graph[i] == 1)[0]  # index of neighbors of user i according to the matrix graph
            for j in neighbor_idx:
                rx = np.random.rand(1)  # user j throughs a coin to decide if to receive or not
                tx = np.random.rand(1)  # user i throughs a coin to decide if to transmit or not
                if rx <= self.args.rx_p and tx <= self.args.tx_p:
                    for p in range(self.args.user_num):  # transmitting pth value in the buffer of user i to user j
                        if (self.slope_buffer[j, p, 0] < self.slope_buffer_rx[0, i, p, 0]) and (self.slope_buffer_rx[1, j, p, 0] < self.slope_buffer_rx[0, i, p, 0]):  # if the quantity in i is newer than the ones in the buffer (or than just received ones from possible other neighbors)
                            self.slope_buffer_rx[1, j, p, 0] = self.slope_buffer_rx[0, i, p, 0]
                            self.slope_buffer_rx[1, j, p, 1] = self.slope_buffer_rx[0, i, p, 1]
                # else:
                #     print("exception")

            self.slope_buffer_rx[0, i] = self.slope_buffer[i]  # simulates the communication delay of 1 timeslot!

        self.slope_buffer[self.slope_buffer_rx[1] != -1] = self.slope_buffer_rx[1, self.slope_buffer_rx[1] != -1]  # updating the buffer by the newly received values (merging two buffers and discarding self.slope_buffer_rx)(the buffer is the one with one timeslot time delay!)
        # self.slope_buffer[self.slope_buffer_rx[0] != -1] = self.slope_buffer_rx[0, self.slope_buffer_rx[0] != -1]  # the same, but with no communication delay

        lr = self.lr / ((self.iter+1) ** (self.lr_decay_factor))  # decaying lr
        # lr = self.lr

        # all the users update their parameters (zeroth-order SGD)
        # all the users do the update asynchronously
        for k in range(model_num):  # updating user kth model
            for l in range(self.layer_num):
                weight_temp = copy.deepcopy(self.model_list[k].fc[l].weight)
                bias_temp = copy.deepcopy(self.model_list[k].fc[l].bias)

                weight_grad_temp = 0
                bias_grad_temp = 0
                for p in range(model_num):  # user k sweeps its buffer
                    time = int(self.slope_buffer[k, p, 0])
                    slope = self.slope_buffer[k, p, 1]

                    delay_idx = len(self.U_weight_buffer)-1 - (self.iter - time)  # the index in the buffer corresponding to delay (eg delay=0 is the last (newest) entry in the buffer)
                    if delay_idx >= 0 and delay_idx < len(self.U_weight_buffer):  # at the beginning, the buffer does not have the entries for large delays
                        weight_grad_temp += slope * self.U_weight_buffer[delay_idx][l][k]
                        bias_grad_temp += slope * self.U_bias_buffer[delay_idx][l][k]

                self.model_list[k].fc[l].weight = copy.deepcopy(torch.nn.Parameter(weight_temp + lr * weight_grad_temp))  # gradient ascend
                self.model_list[k].fc[l].bias = copy.deepcopy(torch.nn.Parameter(bias_temp + lr * bias_grad_temp))

        self.iter += 1
        # loss_mean = loss
        # self.U_dic = {"U_w_pre": U_weight_list, "U_b_pre": U_bias_list}
        return self.model_list, loss_mean
