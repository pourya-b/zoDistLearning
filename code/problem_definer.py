import torch

class Problem:
    def __init__(self, args, r_mins):
        self.args = args
        self.r_mins = r_mins

        self.opt_rates_valid = []
        self.opt_rates_test = []

    def add_model(self, model):
        self.model = model

    def loss_fun(self, powers, channel, Lambda=50):
        powers = powers.unsqueeze(1)
        current_device = powers.device
        current_dtype = powers.dtype

        H = channel.reshape(channel.shape[0], 1, self.args.user_num, self.args.user_num).type(current_dtype)
        idx = torch.arange(0, self.args.user_num)  # index for diagonals

        # H: batchSize, tone_num, user_num, user_num
        # powers: batchSize, tone_num, user_num
        # weights: batchSize, user_num
        powers = torch.reshape(powers, (powers.shape[0], self.args.channel_num, self.args.user_num))

        # Algorithm for calculating user rates: start
        powers = powers[:, :, None, :]
        powers = powers.repeat(1, 1, self.args.user_num, 1)
        # powers: batchSize, tone_num, user_num, user_num

        H_H_net_out = H * H * powers  # batchSize, tone_num, user_num, user_num
        H_H_net_out_D = H_H_net_out[:, :, idx, idx]  # batchSize, tone_num, user_num

        J = torch.sum(H_H_net_out, dim=3) + self.args.noise_level
        J = J - H_H_net_out_D
        # J: batchSize, tone_num, user_num

        bit_loading = torch.log2(1 + H_H_net_out_D / J)
        # bit_loading: batchSize, tone_num, user_num
        # Algorithm for calculating user rates: end

        user_Rate = torch.sum(bit_loading, dim=1)  # rate of each user
        # user_Rate: batchSize, user_num
        user_power = torch.sum(powers[:, :, 0, :], dim=1)

        loss = -torch.sum(user_Rate, dim=1)  # + Lambda * torch.sum(torch.relu(r_mins - user_Rate), dim=1)
        loss = torch.mean(loss)

        # power_violence = torch.mean(torch.sum(torch.relu(user_power - self.args.p_max), dim=1))
        # rate_violance = torch.mean(torch.sum(torch.relu(r_mins - user_Rate), dim=1))
        return loss, user_Rate