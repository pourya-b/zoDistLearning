from problem_definer import Problem
from deep_network_cntr import DeepNet_cntr
from deep_network_dist import DeepNet_dist
from deep_network_cntr_dist_myopt import DeepNet_myopt
from deep_network_dist_myopt_async import DeepNet_dist_myopt_async
from RA_methods import AGG_GNN
from deepModels import deepModel


class Trainer:
    def __init__(self, args):
        self.args = args

        ra_scheme = AGG_GNN(args)
        inputDim = ra_scheme.policyInputDim  # fixed based on the proposed algorithm (refer to the function 'ra_scheme')
        outputDim = ra_scheme.policyOutputDim  # output is 1 user power

        model_list = self.model_configer(args, inputDim, outputDim)

        self.problem = Problem(args, 0)
        self.net_dist = DeepNet_dist(args, ra_scheme, self.problem, model_list)
        self.net_myopt = DeepNet_myopt(args, ra_scheme, self.problem, model_list)
        self.net_dist_myopt_async = DeepNet_dist_myopt_async(args, ra_scheme, self.problem, model_list)
        self.net_cntr = DeepNet_cntr(args, ra_scheme, self.problem, model_list[0])

    def model_configer(self, args, inputDim, outputDim):
        model_list = []
        for _ in range(args.user_num):
            model_list.append(deepModel(inputDim, outputDim, args).to(args.device))

        return model_list

    def run(self, args):
        if args.scenario == "cntr":
            if args.oracle_order == "first":
                self.net_cntr.train()
            elif args.oracle_order == "zeroth":
                self.net_myopt.define_scenario(scenario=args.scenario)
                self.net_myopt.train()
        elif args.scenario == "dist":
            if args.oracle_order == "first":
                self.net_dist.train()
            elif args.oracle_order == "zeroth":
                if args.synchrony == "synchron":
                    self.net_myopt.train()
                elif args.synchrony == "asynchron":
                    self.net_dist_myopt_async.train()
