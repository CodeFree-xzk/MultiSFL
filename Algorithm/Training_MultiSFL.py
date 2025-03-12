import copy
import os
import random
from datetime import datetime
from typing import List, Any

import numpy as np
import torch.nn
import wandb

from loguru import logger
from torch import nn

from Algorithm.Training import Training
from models import Aggregation, LocalUpdate_MultiSFL
from models.SplitModel import Complete_Model
from utils.utils import getTrueLabels

BUDGET_THRESHOLD = 0.2

DECAY = 0.5
DELTA = 0
WEIGHT = 1

WIN = 10

SAMPLE_BUDGET = 0.01
DATASET_SIZE = 50000


@logger.catch
class MultiSFL(Training):
    def __init__(self, args, net_glob, dataset_train, dataset_test, dict_users, net_glob_client, net_glob_server):
        super().__init__(args, net_glob, dataset_train, dataset_test, dict_users)
        # GitSFL Setting
        self.traffic: float = 0
        self.trafficList: List[float] = []
        self.repoSize: int = int(args.num_users * args.frac)
        self.modelServer: List[torch.nn.Module] = [copy.deepcopy(net_glob_server) for _ in range(self.repoSize)]
        self.modelClient: List[torch.nn.Module] = [copy.deepcopy(net_glob_client) for _ in range(self.repoSize)]
        self.cumulative_label_distributions = [np.zeros(args.num_classes) for _ in range(self.repoSize)]
        self.cumulative_label_distribution_weight: List[float] = [0 for _ in range(self.repoSize)]
        self.true_labels: list[np.ndarray[Any, np.dtype[Any]]] = getTrueLabels(self)
        self.help_count: List[int] = [0 for _ in range(args.num_users)]
        self.weakAggWeight: float = WEIGHT

        self.grad_norm = [0 for _ in range(self.repoSize)]
        self.fed_grad_norm = [0 for _ in range(WIN + 2)]
        self.win = WIN
        self.helper_overhead = 1
        self.client_overhead = 1

        self.classify_count = [[[1] for _ in range(DATASET_SIZE)] for _ in range(self.repoSize)]

        self.net_glob_client: nn.Module = net_glob_client
        self.net_glob_server: nn.Module = net_glob_server

        self.dataByLabel = self.organizeDataByLabel()

        if args.model == "resnet":
            if args.dataset in ("cifar10", "cifar100"):
                self.feature_size = int(13_107_622 / 50)
                self.model_size = 614_170
        elif args.model == "vgg":
            if args.dataset in ("cifar10", "cifar100"):
                self.feature_size = 32768
                self.model_size = 1_058_288

    @logger.catch()
    def main(self):
        while (self.traffic / 1024 / 1024) < self.args.comm_limit:
            print("%" * 50)
            selected_users = np.random.choice(range(self.args.num_users), self.repoSize, replace=False)
            for modelIndex, client_index in enumerate(selected_users):
                self.cumulative_label_distribution_weight[modelIndex] = self.cumulative_label_distribution_weight[
                                                                            modelIndex] * DECAY + 1
                self.cumulative_label_distributions[modelIndex] = (
                        (self.cumulative_label_distributions[modelIndex] * DECAY +
                         self.true_labels[client_index]) / self.cumulative_label_distribution_weight[modelIndex])

                self.splitTrain(client_index, modelIndex)

            self.Agg()

            if self.args.DB:
                self.adjustBudget()

            self.net_glob = Complete_Model(self.net_glob_client, self.net_glob_server)
            self.test()
            self.log()

            for modelIndex in range(self.repoSize):
                self.weakAgg(modelIndex)

            self.round += 1

            if self.args.MR != 0:
                print(self.help_count)

        self.saveResult()

    def splitTrain(self, curClient: int, modelIdx: int):
        sampledData = None
        if self.args.MR != 0:
            helpers, provide_data = self.selectHelpers(curClient, modelIdx)
            sampledData = self.sampleData(helpers, provide_data, modelIdx)

        local = LocalUpdate_MultiSFL(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[curClient],
                                     helpers_idx=sampledData)
        mean_grad_norm = local.union_train(self.modelClient[modelIdx], self.modelServer[modelIdx],
                                           self.classify_count[modelIdx])
        self.traffic += self.model_size * 2
        self.traffic += (len(self.dict_users[curClient]) * self.feature_size * self.args.local_ep * 2)
        self.grad_norm[modelIdx] = mean_grad_norm

    def Agg(self):
        w_client = [copy.deepcopy(model_client.state_dict()) for model_client in self.modelClient]
        w_avg_client = Aggregation(w_client, [1 for _ in range(self.repoSize)])
        self.net_glob_client.load_state_dict(w_avg_client)

        w_server = [copy.deepcopy(model_server.state_dict()) for model_server in self.modelServer]
        w_avg_server = Aggregation(w_server, [1 for _ in range(self.repoSize)])
        self.net_glob_server.load_state_dict(w_avg_server)


    def weakAgg(self, modelIdx: int):
        cur_model_client = self.modelClient[modelIdx]
        w = [copy.deepcopy(self.net_glob_client.state_dict()), copy.deepcopy(cur_model_client.state_dict())]
        lens = [self.weakAggWeight, 10]
        w_avg_client = Aggregation(w, lens)
        cur_model_client.load_state_dict(w_avg_client)

        cur_model_server = self.modelServer[modelIdx]
        w = [copy.deepcopy(self.net_glob_server.state_dict()), copy.deepcopy(cur_model_server.state_dict())]
        w_avg_server = Aggregation(w, lens)
        cur_model_server.load_state_dict(w_avg_server)

    def sampleData(self, helpers: List[int], provideData: List[List[int]], modexIdx: int) -> List[int]:
        # randomSample
        if self.args.BS == 0:
            sampledData = []
            for i, helper in enumerate(helpers):
                for classIdx, num in enumerate(provideData[i]):
                    sampledData.extend(random.sample(self.dataByLabel[helper][classIdx], num))
            return sampledData

        # boundarySample
        sampledData = []
        for i, helper in enumerate(helpers):
            for classIdx, num in enumerate(provideData[i]):
                if num == 0:
                    continue
                lst = [(dataIdx, np.mean(self.classify_count[modexIdx][dataIdx]))
                       for dataIdx in self.dataByLabel[helper][classIdx]]
                lst.sort(key=lambda x: x[-1])
                img = [i[0] for i in lst]
                w = [i[1] + 1e-10 for i in lst]
                w.reverse()
                sampledData.extend(random.choices(img, w, k=num))
        return sampledData

    def selectHelpers(self, curClient: int, modelIdx: int):
        overall_requirement = max(self.args.num_classes, int(len(self.dict_users[curClient]) * SAMPLE_BUDGET))
        # overall_requirement = int(len(self.dict_users[curClient]) * SAMPLE_BUDGET)
        cumulative_label_distribution = self.cumulative_label_distributions[modelIdx]
        prior_of_classes = [max(np.mean(cumulative_label_distribution) - label, 0)
                            for label in cumulative_label_distribution]
        requirement_classes = [int(overall_requirement * (prior / sum(prior_of_classes))) for prior in prior_of_classes]
        required = requirement_classes[::]

        helpers = []
        provide_data = []
        candidate = list(range(self.args.num_users))
        candidate.pop(curClient)
        random.shuffle(candidate)
        for client in candidate:
            if sum(requirement_classes) == 0:
                break
            temp = []
            for classIdx, label in enumerate(self.true_labels[client]):
                temp.append(min(label, requirement_classes[classIdx]))
                requirement_classes[classIdx] -= min(label, requirement_classes[classIdx])
            if sum(temp) > 0:
                self.help_count[client] += 1
                helpers.append(client)
                provide_data.append(temp)

        self.traffic += len(helpers) * self.model_size * self.args.local_ep
        self.traffic += (overall_requirement * self.feature_size * self.args.local_ep)
        self.helper_overhead += overall_requirement
        self.client_overhead += len(self.dict_users[curClient])

        print("-----MODEL #{}-----".format(modelIdx))
        print("overall_requirement:\t", overall_requirement)
        print("current_train_data:\t", list(self.true_labels[curClient]))
        print("cumu_label_distri:\t", list(map(int, cumulative_label_distribution)))
        print("prior_of_classes:\t", list(map(int, prior_of_classes)))
        print("required_classes:\t", required)
        print("total_provide_data:\t", provide_data)
        return helpers, provide_data

    def detectCLP(self) -> (bool, float):
        self.fed_grad_norm.append(np.mean(self.grad_norm))
        OldNorm = max([np.mean(self.fed_grad_norm[-self.win - 1:-1]), 0.0000001])
        NewNorm = np.mean(self.fed_grad_norm[-self.win:])
        delta = (NewNorm - OldNorm) / OldNorm
        return delta > DELTA, delta

        # self.weakAggWeight[modelIdx] = 1 - delta

    def adjustBudget(self):
        global SAMPLE_BUDGET
        CLP, delta = self.detectCLP()
        if self.args.DB == 1:
            if CLP:
                if SAMPLE_BUDGET >= BUDGET_THRESHOLD:
                    SAMPLE_BUDGET += 0.01
                else:
                    SAMPLE_BUDGET = min(BUDGET_THRESHOLD, SAMPLE_BUDGET * 2)
                    # COMM_BUDGET = COMM_BUDGET * 2
            else:
                SAMPLE_BUDGET = max(0.01, SAMPLE_BUDGET / 2)
        elif self.args.DB == 2:
            if self.round != 0:
                if self.round > 10:
                    SAMPLE_BUDGET = max(0.05, SAMPLE_BUDGET * (1 + delta))
                else:
                    SAMPLE_BUDGET = max(0.01, SAMPLE_BUDGET * (1 + delta))

        if self.args.AW == 1:
            self.weakAggWeight += delta
        elif self.args.AW == 2:
            self.weakAggWeight = 1 + delta

    def organizeDataByLabel(self) -> list[list[list[int]]]:
        organized = []
        for client in range(self.args.num_users):
            res = [[] for _ in range(self.args.num_classes)]
            all_local_data = self.dict_users[client]
            for data in all_local_data:
                res[self.dataset_train[data][1]].append(data)
            organized.append(res)
        return organized

    def log(self):
        logger.info(
            "Round{}, acc:{:.2f}, max_avg:{:.2f}, max_std:{:.2f}, loss:{:.2f}"
            ", comm:{:.2f}MB, budget:{:.2f}, add:{:.2f}, weight:{:.2f}",
            self.round, self.acc, self.max_avg, self.max_std, self.loss,
            (self.traffic / 1024 / 1024), SAMPLE_BUDGET, (self.helper_overhead / self.client_overhead),
            self.weakAggWeight)
        if self.args.wandb:
            wandb.log({"round": self.round, 'acc': self.acc, 'max_avg': self.max_avg,
                       "max_std": self.max_std, "loss": self.loss,
                       "comm": (self.traffic / 1024 / 1024), "budget": SAMPLE_BUDGET,
                       "add": (self.helper_overhead / self.client_overhead), "weight": self.weakAggWeight})
        self.trafficList.append((self.traffic / 1024 / 1024))

    def saveResult(self):
        path = "./result/{}".format(self.args.data_beta)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = "{}_{}_{}_{}.txt".format(self.args.algorithm, self.args.model,
                                            self.args.dataset, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        logger.critical("TRAINING COMPLETED! START SAVING...")
        with open(os.path.join(path, filename), 'w') as file:
            for i in range(len(self.acc_list)):
                line = str(self.trafficList[i]) + '\t' + str(self.acc_list[i]) + '\n'
                file.write(line)
        logger.critical("SAVE COMPLETED!")
