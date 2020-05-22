# -*- coding:utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
import sys
import random
import torch
import linecache
import pickle
import time
from configparser import ConfigParser

config = ConfigParser()
config.read('conf', encoding='UTF-8')
GPU_DEVICE = int(config['DEFAULT'].get("GPU_DEVICE"))

class RedaData(Dataset):

    def __init__(self,dirname):
        self.userset = set()
        self.itemset = set()
        self.datalen = 0
        self.dataset = []
        self.user2item = {}
        self.user2candidate = {}
        self.max_len = 0
        self.min_len = 100

        fr = open(dirname + "/trainset.txt","r")
        for line in fr:
            lineArr = line.split(":")
            user = int(lineArr[0])
            his_items = set([int(item) for item in lineArr[1].split()])

            self.user2item[user] = his_items
            self.max_len = max(self.max_len,len(his_items)-2)
            self.min_len = min(self.min_len,len(his_items)-2)

            self.userset.add(user)
            for item in his_items:
                self.itemset.add(item)
                self.dataset.append([user,item])
        self.item_size = len(self.itemset)

        item_len = len(self.itemset)
        user_len = len(self.userset)

        self.datalen = len(self.dataset)

        # for user in self.user2item:
        #     # his_items = self.user2item[user]
        #     # candidate_items = list(self.itemset - his_items)

        #     # candidate_items = random.sample(self.itemset,20000)
        #     candidate_items = np.random.choice(self.item_size,20000,replace=False)
        #     self.user2candidate[user] = candidate_items

    def __getitem__(self, idx):
        start = time.time()

        user, item = self.dataset[idx]
        his_items = self.user2item[user]

        # end = time.time()
        # print("================================> get by ids", end-start)
        # start = time.time()

        # candidate_items = self.user2candidate[user]
        # if len(candidate_items) < 18000:
        #     candidate_items = random.sample(self.itemset,20000)
        #     self.user2candidate[user] = candidate_items

        common_his_items = list(his_items - set([item]))
        base_item = random.sample(common_his_items,1)
        common_his_items = list(set(common_his_items) - set(base_item))

        common_his_item = random.sample(common_his_items,1)

        # end = time.time()
        # print("================================> sample common samples", end-start)
        # start = time.time()

        pos_item = item

        neg_item = [0]
        while True:
            # neg_item = random.sample(candidate_items,1)
            # if neg_item[0] in his_items:
            #     candidate_items.remove(neg_item[0])
            #     continue
            # candidate_items.remove(neg_item[0])
            # break

            # start = time.time()
            # print("================================> sample negtive start")
            #neg_item = np.random.choice(self.item_size,1,replace=False)
            neg_item = np.random.choice(self.item_size,1)
            # print("================================> sample negtive end", time.time()-start)

            if neg_item[0] not in his_items:
                break

        # end = time.time()
        # print("================================> sample negtive samples", end-start)
        # start = time.time()

        sample = {
            'common_his_item': torch.LongTensor(common_his_item).cuda(GPU_DEVICE),
            'pos_item': torch.LongTensor([item]).cuda(GPU_DEVICE),
            'neg_item': torch.LongTensor(neg_item).cuda(GPU_DEVICE),
            "base_item": torch.LongTensor(base_item).cuda(GPU_DEVICE)}

        # print("================================> upload to GPU", time.time()-start)

        return sample

    def get_user_item_dim(self):
        return len(self.userset), len(self.itemset)

    def __len__(self):
        return self.datalen