# -*- coding:utf-8 -*-

import numpy as np 
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser

from RedaData import RedaData
from RedaUtils import test_util

import sys
import os
import time

FType = torch.FloatTensor
LType = torch.LongTensor

config = ConfigParser()
config.read('conf', encoding='UTF-8')
GPU_DEVICE = int(config['DEFAULT'].get("GPU_DEVICE"))
start_epoch = int(config['DEFAULT'].get("start_epoch"))
inter_epoch = int(config['DEFAULT'].get("inter_epoch"))

print(GPU_DEVICE, start_epoch, inter_epoch)

class Reda(torch.nn.Module):

    def __init__(self): 
        super(Reda,self).__init__()

        self.init_basic_conf()

        self.opt = Adam(lr=self.learning_rate, params=[{"params":self.item_emb},{"params":self.memory_emb},\
            {"params":self.memory_keys},{"params":self.weight_net.parameters()},{"params":self.weight_factor}], weight_decay=1e-5)
        
        self.loss = torch.FloatTensor().cuda(GPU_DEVICE)
        self.max_measures = [0.0,0.0,0.0,0.0,0.0,0.0]

    def init_basic_conf(self):
        self.learning_rate = float(config['DEFAULT'].get("learning_rate"))
        self.batch_size = int(config['DEFAULT'].get("batch_size"))
        self.l2_reg = float(config['DEFAULT'].get("l2_reg"))
        self.epoch_num = int(config['DEFAULT'].get("epoch_num"))
        self.emb_size = int(config['DEFAULT'].get("emb_size"))
        self.memory_size = int(config['DEFAULT'].get("memory_size"))
        self.FM_K = int(config['DEFAULT'].get("FM_K"))
        self.weightAtt_size = int(config['DEFAULT'].get("weightAtt_size"))
        self.dirname = config['DEFAULT'].get("dirname")

        print(self.learning_rate, self.memory_size, self.emb_size, self.batch_size, self.FM_K, self.dirname)

        self.data = RedaData(self.dirname)
        self.num_users, self.num_items = self.data.get_user_item_dim()

        print(len(self.data), self.num_users, self.num_items, self.data.max_len, self.data.min_len)

        self.user_emb = Variable(torch.from_numpy(np.random.uniform(
            -0.1, 0.1, (self.num_users, self.emb_size)))\
            .type(FType).cuda(GPU_DEVICE), requires_grad=False)

        self.item_emb = Variable(torch.from_numpy(np.random.uniform(
            -0.1, 0.1, (self.num_items, self.FM_K, self.emb_size)))\
            .type(FType).cuda(GPU_DEVICE), requires_grad=True)

        self.memory_keys = Variable(torch.from_numpy(np.random.uniform(
            -0.1, 0.1, (self.emb_size, self.memory_size)))\
            .type(FType).cuda(GPU_DEVICE), requires_grad=True)

        self.memory_emb = Variable(torch.from_numpy(np.random.uniform(
            -0.1, 0.1, (self.memory_size, self.emb_size)))\
            .type(FType).cuda(GPU_DEVICE), requires_grad=True)

        self.weight_net = torch.nn.Sequential(
                    nn.Linear(self.emb_size,self.weightAtt_size),
                    nn.ReLU()).cuda(GPU_DEVICE)
        self.weight_factor = Variable(torch.from_numpy(np.random.uniform(
            -0.1, 0.1, (self.weightAtt_size, 1)))\
            .type(FType).cuda(GPU_DEVICE), requires_grad=True)

    def interact_layer(self,item_emb,common_his_item_emb):
        batch = item_emb.size()[0]
        emb_size = self.emb_size
        FM_K = self.FM_K
        interact_emb = torch.from_numpy(np.zeros((batch,FM_K**2,emb_size))).float().cuda(GPU_DEVICE)
        for i in range(FM_K):
            sub_item_emb = item_emb[:,i,:]
            sub_item_emb = sub_item_emb.view(batch,1,emb_size).expand(batch,FM_K,emb_size)
            sub_interact_emb = sub_item_emb.mul(common_his_item_emb)
            interact_emb[:,(i*FM_K):(i+1)*FM_K,:] = sub_interact_emb
        return interact_emb

    def pooling_layer(self,multi_relation_emb,multi_attention_weight):
        batch, attr_size, _ = multi_attention_weight.size()
        multi_attention_weight = multi_attention_weight.expand(batch,attr_size,self.emb_size)
        weighted_relation_emb = multi_attention_weight.mul(multi_relation_emb)
        relation_emb = weighted_relation_emb.sum(dim=1)
        return relation_emb

    def weight_attention(self,interact_emb): 
        hidden = self.weight_net(interact_emb)
        attention = hidden.matmul(self.weight_factor)
        soft_attention = F.softmax(attention,dim=1)
        return soft_attention

    def memory_attention(self,interact_emb):
        attention = interact_emb.matmul(self.memory_keys)
        soft_attention = F.softmax(attention,dim=2) 
        multi_relation_emb = soft_attention.matmul(self.memory_emb)
        return multi_relation_emb

    def relation_attention(self,item_emb,common_his_item_emb):
        interact_emb = self.interact_layer(item_emb,common_his_item_emb)
        multi_relation_emb = self.memory_attention(interact_emb) 
        multi_attention_weight = self.weight_attention(interact_emb) 
        relation_emb = self.pooling_layer(multi_relation_emb, multi_attention_weight)
        return relation_emb

    def rank_loss_construct(self,pos_sum_relation_emb,neg_sum_relation_emb,base_sum_relation_emb):
        pos_diff = torch.sum(pos_sum_relation_emb.mul(base_sum_relation_emb),dim=1)
        neg_diff = torch.sum(neg_sum_relation_emb.mul(base_sum_relation_emb),dim=1)

        rank_loss = torch.sum((neg_diff - pos_diff).sigmoid())
        return rank_loss

    def forward_train(self, common_his_item, pos_item, neg_item, base_item):
        batch = pos_item.size()[0]
        loss = torch.FloatTensor([0.0]).cuda(GPU_DEVICE)

        pos_item_emb = self.item_emb.index_select(0, Variable(pos_item.view(-1))).view(batch,self.FM_K,self.emb_size) # batch*FM_K*embed
        neg_item_emb = self.item_emb.index_select(0, Variable(neg_item.view(-1))).view(batch,self.FM_K,self.emb_size) 
        common_his_item_emb = self.item_emb.index_select(0, Variable(common_his_item.view(-1))).view(batch,self.FM_K,self.emb_size) 
        base_item_emb = self.item_emb.index_select(0, Variable(base_item.view(-1))).view(batch,self.FM_K,self.emb_size)

        pos_relation_emb = self.relation_attention(pos_item_emb,common_his_item_emb)
        neg_relation_emb = self.relation_attention(neg_item_emb,common_his_item_emb)
        base_relation_emb = self.relation_attention(base_item_emb,common_his_item_emb)

        rank_loss = self.rank_loss_construct(pos_relation_emb,neg_relation_emb,base_relation_emb)
        return rank_loss

    def user_prefer(self,user,pair_one,pair_two):
        batch = pair_one.size()[0]
        pair_one = self.item_emb.index_select(0, Variable(pair_one.view(-1))).view(batch,self.FM_K,self.emb_size)
        pair_two = self.item_emb.index_select(0, Variable(pair_two.view(-1))).view(batch,self.FM_K,self.emb_size)
        relation_emb = self.relation_attention(pair_one,pair_two)
        relation_emb = relation_emb.sum(dim=0)

        self.user_emb[user] = relation_emb

    def forward_test(self, user, target_item_pair_one, target_item_pair_two):
        user = torch.LongTensor(user).cuda(GPU_DEVICE)
        target_item_pair_one = torch.LongTensor(target_item_pair_one).cuda(GPU_DEVICE)
        target_item_pair_two = torch.LongTensor(target_item_pair_two).cuda(GPU_DEVICE)

        batch = target_item_pair_one.size()[0]

        user_emb = self.user_emb.index_select(0, Variable(user.view(-1))).view(1, self.emb_size) # 1*emb
        target_item_pair_one = self.item_emb.index_select(0, Variable(target_item_pair_one.view(-1))).view(batch,self.FM_K,self.emb_size) 
        target_item_pair_two = self.item_emb.index_select(0, Variable(target_item_pair_two.view(-1))).view(batch,self.FM_K,self.emb_size) 

        target_relation_emb = self.relation_attention(target_item_pair_one,target_item_pair_two)
        target_relation_emb = target_relation_emb.sum(dim=0)

        prefer_score = torch.sum(user_emb.mul(target_relation_emb))

        return prefer_score.cpu().data.numpy()

    def update(self, common_his_item, pos_item, neg_item, base_item):
        self.opt.zero_grad()
        loss = self.forward_train(common_his_item, pos_item, neg_item, base_item)

        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def model_train(self):
        for epoch in range(self.epoch_num):
            self.loss =  0.0
            loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

            start = time.time()

            for i_batch, sample_batched in enumerate(loader):
                end = time.time()
                # print("=======================> sample_data", i_batch, end-start)

                with torch.cuda.device(GPU_DEVICE):
                    self.update(sample_batched['common_his_item'],sample_batched['pos_item'],
                        sample_batched['neg_item'],sample_batched["base_item"])

                # end = time.time()
                # print("=======================> training", i_batch, end-start)

            print("\repoch "+ str(epoch) +" : avg loss = " + str(self.loss/len(self.data)))
            print("=============================> cosing time ", time.time()-start)

            # if epoch >= start_epoch:
            #     if (epoch) % inter_epoch == 0:
            #         self.model_test()
            #         # torch.cuda.empty_cache()

            if epoch >= start_epoch and (epoch) % inter_epoch == 0:
                test_start = time.time()
                self.model_test()
                print("One Test Has Finished", time.time()-test_start)

    def model_test(self,epoch=None):
        with torch.no_grad():
            for user in self.data.user2item:
                his_items = list(self.data.user2item[user])
                pair_one, pair_two = [], []

                max_len = len(his_items)
                if max_len > 500:
                    random.shuffle(his_items)
                for i in range(max_len):
                    for j in range(i+1,max_len):
                        pair_one.append(his_items[i])
                        pair_two.append(his_items[j])
                if max_len > 500:
                    pair_one = pair_one[:125000]
                    pair_two = pair_two[:125000]

                pair_one = torch.LongTensor(pair_one).cuda(GPU_DEVICE)
                pair_two = torch.LongTensor(pair_two).cuda(GPU_DEVICE)
                user = torch.LongTensor([user]).cuda(GPU_DEVICE)
                self.user_prefer(user,pair_one,pair_two)

            hr_5,ndcg_5,hr_10,ndcg_10,hr_15,ndcg_15 = test_util(self,self.dirname)
            max_hr_5 = max(self.max_measures[0],hr_5)
            max_ndcg_5 = max(self.max_measures[1],ndcg_5)
            max_hr_10 = max(self.max_measures[2],hr_10)
            max_ndcg_10 = max(self.max_measures[3],ndcg_10)
            max_hr_15 = max(self.max_measures[4],hr_15)
            max_ndcg_15 = max(self.max_measures[5],ndcg_15)

            print("<=================================================>")
            print("Max_HR@5: ", max_hr_5)
            print("Max_ndcg@5: ", max_ndcg_5)
            print("Max_HR@10: ", max_hr_10)
            print("Max_ndcg@10: ", max_ndcg_10)
            print("Max_HR@15: ", max_hr_15)
            print("Max_ndcg@15: ", max_ndcg_15)
            print("<=================================================>")
            self.max_measures = [max_hr_5, max_ndcg_5, max_hr_10, max_ndcg_10, max_hr_15, max_ndcg_15]

if __name__ == '__main__':
    mr_model = Reda()
    mr_model.model_train()
    mr_model.model_test()
