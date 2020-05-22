# -*- coding:utf-8 -*-

import numpy as np 
import random
import math

def get_test_batch(test_file, dataset):
    itemset = dataset.itemset
    fr = open(test_file,"r")
    for line in fr:
        lineArr = line.split(":")
        user = np.array([int(lineArr[0])])

        common_his_items = [int(val) for val in lineArr[1].split()]
        candidate_items = list(itemset - set(common_his_items))

        target_item = [int(val) for val in lineArr[2].split()]
        candidate_items = random.sample(candidate_items,100) + target_item

        if len(target_item) == 0:
            continue

        yield user, common_his_items, candidate_items, target_item
    fr.close()

def test_util(model,dirname):
    THR_5 = 0.0
    TNDCG_5 = 0.0
    THR_10 = 0.0
    TNDCG_10 = 0.0
    THR_15 = 0.0
    TNDCG_15 = 0.0
    
    count = 0

    for user, common_his_items, candidate_items, target_item in get_test_batch(dirname + "/testset.txt",model.data):
        result = {}
        for candidate_item in candidate_items:
            target_item_pair_two = common_his_items
            target_item_pair_one = []
            for _ in range(len(common_his_items)):
                target_item_pair_one.append(candidate_item)
            
            rec_score = model.forward_test(user, target_item_pair_one, target_item_pair_two)
            result[candidate_item] = rec_score

        result = sorted(result.items(), key=lambda item:item[1], reverse=True)
        topN = [int(item[0]) for item in result]
        count += 1
    
        hr_5 = HR(topN,target_item,5)
        THR_5 += hr_5
        nd_5 = ndcg(topN,target_item,5)
        TNDCG_5 += nd_5

        hr_10 = HR(topN,target_item,10)
        THR_10 += hr_10
        nd_10 = ndcg(topN,target_item,10)
        TNDCG_10 += nd_10

        hr_15 = HR(topN,target_item,15)
        THR_15 += hr_15
        nd_15 = ndcg(topN,target_item,15)
        TNDCG_15 += nd_15

    print("<=================================================>")
    print("HR@5: ", THR_5/count)
    print("ndcg@5: ", TNDCG_5/count)
    print("HR@10: ", THR_10/count)
    print("ndcg@10: ", TNDCG_10/count)
    print("HR@15: ", THR_15/count)
    print("ndcg@15: ", TNDCG_15/count)
    print("<=================================================>")

    return THR_5/count, TNDCG_5/count, THR_10/count, TNDCG_10/count, THR_15/count, TNDCG_15/count
        
def HR(recList,selectList,L):
    count = 0
    for item_id in selectList:
        item_id = int(item_id)
        rank = recList.index(item_id)
        if rank < L:
            count += 1
    return float(count)

def ndcg(recList,selectList,L):
    resultList = []
    flag = 0
    for i in range(L):
        item = str(recList[i])
        if item in selectList or int(item) in selectList:
            resultList.append(1)
            flag = 1
        else:
            resultList.append(0)
    if flag == 0:
        return 0.0
    dcg = getDCG(resultList)
    idcg = getIdcg(resultList)
    return dcg/idcg

def getDCG(resultList):
    dcg = resultList[0]
    i = 3
    for val in resultList[1:]:
        dcg += (pow(2,val)-1)/math.log(i,2)
        i += 1
    return dcg

def getIdcg(resultList):
    resultList.sort()
    resultList.reverse()
    return getDCG(resultList)