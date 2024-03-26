# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import wandb

import argparse
import os
import numpy as np
import math
import sys

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

os.environ["WANDB_API_KEY"] = "15b5e8572b0516899bae70d5bbb5c9091d1667a7"

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd

import pdb
from collections import defaultdict
import time
import collections
from shutil import copyfile

from evaluate import *
from data_utils import *
import C

wandb.login()

dataset_base_path='./Gowalla'

epoch_num=C.EPOCH_NUM   

user_num=C.user_num
item_num=C.item_num

factor_num=256
batch_size=1024*4
learning_rate=0.0001

num_negative_test_val=-1##all

run_id=C.RUN_ID
print(run_id)
dataset='Gowalla'

path_save_model_base='./newlossModel_LWC/'+dataset+'/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

base = read(dataset_base_path + "/check_in.json", [0, 0.6])
block = read(dataset_base_path + "/check_in.json", [0.6, 0.7])
training_user_set, training_item_set = list_to_set(block)
training_user_set_, training_item_set_ = list_to_set(base)
training_set_count = count_interaction(training_user_set)
training_set_count_ = count_interaction(training_user_set_)
user_rating_set_all, item_rating_set_all = json_to_set(dataset_base_path + "/check_in.json")

print("block: ", training_set_count)
print("base: ", training_set_count_)

training_user_set[user_num-1].add(item_num-1)
training_item_set[item_num-1].add(user_num-1)
training_user_set_[user_num-1].add(item_num-1)
training_item_set_[item_num-1].add(user_num-1)

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
u_d_=readD(training_user_set_,user_num)
i_d_=readD(training_item_set_,item_num)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)
sparse_u_i_=readTrainSparseMatrix(training_user_set_,u_d_,i_d_,True)
sparse_i_u_=readTrainSparseMatrix(training_item_set_,u_d_,i_d_,False)

train_dataset = BPRData(
        train_dict_U=training_user_set_, train_dict_I=training_item_set_, 
        num_item=item_num, num_user=user_num, 
        num_ng=C.NUM_NG, is_training=True, data_set_count=training_set_count_, 
        all_rating_U=user_rating_set_all, all_rating_I=item_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)

PATH_model='./newlossModel_LWC/'+dataset+'/'+C.BASE+'/epoch'+str(C.BASE_EPOCH)+'.pt'

model_ = BPR(user_num, item_num, factor_num, sparse_u_i_, sparse_i_u_).to('cuda')
model_.load_state_dict(torch.load(PATH_model))
model_.eval()
with torch.no_grad():
    old_U_emb, old_I_emb = model_.inference()

model = BPR(user_num,item_num,factor_num,sparse_u_i,sparse_i_u,
    old_U_emb=old_U_emb,
    old_I_emb=old_I_emb).to('cuda')

# X = 801
# PATH_model='./newlossModel_LWC/'+dataset+'/'+"s1"+'/epoch'+ str(X) + '.pt'

model.load_state_dict(torch.load(PATH_model))
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.5, 0.99))

run = wandb.init(
    project="KD-LWC",
    name=run_id,
    mode="offline",
)

########################### TRAINING #####################################

# testing_loader_loss.dataset.ng_sample()

print('--------training processing-------')
for epoch in range(epoch_num):

    model.train() 
    start_time = time.time()

    train_loader.dataset.ng_sample()

    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()

    train_loss_sum=[]

    # degree = torch.tensor(train_loader.dataset.degree).cuda()
    # item_z = torch.tensor(train_loader.dataset.item_z).cuda()

    for user, item_i, item_j, item_z_U, degree_U, user_, item_i_, item_j_, item_z_I, degree_I in train_loader:

        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()
        item_z_U = item_z_U.int().cuda()
        degree_U = degree_U.int().cuda()

        user_ = user_.cuda()
        item_i_ = item_i_.cuda()
        item_j_ = item_j_.cuda()
        item_z_I = item_z_I.int().cuda()
        degree_I = degree_I.int().cuda()

        model.zero_grad()
        loss = model(user, item_i, item_j, degree_U, item_z_U, user_, item_i_, item_j_, degree_I, item_z_I)
        loss.backward()

        optimizer_bpr.step()
        train_loss_sum.append(loss.item())

        elapsed_time = time.time() - start_time

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)

    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)
    # print('--train--',elapsed_time)

    wandb.log({"train_loss": train_loss})

    print(str_print_train)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)
