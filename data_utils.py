# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
import scipy

import torch.nn as nn 
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch import linalg as LA
import torch
import math
import random
import collections
import json

from sklearn.cluster import KMeans
import torch.nn.functional as F
import C

t = 1

class BPR(nn.Module):
    def __init__(self, user_num,item_num,factor_num,user_item_matrix,item_user_matrix,
        old_sparse_u_i=None,
        old_sparse_i_u=None,
        old_U_emb=None,
        old_I_emb=None):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """

        # Original LightGCN
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.user_num = user_num
        self.item_num = item_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        if old_U_emb == None and old_I_emb == None:
            pass
        else:
            # Distillation 
            self.old_U_emb = old_U_emb
            self.old_I_emb = old_I_emb


    def contrastive(self, user, item_i, item_j, degree, item_z, user_num, gcn_users_embedding, gcn_items_embedding):

        user = F.embedding(user, gcn_users_embedding)
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)

        numerator = torch.exp( (user*item_i).sum(dim=-1) / t )
        denominator = torch.exp( (user*item_j).sum(dim=-1) / t ) + torch.exp( (user*item_i).sum(dim=-1) / t )
        loss = - (torch.log(numerator / denominator)*degree).sum() / user_num

        return loss

    def forward(self, user, item_i, item_j, degree_U, item_z_U, user_, item_i_, item_j_, degree_I, item_z_I):

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
        
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

        loss = torch.zeros(1,).cuda()

        loss += self.contrastive(user, item_i, item_j, degree_U, item_z_U, self.user_num, self.old_U_emb, gcn_users_embedding)
        loss += self.contrastive(user_, item_i_, item_j_, degree_I, item_z_I, self.item_num, self.old_I_emb, gcn_items_embedding)

        return loss

    def inference(self):

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding

        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

        return gcn_users_embedding, gcn_items_embedding

def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d

def readTrainSparseMatrix(set_matrix,u_d,i_d,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        # len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    # user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    # user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    user_items_matrix_i=torch.tensor(user_items_matrix_i, dtype=torch.long, device='cuda')
    user_items_matrix_v=torch.tensor(user_items_matrix_v, dtype=torch.float, device='cuda')
    # return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
    return torch.sparse_coo_tensor(user_items_matrix_i.t(), user_items_matrix_v, dtype=torch.float, device='cuda')

class BPRData(data.Dataset):
    def __init__(self, train_dict_U=None, train_dict_I=None, num_item=0, num_user=0, num_ng=1, is_training=None, data_set_count=0, all_rating_U=None, all_rating_I=None):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.num_user = num_user
        self.train_dict_U = train_dict_U
        self.train_dict_I = train_dict_I
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating_U=all_rating_U
        self.all_rating_I=all_rating_I
        self.set_all_item=set(range(num_item))
        self.set_all_user=set(range(num_user))
        self.degree_U = np.zeros(len(train_dict_U))
        self.degree_I = np.zeros(len(train_dict_I))
        for x in self.train_dict_U:
            self.degree_U[x] = len(self.train_dict_U[x]) + 1
        for x in self.train_dict_I:
            self.degree_I[x] = len(self.train_dict_I[x]) + 1
        self.item_z_U = np.zeros([len(self.train_dict_U), self.num_item])
        self.item_z_I = np.zeros([len(self.train_dict_I), self.num_user])

    def ng_sample(self):

        self.features_fill_U = []
        for user_id in self.train_dict_U:
            positive_list=self.train_dict_U[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating_U[user_id]

            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill_U.append([user_id,item_i,item_j])
                    self.item_z_U[user_id][item_j] = 1.

        self.features_fill_I = []
        for user_id in self.train_dict_I:
            positive_list=self.train_dict_I[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating_I[user_id]

            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_user)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_user)
                    self.features_fill_I.append([user_id,item_i,item_j])
                    self.item_z_I[user_id][item_j] = 1.

    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict)

    def __getitem__(self, idx):
        features_U = self.features_fill_U
        features_I = self.features_fill_I

        user = features_U[idx][0]
        item_i = features_U[idx][1]
        item_j = features_U[idx][2]

        user_ = features_I[idx][0]
        item_i_ = features_I[idx][1]
        item_j_ = features_I[idx][2]

        return user, item_i, item_j, self.item_z_U[user], self.degree_U[user], user_, item_i_, item_j_, self.item_z_I[user_], self.degree_I[user_]