# Auther        : wangrc
# Date          : 2018-12-26
# Description   :
# Refers        :
# Returns       :


import torch
import torch.nn as nn
import numpy as np


class DeepNet(nn.Module):
    def __init__(self, feature_num, deep_layer: list):
        super(DeepNet, self).__init__()
        fc_layer_list = []
        fc_layer_list.append(nn.Linear(feature_num, deep_layer[0]))
        # batch norm
        fc_layer_list.append(nn.BatchNorm1d(deep_layer[0], affine=False))
        fc_layer_list.append(nn.ReLU(inplace=True))

        for i in range(1, len(deep_layer)):
            fc_layer_list.append(nn.Linear(deep_layer[i - 1], deep_layer[i]))
            fc_layer_list.append(nn.BatchNorm1d(deep_layer[i], affine=False))
            fc_layer_list.append(nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*fc_layer_list)

    def forward(self, input):
        dense_output=self.deep(input)
        return dense_output

class CrossNet(nn.Module):
    def __init__(self,feature_num,cross_layer: int):
        super(CrossNet,self).__init__()
        self.cross_layer=cross_layer+1 # add first calculate
        weight_w=[]
        weight_b=[]
        batch_norm=[]
        for i in range(self.cross_layer):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(feature_num))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(feature_num))))
            batch_norm.append(nn.BatchNorm1d(feature_num,affine=False))
        self.weight_w=nn.ParameterList(weight_w)
        self.weight_b=nn.ParameterList(weight_b)
        self.batch_norm=nn.ModuleList(batch_norm)

    def forward(self, input):
        # Residual part
        # input is the original x0, never updated.
        output=input
        # Flatten the input
        input=input.reshape(input.shape[0],-1,1)

        for i in range(self.cross_layer):
            output=torch.matmul(torch.bmm(
                input,torch.transpose(output.reshape(output.shape[0],-1,1),1,2))
                                , self.weight_w[i])+self.weight_b[i]+output
            output=self.batch_norm[i](output)
        # The output size is unchanged
        return output

class DCNet(nn.Module):
    def __init__(self,embedding_index: list, embedding_size:list,dense_feature_num:int, cross_layer_num:int,deep_layer:list,output_num):
        super(DCNet, self).__init__()
        assert len(embedding_index)==len(embedding_size)
        self.embedding_index=embedding_index
        self.embedding_size=embedding_size
        self.output_num=output_num
        embedding_num = list(map(lambda x: int(6 * pow(x, 0.25)), self.embedding_size))
        # embedding_num=10
        input_feature_num=np.sum(embedding_num)+dense_feature_num
        embedding_list=[]
        # For each categorial feature
        for i in range(len(embedding_size)):
            embedding_list.append(nn.Embedding(embedding_size[i],embedding_num[i],scale_grad_by_freq=True))
            self.embedding_layer=nn.ModuleList(embedding_list)
            self.batch_norm=nn.BatchNorm1d(input_feature_num,affine=False)
            self.cross_net=CrossNet(input_feature_num,cross_layer_num)
            self.deep_net=DeepNet(input_feature_num,deep_layer)
            last_layer_feature_num=input_feature_num+deep_layer[-1]
            self.output_layer=nn.Linear(last_layer_feature_num,self.output_num)

    def forward(self, sparse_feature, dense_feature):
        num_sample=sparse_feature.shape[0]
        if isinstance(self.embedding_index[0],list):
            embedding_feature=torch.mean(self.embedding_layer[0](sparse_feature[:,self.embedding_index[0]].to(torch.long)),dim=1)
        else:
            embedding_feature=torch.mean(self.embedding_layer[0](sparse_feature[:,self.embedding_index[0]].to(torch.long).reshape(num_sample,1)),dim=1)
        for i in range(1, len(self.embedding_index)):
            if isinstance(self.embedding_index[i], list):
                embedding_feature = torch.cat((embedding_feature, torch.mean(self.embedding_layer[i](sparse_feature[:, self.embedding_index[i]].to(torch.long)), dim=1)), dim=1)
            else:
                embedding_feature = torch.cat((embedding_feature, torch.mean(self.embedding_layer[i](sparse_feature[:, self.embedding_index[i]].to(torch.long).reshape(num_sample, 1)), dim=1)), dim=1)
        input_feature = torch.cat((embedding_feature, dense_feature), 1)
        # print(input_feature.shape)
        input_feature=self.batch_norm(input_feature)
        out_cross=self.cross_net(input_feature)
        out_deep=self.deep_net(input_feature)
        final_feature=torch.cat((out_cross,out_deep),dim=1)
        output=self.output_layer(final_feature)
        # output=output.view(-1)
        # output=torch.sigmoid(output)
        return output




































