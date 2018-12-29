# Auther        : wangrc
# Date          : 2018-12-26
# Description   :
# Refers        :
# Returns       :
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler


class ForestDataset(Dataset):
    def __init__(self,path):
        self.all_tr,self.cate_tr,self.cont_tr,self.label=self.load_data(path)

    def __len__(self):
        return  len(self.label)

    def __getitem__(self, index):
        sparse_feature,dense_feature,label=self.__process__(index)
        return torch.tensor(sparse_feature,dtype=torch.float),torch.tensor(dense_feature,dtype=torch.float),\
               torch.tensor(label,dtype=torch.float)

    def load_data(self,path):
        ## Load data
        tr = pd.read_csv(path)
        del tr['Id']

        ## Preprocess
        onehot = tr[[c for c in tr.columns.tolist() if 'Soil_Type' in c]]
        tr["Soil"] = onehot.dot(np.array(range(onehot.columns.size))).astype(int)
        cate_col = ['Soil']

        tr.drop([c for c in tr.columns.tolist() if 'Soil_Type' in c], axis=1, inplace=True)

        # Handle label
        label = np.array(OneHotEncoder().fit_transform(tr['Cover_Type'].values.reshape(-1, 1)).todense())
        del tr['Cover_Type']

        cont_col = [c for c in tr.columns if c != 'Soil']

        # Feature Nornolization
        scaler = StandardScaler()
        # x-mean / variance
        cont_tr = pd.DataFrame(scaler.fit_transform(tr[cont_col]), columns=cont_col)
        cate_tr = tr[cate_col]
        final_tr = pd.concat([cate_tr, cont_tr], axis=1)
        return final_tr,cate_tr,cont_tr,label


    def __process__(self,index):
        sparse_feature=self.cate_tr.values[index]
        dense_feature=self.cont_tr.values[index]
        label=self.label[index]
        return sparse_feature,dense_feature,label


class ForestDatasetTest(Dataset):
    def __init__(self,path):
        self.all_tr,self.cate_tr,self.cont_tr=self.load_data(path)

    def __len__(self):
        return  len(self.cate_tr)

    def __getitem__(self, index):
        sparse_feature,dense_feature=self.__process__(index)
        return torch.tensor(sparse_feature,dtype=torch.float),torch.tensor(dense_feature,dtype=torch.float)

    def load_data(self,path):
        ## Load data
        tr = pd.read_csv(path)
        del tr['Id']

        ## Preprocess
        onehot = tr[[c for c in tr.columns.tolist() if 'Soil_Type' in c]]
        tr["Soil"] = onehot.dot(np.array(range(onehot.columns.size))).astype(int)
        cate_col = ['Soil']

        tr.drop([c for c in tr.columns.tolist() if 'Soil_Type' in c], axis=1, inplace=True)

        cont_col = [c for c in tr.columns if c != 'Soil']

        # Feature Nornolization
        scaler = StandardScaler()
        # x-mean / variance
        cont_tr = pd.DataFrame(scaler.fit_transform(tr[cont_col]), columns=cont_col)
        cate_tr = tr[cate_col]
        final_tr = pd.concat([cate_tr, cont_tr], axis=1)
        return final_tr,cate_tr,cont_tr


    def __process__(self,index):
        sparse_feature=self.cate_tr.values[index]
        dense_feature=self.cont_tr.values[index]
        return sparse_feature,dense_feature