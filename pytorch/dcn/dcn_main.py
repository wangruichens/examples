# Auther        : wangrc
# Date          : 2018-12-27
# Description   :
# Refers        :
# Returns       :

import pytorch.dcn.dataset_helper as ds
import pytorch.dcn.DCN as DCNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda as cuda
import time
import numpy as np

train_dir = './forest_cover/train.csv'
test_dir='./forest_cover/test.csv'

train_data = ds.ForestDataset(train_dir)
# Drop : whether to drop the last batch may be not complete
train_loader = DataLoader(train_data, batch_size=64, num_workers=1, drop_last=False)

# device = torch.device("cuda:0" if cuda.is_available() else "cpu")
device='cpu'
print(device)
print(cuda.get_device_name(0))

model = DCNet.DCNet(embedding_index=[0], embedding_size=[40], dense_feature_num=14, cross_layer_num=3,
                    deep_layer=[256, 128, 32], output_num=7)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

loss_log=[]
iter_list=[]
acc_list=[]
epoch_num=5
iter_num=0

for epoch in range(epoch_num):
    model.train()
    for sparse_feature,dense_feature,label in train_loader:
        iter_num+=1
        begin_time=time.time()
        sparse_feature,dense_feature,label=sparse_feature.to(device),dense_feature.to(device),label.to(device)
        # print(sparse_feature.shape)
        # print(dense_feature.shape)
        output=model(sparse_feature,dense_feature)
        # print(output.shape)
        # print(label.shape)
        loss=criterion(output,label)
        v,i=torch.max(output,1)
        v2,i2=torch.max(label,1)
        acc=(i==i2).sum().data.numpy()/output.size()[0]
        # print('acc',acc)

        iter_loss=loss.item()

        loss_log.append(iter_loss)
        iter_list.append(iter_num)
        acc_list.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time=time.time()
        print("epoch {}/{}, total_iter is {}, logloss is {:.2f}, cost time is {:.2f}s, acc is {}".format(epoch + 1,
                                                                                                    epoch_num,
                                                                                              iter_num, iter_loss,
                                                                                              end_time - begin_time,
                                                                                                         acc))
        if iter_num % 20 == 0:
            total_loss = np.mean(loss_log)
            logloss = []

        if iter_num %1000==0:
            save_dir='./model'+str(iter_num)+'.pkl'
            torch.save(model.state_dict(),save_dir)
            model.train()

import matplotlib.pyplot as plt
plt.plot(iter_list,loss_log)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")

plt.figure()
plt.plot(iter_list,acc_list)
plt.xlabel("Number of iteration")
plt.ylabel("Acc")
plt.title("Acc vs Number of iteration")
# plt.show()



# Predict
#
# test_data=ds.ForestDatasetTest(test_dir)
# test_loader = DataLoader(test_data, batch_size=64, num_workers=1, drop_last=False,shuffle=False)
# predicted_list=[]
#
# with torch.no_grad():
#     for sparse_feature,dense_feature in test_loader:
#         sparse_feature,dense_feature=sparse_feature.to(device),dense_feature.to(device)
#         # print(sparse_feature.shape)
#         # print(dense_feature.shape)
#         output=model(sparse_feature,dense_feature)
#         # print(output.shape)
#         _, predicted = torch.max(output.data, 1)
#         # print(predicted.shape)
#         # print(predicted)
#         predicted_list=predicted_list+list(predicted.numpy())
#
# import pandas as pd
# df_pred = pd.DataFrame()
# aux = pd.read_csv(test_dir)
# df_pred['Id'] = aux['Id']
# df_pred['Cover_Type'] = [x+1 for x in predicted_list]
# df_pred.to_csv('./dcn.csv', index=False)
# print('fininshed...')
plt.show()









