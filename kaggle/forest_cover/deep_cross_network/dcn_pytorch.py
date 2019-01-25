# Auther        : wangrc
# Date          : 2018-12-25
# Description   :
# Refers        :
# Returns       :

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Id,Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,
#
# Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40,
#
# Cover_Type


if __name__ == '__main__':
    ## Load data
    tr = pd.read_csv('./forest_cover/train.csv')

    ## Preprocess
    onehot = tr[[c for c in tr.columns.tolist() if 'Soil_Type' in c]]
    tr["Soil"] = onehot.dot(np.array(range(onehot.columns.size))).astype(int)
    cate_col = ['Soil']

    tr.drop([c for c in tr.columns.tolist() if 'Soil_Type' in c], axis=1, inplace=True)

    # Handle label
    y = np.array(OneHotEncoder().fit_transform(tr['Cover_Type'].values.reshape(-1, 1)).todense())
    del tr['Cover_Type']

    cont_col = [c for c in tr.columns if c != 'Soil']


    # Feature Nornolization
    scaler = StandardScaler()
    # x-mean / variance
    tr_cont = pd.DataFrame(scaler.fit_transform(tr[cont_col]), columns=cont_col)
    tr_cate = tr[cate_col]
    print(tr_cont.columns)
    final_tr = pd.concat([tr_cate, tr_cont], axis=1)
    cate_val = final_tr[cate_col].values
    cont_val = final_tr[cont_col].values

    # Embedding
    # embedding_tensor = []
    # continuous_tensor = []

    cate_num = []
    print(cate_val.shape)
    for i in range(cate_val.shape[1]):
        unique_num = np.unique(cate_val[:, i]).shape[0]
        cate_num.append(unique_num)
    print(cate_num)

    print(y.shape)
    x=final_tr.values
    print(type(y))
    print(type(x))
    print(len(y))
    print(len(final_tr))
    print(x[1])
    print(y[1])