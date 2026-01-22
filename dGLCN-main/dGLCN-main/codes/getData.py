import numpy as np
import scipy.io as sio
from utils import feature_normalized,preprocess_features,create_graph_from_embedding,preprocess_adj,sim_matrix_euclidean_distance

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import pandas as pd
import torch

def process_adj_SF(A):
    adj = create_graph_from_embedding(A, name='knn', n=30)
    adj, edge = preprocess_adj(adj)
    adj = adj.todense()
    adj = torch.from_numpy(adj).to(torch.float32)
    return adj


def permute_matrix(feature, label, index):
    seed_100 = np.arange(100)
    print("seed: ", seed_100[index])
    rand_list = np.random.RandomState(seed_100[2*index]).permutation(feature.shape[0])
    permute_feature = []
    permute_label = []
    for i in range(len(rand_list)):
        permute_feature.append(feature[rand_list[i]])
        permute_label.append(label[rand_list[i]])

    return np.array(permute_feature), np.array(permute_label),rand_list



def get_BL_Data(index,dataset):

    feature_file = 'data/PPMI_238_62NC_142PD_34SWEDD_mat_add_age_sex.mat'
    data_feature = sio.loadmat(feature_file)
    dataT1_CSF_DTI = data_feature['PPMI_238_62NC_142PD_34SWEDD_mat_add_age_sex']
    dataT1_CSF_DTI = dataT1_CSF_DTI[:,1:]
    T1_G_116 = dataT1_CSF_DTI[:, 0:116]   # 116 灰质
    T1_W_116 = dataT1_CSF_DTI[:, 116:232] # 116 白质
    T1_C_116 = dataT1_CSF_DTI[:, 232:348] # 116 T1_CSF
    FA_116 = dataT1_CSF_DTI[:, 348:464]   # 116 DTI_FA
    MD_116 = dataT1_CSF_DTI[:, 464:580]   # 116 DTI_MD
    L1_116 = dataT1_CSF_DTI[:, 580:696]   # 116 DTI_L1
    L2_116 = dataT1_CSF_DTI[:, 696:812]   # 116 DTI_L2
    L3_116 = dataT1_CSF_DTI[:, 812:928]   # 116 DTI_L3
    V1_116 = dataT1_CSF_DTI[:, 928:1044]  # 116 DTI_V1
    V2_116 = dataT1_CSF_DTI[:, 1044:1160] # 116 DTI_V2
    V3_116 = dataT1_CSF_DTI[:, 1160:1276] # 116 DTI_V3
    DSSM = dataT1_CSF_DTI[:, 1276:1280]   # 4 得分
    label = dataT1_CSF_DTI[:, 1280]       # 1 标签
    age = dataT1_CSF_DTI[:, 1281]         # 年龄
    sex = dataT1_CSF_DTI[:, 1282]         # 性别

    T1_G_116 = T1_G_116.astype(np.float64)
    T1_G_116 = feature_normalized(T1_G_116)

    L1_116 = L1_116.astype(np.float64)
    L1_116 = feature_normalized(L1_116)

    V1_116 = V1_116.astype(np.float64)
    V1_116 = feature_normalized(V1_116)



    Temp_Sample = np.hstack((T1_G_116,L1_116,V1_116))
    label_PD = label == -1
    label_SWEDD = label == 2
    label_NC = label == 1

    Sample_PD = Temp_Sample[label_PD,:]
    Sample_SWEDD = Temp_Sample[label_SWEDD,:]
    Sample_NC = Temp_Sample[label_NC,:]

    # Sample_PD = Sample_PD.astype(np.float64)
    # Sample_SWEDD = Sample_SWEDD.astype(np.float64)
    # Sample_NC = Sample_NC.astype(np.float64)



    if dataset == "PD-NC":
        sample = np.vstack((Sample_PD,Sample_NC))
        sampleLabel = np.hstack((np.ones(Sample_PD.shape[0]),0*np.ones(Sample_NC.shape[0])))
    elif dataset == "SWEDD-NC":
        sample = np.vstack((Sample_SWEDD,Sample_NC))
        sampleLabel = np.hstack((np.ones(Sample_SWEDD.shape[0]), 0 * np.ones(Sample_NC.shape[0])))
    elif dataset == "PD-SWEDD":
        sample = np.vstack((Sample_PD,Sample_SWEDD))
        sampleLabel = np.hstack((np.ones(Sample_PD.shape[0]), 0 * np.ones(Sample_SWEDD .shape[0])))

    # features, labels = permute_matrix(Sample, Label, index)


    column_sums = sample.sum(axis=0)
    nonzero_columns = np.where(column_sums != 0)[0]
    sample = sample[:, nonzero_columns]

    features, sampleLabel, rand_list = permute_matrix(sample, sampleLabel, index)

    features = preprocess_features(features)


    features1 = torch.tensor(features,dtype=torch.float32)
    adj = sim_matrix_euclidean_distance(features1,0.002)

    edge_index, edge_attr = dense_to_sparse(adj)

    sampleLabel = torch.tensor(sampleLabel, dtype=torch.long)
    sampleLabel = torch.nn.functional.one_hot(sampleLabel, 2)
    # sampleLabel = np.argmax(sampleLabel, axis=1)


    return features,edge_index,edge_attr,sampleLabel,rand_list

# def get_12M_Data(index,dataset):
#
#     dataBL,_,_,labelBL,rand_listBL = get_BL_Data(index,dataset)
#
#     feature_file = 'data/final_mat_12m_data_186_54NC_123PD_9SWEDD_add_age_sex.mat'
#     data_feature = sio.loadmat(feature_file)
#     dataT1_CSF_DTI = data_feature['final_mat_12m_data']
#     dataT1_CSF_DTI = dataT1_CSF_DTI[:,1:]
#     T1_G_116 = dataT1_CSF_DTI[:, 0:116]   # 116 灰质
#     T1_W_116 = dataT1_CSF_DTI[:, 116:232] # 116 白质
#     T1_C_116 = dataT1_CSF_DTI[:, 232:348] # 116 T1_CSF
#     FA_116 = dataT1_CSF_DTI[:, 348:464]   # 116 DTI_FA
#     MD_116 = dataT1_CSF_DTI[:, 464:580]   # 116 DTI_MD
#     L1_116 = dataT1_CSF_DTI[:, 580:696]   # 116 DTI_L1
#     L2_116 = dataT1_CSF_DTI[:, 696:812]   # 116 DTI_L2
#     L3_116 = dataT1_CSF_DTI[:, 812:928]   # 116 DTI_L3
#     V1_116 = dataT1_CSF_DTI[:, 928:1044]  # 116 DTI_V1
#     V2_116 = dataT1_CSF_DTI[:, 1044:1160] # 116 DTI_V2
#     V3_116 = dataT1_CSF_DTI[:, 1160:1276] # 116 DTI_V3
#     DSSM = dataT1_CSF_DTI[:, 1276:1279]   # 4 得分
#     label = dataT1_CSF_DTI[:, 1279]       # 1 标签
#     age = dataT1_CSF_DTI[:, 1280]         # 年龄
#     sex = dataT1_CSF_DTI[:, 1281]         # 性别
#
#     T1_G_116 = T1_G_116.astype(np.float64)
#     T1_G_116 = feature_normalized(T1_G_116)
#
#     L1_116 = L1_116.astype(np.float64)
#     L1_116 = feature_normalized(L1_116)
#
#     V1_116 = V1_116.astype(np.float64)
#     V1_116 = feature_normalized(V1_116)
#
#     Temp_Sample = np.hstack((T1_G_116,L1_116,V1_116))
#     label_PD = label == -1
#     label_SWEDD = label == 2
#     label_NC = label == 1
#     Sample_PD = Temp_Sample[label_PD,:]
#     Sample_SWEDD = Temp_Sample[label_SWEDD,:]
#     Sample_NC = Temp_Sample[label_NC,:]
#
#
#     if dataset == "PD-NC":
#         sample_12 = np.vstack((Sample_PD,Sample_NC))
#         sampleLabel_12 = np.hstack((np.ones(Sample_PD.shape[0]),0*np.ones(Sample_NC.shape[0])))
#     elif dataset == "SWEDD-NC":
#         sample_12 = np.vstack((Sample_SWEDD,Sample_NC))
#         sampleLabel_12 = np.hstack((np.ones(Sample_SWEDD.shape[0]), 0 * np.ones(Sample_NC.shape[0])))
#     elif dataset == "PD-SWEDD":
#         sample_12 = np.vstack((Sample_PD,Sample_SWEDD))
#         sampleLabel_12 = np.hstack((np.ones(Sample_PD.shape[0]), 0 * np.ones(Sample_SWEDD .shape[0])))
#
#
#
#     sample = np.vstack((dataBL,sample_12))
#
#     sampleLabel_12 = torch.tensor(sampleLabel_12, dtype=torch.long)
#     sampleLabel_12 = torch.nn.functional.one_hot(sampleLabel_12, 2)
#
#     sampleLabel = np.vstack((labelBL, sampleLabel_12))
#
#
#     column_sums = sample.sum(axis=0)
#     nonzero_columns = np.where(column_sums != 0)[0]
#     sample = sample[:, nonzero_columns]
#
#     features, sampleLabel,rand_list = permute_matrix(sample, sampleLabel, index)
#
#     features = preprocess_features(features)
#
#     features1 = torch.tensor(features, dtype=torch.float32)
#     adj = sim_matrix_euclidean_distance(features1, 0.002)
#
#     edge_index, edge_attr = dense_to_sparse(adj)
#
#     sampleLabel = torch.tensor(sampleLabel, dtype=torch.long)
#
#
#
#     return features,edge_index,edge_attr,sampleLabel,rand_list,rand_listBL
#
# def get_data_AD_SF(index, dataset):
#     feature_file = 'data/PPMIDel0_scores_303_67NC_199PD_37SWEDD_baseline.csv'
#
#     data_feature = pd.read_csv(feature_file)
#     data = data_feature.iloc[:, 1:]
#     data = data.values
#     GM = data[:, 0:116]
#     L1 = data[:, 580:696]
#     V1_1 = data[:, 928:1044]
#     data_label = data[:,-1]
#
#     data_feature = np.concatenate((GM, L1, V1_1), axis=1)
#
#
#
#     PD = data_feature[data_label == 2,:]
#     SWEDD = data_feature[data_label == 1,:]
#     NC = data_feature[data_label == 0,:]
#
#     if dataset == "PD-NC":
#         feature = np.concatenate((PD, NC), axis=0)
#         label = np.concatenate((data_label[data_label == 2], data_label[data_label == 0]))
#         label = np.column_stack((label == 2, label == 0)).astype(int)
#         label = np.argmax(label, axis=1)
#
#     elif dataset == "SWEDD-NC":
#         feature = np.concatenate((SWEDD, NC), axis=0)
#         label = np.concatenate((data_label[data_label == 1], data_label[data_label == 0]))
#         label = np.column_stack((label == 1, label == 0)).astype(int)
#         label = np.argmax(label, axis=1)
#     elif dataset == "PD-SWEDD":
#         feature = np.concatenate((PD, SWEDD), axis=0)
#         label = np.concatenate((data_label[data_label == 2], data_label[data_label == 1]))
#         label = np.column_stack((label == 2, label == 1)).astype(int)
#         label = np.argmax(label, axis=1)
#
#
#     # data_feature = sio.loadmat(feature_file)
#     # data_label = sio.loadmat(label_file)
#
#     # feature = data_feature['feature']
#     # label = data_label['label']
#     # label = np.argmax(label, axis=1)
#
#     features, labels = permute_matrix(feature, label, index)
#     features = feature_normalized(features)
#
#     adj = process_adj_SF(features)
#     # s = process_adj_SF(features.T)
#     features = preprocess_features(features)
#     edge_index, edge_attr = dense_to_sparse(adj)
#
#     return features,edge_index,edge_attr,labels



