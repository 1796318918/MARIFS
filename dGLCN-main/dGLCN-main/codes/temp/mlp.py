import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
import scipy.io as sio
from utils import feature_normalized, create_graph_from_embedding, sample_mask, preprocess_adj, preprocess_features
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score,auc,roc_curve
from heapq import nlargest
from collections import Counter
from torch_geometric.nn import MLP, global_mean_pool,GCNConv
from sklearn.model_selection import StratifiedKFold
from getData import get_BL_Data,get_data_AD_SF
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score
from matplotlib import pyplot as plt
from scipy.io import savemat

class MLPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x):
        x = self.mlp(x)

        return x


criterion = torch.nn.CrossEntropyLoss()
def train_SF():
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(tensor_x)
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = criterion(logits[train_mask], labels[train_mask].to(torch.float32))
        loss.backward()
        optimizer.step()

    return loss


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)  # 灵敏度
    specificity = tn / (tn + fp)  # 特异性

    return sensitivity, specificity
def sen_spe(a, b):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(len(a)):  # a:label
        if a[i] == 1 and b[i] == 1:
            TP = TP + 1
        elif a[i] == 1 and b[i] == 0:
            FN = FN + 1
        elif a[i] == 0 and b[i] == 1:
            FP = FP + 1
        elif a[i] == 0 and b[i] == 0:
            TN = TN + 1
        else:
            pass

    TPR = TP / (TP + FN + 1e-6)  # True positive rate, Sensitivity
    TNR = TN / (TN + FP + 1e-6)  # True Negative Rate, Specificity
    return TPR, TNR

def test_SF(mask):
    model.eval()

    out = model(tensor_x)
    mask_logits = out[mask]

    predicted = mask_logits.max(dim=1)[1].long()
    label2 = labels[mask]

    predicted = predicted.cpu().detach().numpy()
    label = label2.cpu().detach().numpy()
    label = np.argmax(label, axis=1)

    accuracy = accuracy_score(label, predicted)
    sensitivity, specificity = sen_spe(label, predicted)
    continue_pred = F.softmax(mask_logits, dim=1)
    continue_pred = continue_pred[:, 1]
    auc = roc_auc_score(label, continue_pred.cpu().detach().numpy())
    f1 = f1_score(label, predicted)
    Prec = precision_score(label,predicted,zero_division=1)
    Uar = recall_score(label,predicted,average='macro')
    return accuracy, sensitivity, specificity, auc, f1, Prec, Uar, label2,out[mask]


def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])


# dataset = ["PD-NC","SWEDD-NC", "PD-SWEDD"]
dataset = ["PD-SWEDD"]

for dataset_index in range(len(dataset)):
    if __name__ == "__main__":
        learning_rate = 0.2
        weight_decay = 1e-2
        epochs = 50

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ACC_1 = []
        log = '{:.4f}  {:.4f} {:.4f} {:.4f}'
        List_acc = []
        List_sen = []
        List_spe = []
        List_auc = []
        List_F1 = []
        List_PREC = []
        List_UAR = []
        FEA = []

        List_Ldec = []
        List_Ltest = []
        Res1 = {}

        for j in range(10):
            x,edge_index,edge_weight,labels,rand_list = get_BL_Data(j, dataset[dataset_index])

            labels = labels.to(device)

            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            Ldec = np.zeros((x.shape[0], 2))
            Ltest = np.zeros((x.shape[0], 2))

            input = x.shape[1]
            output = 2

            kf = KFold(n_splits=10)
            ACC = []
            SEN = []
            SPE = []
            PREC = []
            AUC = []
            UAR = []
            F1 = []
            FEATURE = []
            best_acc = 0
            torch.manual_seed(0)
            index = 0
            stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            y1 = labels.cpu().numpy()
            y1 = y1[:, 0]

            for idx_train, idx_test in stratified_kfold.split(x,y1):
                pth = dataset[dataset_index] + "net_params_" + str(index) + ".pth"
                train_mask = sample_mask(idx_train, labels.shape[0])
                val_mask = sample_mask(idx_test, labels.shape[0])
                test_mask = sample_mask(idx_test, labels.shape[0])

                tensor_x = torch.from_numpy(x).to(torch.float32).to(device)

                model = MLPNet(input, 50,output).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                #
                # model.load_state_dict(torch.load(pth))
                #
                loss = train_SF()


                test_accs, test_sen, test_spe, test_auc,test_f1, test_prec, test_uar,maxAuc_Ltest,maxAuc_Ldec = test_SF(test_mask)
                maxAuc_Ltest = maxAuc_Ltest.cpu().detach().numpy()
                maxAuc_Ldec = maxAuc_Ldec.cpu().detach().numpy()
                Ldec[rand_list[idx_test]] = maxAuc_Ldec
                Ltest[rand_list[idx_test]] = maxAuc_Ltest

                print(test_accs)

                ACC.append(test_accs)
                SEN.append(test_sen)
                SPE.append(test_spe)
                AUC.append(test_auc)
                F1.append(test_f1)
                UAR.append(test_uar)
                PREC.append(test_prec)

                index += 1
                count = topn_dict(Counter(FEATURE), 20)
                # print(count.sort())
                FEA = FEA + count

            mean_acc = np.mean(np.array(ACC))
            List_acc.append(mean_acc)
            print(j)
            print(dataset[dataset_index], "： Mean times", j, "ACC+/-std", mean_acc, "+/-", np.std(np.array(ACC)))

            mean_sen = np.mean(np.array(SEN))
            List_sen.append(mean_sen)
            print("Mean times", j, "SEN+/-std", mean_sen, "+/-", np.std(np.array(SEN)))

            mean_spe = np.mean(np.array(SPE))
            List_spe.append(mean_spe)
            print("Mean times", j, "SPE+/-std", mean_spe, "+/-", np.std(np.array(SPE)))

            mean_prec = np.mean(np.array(PREC))
            List_PREC.append(mean_prec)
            print("Mean times", j, "PREC+/-std", mean_prec, "+/-", np.std(np.array(PREC)))

            mean_uar = np.mean(np.array(UAR))
            List_UAR.append(mean_uar)
            print("Mean times", j, "UAR+/-std", mean_uar, "+/-", np.std(np.array(UAR)))

            mean_f1 = np.mean(np.array(F1))
            List_F1.append(mean_f1)
            print("Mean times", j, "F1score+/-std", mean_f1, "+/-", np.std(np.array(F1)))

            mean_auc = np.mean(np.array(AUC))
            List_auc.append(mean_auc)
            print("Mean times", j, "AUC+/-std", mean_auc, "+/-", np.std(np.array(AUC)))

            List_Ldec.append(Ldec.tolist())
            List_Ltest.append(Ltest.tolist())

        maxSvmFs_Ldec = np.mean(np.array(List_Ldec), axis=0)
        maxSvmFs_Ltest = np.mean(np.array(List_Ltest), axis=0)
        plt.figure()
        fpr_seed, tpr_seed, _ = roc_curve(maxSvmFs_Ltest.ravel(), maxSvmFs_Ldec.ravel())
        roc_auc = auc(fpr_seed, tpr_seed)

        tosavedata = {'Res1': {'svm_Rs': {'FP': fpr_seed, 'TP': tpr_seed, 'AUC': roc_auc}}}
        # Res1 = {'tpr_seed': tpr_seed, 'fpr_seed': fpr_seed}
        data_to_save = tosavedata
        file_path = 'R1_D1111_3.12.Mat__PDvsSW6_MLP_baseline.mat'

        # 使用 savemat 保存.mat文件
        savemat(file_path, data_to_save)

        plt.plot(fpr_seed, tpr_seed, lw=2)
        #
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()



        print("The final result after 100 times: ")
        print("Mean times", j, "ACC+/-std", np.mean(np.array(List_acc)), "+/-", np.std(np.array(List_acc)))
        print("Mean times", j, "SEN+/-std", np.mean(np.array(List_sen)), "+/-", np.std(np.array(List_sen)))
        print("Mean times", j, "SPE+/-std", np.mean(np.array(List_spe)), "+/-", np.std(np.array(List_spe)))
        print("Mean times", j, "PREC+/-std", np.mean(np.array(List_PREC)), "+/-", np.std(np.array(List_PREC)))
        print("Mean times", j, "UAR+/-std", np.mean(np.array(List_UAR)), "+/-", np.std(np.array(List_UAR)))
        print("Mean times", j, "F1score+/-std", np.mean(np.array(List_F1)), "+/-", np.std(np.array(List_F1)))
        print("Mean times", j, "AUC+/-std", np.mean(np.array(List_auc)), "+/-", np.std(np.array(List_auc)))

        # print(Counter(FEA))