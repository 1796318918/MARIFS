import argparse

from matplotlib import pyplot as plt


import numpy as np

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

def read_csv(path):
    x = np.empty((len(path), 261, 3))
    y = np.empty((len(path), 261, 3))
    for step, path_one in enumerate(path):
        path_one = './' + path_one
        data = np.loadtxt(path_one, delimiter=',')
        x[step], y[step] = np.split(data, 2, axis=1)
    Draw_Roc_Class(x, y)
    # print(x)


def Draw_Roc_Class(y_label, y_score):
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for x in range(1):
        fpr[x] = {}
        tpr[x] = {}
        roc_auc[x] = {}
        for i in range(n_classes):
            fpr[x][i], tpr[x][i], _ = roc_curve(y_label[x, :, i], y_score[x, :, i])
            roc_auc[x][i] = auc(fpr[x][i], tpr[x][i])

    colors = cycle(['aqua', 'darkorange'])


    #todo 画微平均ROC曲线
    plt.figure()
    for x, color in zip(range(1), colors):
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label[x].ravel(), y_score[x].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        if x == 0:
            p = 'our'
        elif x == 1:
            p = 'ref'

        plt.plot(fpr["micro"], tpr["micro"], color=color, lw=2,
                 label='ROC curve of  {0} (area = {1:0.2f})'
                       ''.format(p, roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro' + ' ROC')
    plt.legend(loc="lower right")
    plt.show()


    #todo 画宏平均ROC曲线
    plt.figure()
    for x, color in zip(range(1), colors):
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[x][i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[x][i], tpr[x][i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        if x == 0:
            p = 'our'
        elif x == 1:
            p = 'ref'

        plt.plot(fpr["macro"], tpr["macro"], color=color, lw=2,
                 label='ROC curve of  {0} (area = {1:0.2f})'
                       ''.format(p, roc_auc["macro"]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro' + ' ROC')
    plt.legend(loc="lower right")
    plt.show()

    lw = 2

    #todo 画各个类别的ROC曲线
    for i in range(3):
        plt.figure()
        for x, color in zip(range(1), colors):
            if x == 0:
                p = 'our'
            elif x == 1:
                p = 'ref'

            plt.plot(fpr[x][i], tpr[x][i], color=color, lw=lw,
                     label='ROC curve of {0} (area = {1:0.3f})'
                           ''.format(p, roc_auc[x][i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if i == 0:
            args.name = 'Hyperechoic'
        elif i ==1:
            args.name = 'Hypoechoic'
        else:
            args.name = 'Mixed-echoic'
        plt.title(args.name+' ROC')
        plt.legend(loc="lower right")
        plt.show()


def Draw_Roc(y_label, y_score, name):

    n_classes = 3
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        if i == 0:
            x = 'high'
        elif i == 1:
            x = 'low'
        else:
            x = 'mix'
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(x, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+' multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MyMatch Training')
    parser.add_argument('--path', type=str, default='',
                        help='csv_path')
    parser.add_argument('--name', type=str, default='',
                        help='model_name')
    args = parser.parse_args()
    args.path = ['our_roc.csv']
    # args.path = ['our_roc.csv', 'resnest_roc.csv']
    # args.name = 'FreeMatch'
    read_csv(args.path)





