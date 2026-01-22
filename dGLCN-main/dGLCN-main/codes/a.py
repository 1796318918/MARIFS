from sklearn.metrics import confusion_matrix


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)  # 灵敏度
    specificity = tn / (tn + fp)  # 特异性

    return sensitivity, specificity


# 示例数据
y_true = [1, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0]

# 计算灵敏度和特异性
sensitivity, specificity = sensitivity_specificity(y_true, y_pred)

# 打印结果
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')