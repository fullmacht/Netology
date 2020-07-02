import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix


def calculate_tpr_and_fpr(y_true, y_pred_probas, alpha):
    list = []
    print(y_pred_probas)
    for j in y_pred_probas:
        if j >= alpha:
            j = 1
            list.append(j)
        else:
            j = 0
            list.append(j)
    tn, fp, fn, tp = confusion_matrix(y_true, list).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return tpr, fpr


def roc_auc_plot(y_true, y_pred_probas):
    alphas = np.arange(0, 1, 0.001)
    tprs, fprs = [], []
    for a in alphas:
        tpr, fpr = calculate_tpr_and_fpr(y_true, y_pred_probas, a)
        tprs.append(tpr)
        fprs.append(fpr)

    plt.plot(fprs, tprs)
    plt.show()

roc_auc_plot(y_test,Y_pred_proba[: ,1])



roc_auc_plot(yval, yval_pred_proba[:, 1])