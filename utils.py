import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def get_dichotomous(data):
    '''Use length < 3, to find dichotomous variables'''

    binaries = []
    for k, v in data.items():
        u = [x for x in v.unique() if pd.notnull(x)]
        if len(u) < 3:
            binaries.append(k)
    return binaries


# metrics dict
cmetrics = {}
cmetrics['accuracy_score'] = accuracy_score
cmetrics['f1_score'] = f1_score
cmetrics['precision_score'] = precision_score
cmetrics['recall_score'] = recall_score


def roc_wrapper(ytest, ypred, yprob):
    '''Convenience Function'''
    
    ometrics = {}
    # plot metrics
    for k, v in cmetrics.items():
        metric = v(ytest, ypred)
        ometrics[k] = metric
        print(f'{k:16} {metric:.3f}')

    auc = roc_auc_score(ytest, yprob[:, 1])
    ometrics['auc'] = auc
    print(f'{"auc":16} {auc:.3f}')

    # calculate the ROC values
    fpr, tpr, thresholds = roc_curve(ytest, yprob[:,1])

    # plot ROC curve
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(fpr, tpr, label='tpr')
    axs[0].plot([0,1], [0,1])
    axs[0].set_aspect('equal')

    # probabilities the model predicts a 1
    sns.histplot(yprob[:, 1], ax=axs[1]);

    return ometrics



def confusion_matrix_wrapper(ytest, ypred):
    '''Convenience Function'''
    tn, fp, fn, tp = confusion_matrix(ytest, ypred, normalize='all').ravel()

    fig, ax = plt.subplots()
    sns.heatmap([[tp, fp],[fn, tn]], cmap='Blues', vmax=1, annot=True, xticklabels=[1, 0], yticklabels=[1, 0], ax=ax);

    ax.xaxis.tick_top();
    ax.xaxis.set_label_position('top');
    ax.set_xlabel('Actual');
    ax.set_ylabel('Predicted');