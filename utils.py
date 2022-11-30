import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# models
from sklearn.linear_model import LinearRegression

# metrics classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# metrics regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


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


# metrics regression
def rmse_score(*args, **kwargs):
    return mean_squared_error(*args, **kwargs, squared=False)

rmetrics = {}
rmetrics['r2'] = r2_score
rmetrics['mse'] = mean_squared_error
rmetrics['rmse'] = rmse_score


def linear_regression_wrapper(xtrain, xtest, ytrain, ytest, data, show=True):
    '''
    Convenience function wrapping
    the application of linear regression
    calculation of metrics and
    plotting the results
    '''

    # fit the linear regression model
    lr = LinearRegression()
    lr.fit(xtrain, ytrain)

    # model predictions
    ypred = lr.predict(xtest)

    # print and plot results
    if show:
        print('*** model paramters:')
        print('coeff.: ', ', '.join([f'{x:.3f}' for x in lr.coef_]))
        print(f'inter.: {lr.intercept_:.3f}')
        print()
        print('*** scores:')
        for k, v in rmetrics.items():
            score = v(ytest, ypred)
            print(f'{k:22} {score:.3f}')

        plot = sns.jointplot(data=data, x='square_meter', y='price', marker='.', marginal_kws=dict(bins=25));
        plot.ax_joint.plot(xtest, ypred, '-', color='violet' );
    return ypred