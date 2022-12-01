import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

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


def adaboost_wrapper(params, xtrain, xtest, ytrain, ytest):
    max_depth = params['max_depth']
    min_samples_leaf = params['min_samples_leaf']
    learning_rate = params['learning_rate']
    n_estimators = params['n_estimators']

    if params['problem'] == 'regression':
        base_estimator = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model = AdaBoostRegressor(base_estimator, learning_rate=learning_rate, n_estimators=n_estimators, random_state=0)
    else:
        base_estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model = AdaBoostClassifier(base_estimator, learning_rate=learning_rate, n_estimators=n_estimators, random_state=0)

    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    
    if params['problem'] == 'regression':
        m1 = r2_score(ytest, ypred)
        m2 = mean_squared_error(ytest, ypred)
    else:
        m1 = f1_score(ytest, ypred)
        m2 = accuracy_score(ytest, ypred)        
    return model, m1, m2


# Plot lift
def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col:str, probability_col:str):

    df.sort_values(by=probability_col, ascending=False, inplace=True)

    subset = df[df[predicted_col] == True]

    rows = []
    for group in np.array_split(subset, 10):
        score = accuracy_score(group[actual_col].tolist(),
                                                   group[predicted_col].tolist(),
                                                   normalize=False)

        rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    lift = pd.DataFrame(rows)

    #Cumulative Gains Calculation
    lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
    lift['PercentCorrect'] = lift.apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x['RunningCorrect'], axis=1)

    lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()

    lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
        lambda x: 100 if (100 / lift['NumCorrectPredictions'].sum()) * x > 100 else (100 / lift[
            'NumCorrectPredictions'].sum()) * x)

    lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)

    lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()

    lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x)

    #Lift Chart
    lift['NormalisedPercentAvg'] = 1
    lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / lift['PercentAvgCase']

    return lift


def plot_cumulative_gains(lift: pd.DataFrame):
    fig, ax = plt.subplots()
    fig.canvas.draw()

    xdata = np.array(range(1, len(lift['PercentCorrect']) + 1)) * 10

    handles = []
    handles.append(ax.plot(xdata, lift['PercentCorrect'], 'r-', label='Percent Correct Predictions'))
    handles.append(ax.plot(xdata, lift['PercentCorrectBestCase'], 'g-', label='Best Case'))
    handles.append(ax.plot(xdata, lift['PercentAvgCase'], 'b-', label='Average Case'))

    ax.set_xlabel('Total Population (%)')
    ax.set_ylabel('Number of Respondents (%)')

    #ax.set_xlim([0, 9])
    ax.set_ylim([0, 100])

    fig.legend(handles, labels=[h[0].get_label() for h in handles])
    fig.show()


def plot_lift_chart(lift: pd.DataFrame):
    xdata = np.array(range(1, len(lift['PercentCorrect']) + 1)) * 10

    plt.figure()
    plt.plot(xdata, lift['NormalisedPercentAvg'], 'r-', label='Normalised \'response rate\' no model')
    plt.plot(xdata, lift['NormalisedPercentWithModel'], 'g-', label='Normalised \'response rate\' using model')
    plt.legend()
    plt.show()