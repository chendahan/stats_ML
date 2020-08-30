from time import time
import shap
from matplotlib import pyplot
from scipy.stats import uniform, randint
from sklearn import metrics
import xgboost as xgb
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings('always')

# model params- used for hyperparameter tuning
model_params = {
    'learning_rate': uniform(0.1, 0.5),
    'max_depth': randint(2, 6),
    'n_estimators': randint(10, 200),
    "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
    'subsample': uniform(0.6, 0.4)
}


def binaryStat(gbm, X_test, y_test, y_pred):
    """
    calculate stats for binary classification
    :param gbm: gbm classifier
    :param X_test: test data
    :param y_test: test classification
    :param y_pred: predicted classification using auggbmboost_m
    :return: accuracy, TPR, FPR, precision, roc_auc, PR_curve
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    y_prob = gbm.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob[:, 1])
    except ValueError as inst:
        print(inst)
        roc_auc = None
    # calculate TPR & FPR using the confusion_matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    TPR = 0 if (TP + FN) == 0 else TP / (TP + FN)
    FPR = 0 if (FP + TN) == 0 else FP / (FP + TN)
    _precision, _recall, thresholds = metrics.precision_recall_curve(y_test, y_prob[:,1], )
    PR_curve = metrics.auc(_recall, _precision)

    return accuracy, TPR, FPR, precision, roc_auc, PR_curve


# dirname: dataset directory
# returns joined dataframe
def join_results(dirname):
    # preprocess- read results and join to one table
    res_df = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'roc_auc'])
    # for each results dataset
    for filename in os.listdir(dirname):
        filename_, file_extension = os.path.splitext(filename)
        if file_extension == ".xlsx":
            data = pd.read_excel(dirname+'\\'+filename,usecols=['dataset_name', 'algorithm_name', 'roc_auc'])
        else:
            data = pd.read_csv(dirname+'\\'+filename, usecols=['dataset_name', 'algorithm_name', 'roc_auc'])
        # average auc for each dataset
        avg_auc = data.groupby(['dataset_name', 'algorithm_name'], as_index= False).mean()
        res_df = res_df.append(avg_auc)
    # get all datasets names with results from all four algorithms
    dataset_names = res_df.groupby('dataset_name', as_index = False).count()
    dataset_names = list(dataset_names[dataset_names['algorithm_name'] == 4]['dataset_name'])
    # filter result to contain only datasets that are in dataset_names
    res_df = res_df[res_df['dataset_name'].apply(lambda x: x in dataset_names)]
    # sort by dataset name
    res_df = res_df.sort_values('dataset_name')
    res_df.reset_index(inplace=True, drop=True)

    return res_df

def mark_success(res_df):
    # for each dataset mark winner as 1
    winner_col = []
    for i in range(0, len(res_df), 4):
        max_val = max(res_df['roc_auc'][i:i + 4])
        for j in range(i, i + 4):
            # in case more than one algorithm got the max auc, all of them considered as 1
            if res_df.iloc[j, res_df.columns.get_loc('roc_auc')] < max_val:
                winner_col.append(0)
            else:
                winner_col.append(1)
    res_df['target'] = winner_col

def plot_importance(xgb_model, imp_type):
    imp = xgb_model.get_booster().get_score(importance_type=imp_type)
    keys = list(imp.keys())
    values = list(imp.values())
    results = pd.DataFrame(data=values[:30], index=keys[:30], columns=["score"]).sort_values(by="score", ascending=False)
    results.plot(kind='barh',title=imp_type)



dirname='./data'
res_df = join_results(dirname) #join results files
mark_success(res_df) # create 0/1 col
#change column to match ClassificationAllMetaFeatures dataframe
res_df['dataset_name']=res_df['dataset_name'].apply(lambda x: x.split('.csv')[0])
res_df.drop('roc_auc', axis=1,inplace=True)
# read ClassificationAllMetaFeatures.csv
meta_learner_data = pd.read_csv('ClassificationAllMetaFeatures.csv')
meta_learner_data = res_df.merge(meta_learner_data, right_on='dataset', left_on='dataset_name', how='inner')
meta_learner_data = meta_learner_data.sort_values('dataset_name')
#pre-process
enc = LabelEncoder()
meta_learner_data['algorithm_name'] = enc.fit_transform(meta_learner_data['algorithm_name'].astype(str))
meta_learner_data.drop(['dataset'], axis=1, inplace=True)
# results df
df_redults = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'accuracy',
                          'TPR', 'FPR', 'precision', 'roc_auc', 'PR_curve', 'training_time',
                          'inference_time'])
# parameter tuning
X_train_all = meta_learner_data.drop(['dataset_name', 'target'], axis=1)
y_train_all = meta_learner_data['target']
xgb_model = xgb.XGBClassifier()
rs_model = RandomizedSearchCV(xgb_model, model_params, n_iter=50, random_state=10).fit(X_train_all, y_train_all)


# xgboost - leave one out - the dataset is sorted according to dataset name, so can loop the data frame in order
df_redults.to_csv('xgb_results.csv')
for i in range(0, len(meta_learner_data), 4):
    # leave one out
    X_train= meta_learner_data[~meta_learner_data.index.isin(range(i, i+4))].drop(['dataset_name','target'], axis=1)
    y_train= meta_learner_data[~meta_learner_data.index.isin(range(i, i+4))]['target']
    X_test=meta_learner_data.iloc[i:i+4].drop('target', axis=1)
    y_test= meta_learner_data.iloc[i:i+4]['target']
    filename=X_test.iloc[0,X_test.columns.get_loc('dataset_name')] # get tested file
    X_test=X_test.drop('dataset_name',axis=1)
    xgb_model = xgb.XGBClassifier(random_state=0)
    xgb_model.set_params(**rs_model.best_params_)
    # get training time
    start = time()
    xgb_model.fit(X_train,y_train)
    training_time = time() - start
    # get inference time
    start = time()
    y_pred = xgb_model.predict(X_test)
    inference_time = (time() - start) / len(X_test)  # single inference time
    inference_time *= 1000  # 1000 lines
    accuracy, TPR, FPR, precision, roc_auc, PR_curve = binaryStat(xgb_model, X_test, y_test, y_pred)
    df_redults = df_redults.append({'dataset_name': filename, 'algorithm_name': 'Xgboost',
                                    'accuracy': accuracy, 'TPR': TPR, 'FPR': FPR, 'precision': precision,
                                    'roc_auc': roc_auc, 'PR_curve': PR_curve, 'training_time': training_time,
                                    'inference_time': inference_time}, ignore_index=True)

# write df_results to file
df_redults.to_csv('xgb_results.csv', mode='a', header=False)
xgb_model = xgb.XGBClassifier(random_state=0).set_params(**rs_model.best_params_)
xgb_model.fit(X_train_all, y_train_all)
# plot feature_importances_
pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
pyplot.title("feature_importances_")
pyplot.show()
# # cover
plot_importance(xgb_model,'cover')
# # gain
plot_importance(xgb_model,'gain')
#weight
plot_importance(xgb_model,'weight')
# #shap
explainerModel = shap.TreeExplainer(xgb_model)
shap_values_Model = explainerModel.shap_values(X_train_all)
# shap summary bar plot
shap.summary_plot(shap_values_Model, X_train_all, plot_type="bar", max_display=10)