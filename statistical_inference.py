import pandas as pd
import os
from scipy import stats
import scikit_posthocs as sp

res_df = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'roc_auc'])

dirname = 'Data'
for filename in os.listdir(dirname):
    filename_, file_extension = os.path.splitext(filename)
    if file_extension == ".xlsx":
        data = pd.read_excel(dirname+'\\'+filename,usecols=['dataset_name', 'algorithm_name', 'roc_auc'])
    else:
        data = pd.read_csv(dirname+'\\'+filename, usecols=['dataset_name', 'algorithm_name', 'roc_auc'])
    # average auc for each dataset
    avg_auc = data.groupby(['dataset_name', 'algorithm_name'], as_index= False).mean()
    res_df = res_df.append(avg_auc)
res_df.reset_index(inplace=True, drop=True)
# get all datasets names with results from all four algorithms
dataset_names = res_df.groupby('dataset_name', as_index = False).count()
dataset_names = list(dataset_names[dataset_names['algorithm_name'] == 4]['dataset_name'])
# filter result to contain only datasets that are in dataset_names
res_df = res_df[res_df['dataset_name'].apply(lambda x: x in dataset_names)]
alog_names = list(res_df['algorithm_name'].unique())

# Friedman test
stat, p = stats.friedmanchisquare(res_df[res_df['algorithm_name'] == alog_names[0]].sort_values(by='dataset_name')['roc_auc'],
                              res_df[res_df['algorithm_name'] == alog_names[1]].sort_values(by='dataset_name')['roc_auc'],
                              res_df[res_df['algorithm_name'] == alog_names[2]].sort_values(by='dataset_name')['roc_auc'],
                              res_df[res_df['algorithm_name'] == alog_names[3]].sort_values(by='dataset_name')['roc_auc'])

# interpret results
alpha = 0.05
print('Statistics=%.3f, p=%.3f' % (stat, p))
if(p < alpha):
    print('null hypothesis rejected')
    # perform post-hoc test
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(sp.posthoc_nemenyi_friedman(res_df, y_col='roc_auc',block_col='dataset_name', group_col='algorithm_name',melted=True))
