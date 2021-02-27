import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_data(whole_data_txt, save_dir, fold_num=5):
    whole_data = np.loadtxt(whole_data_txt, delimiter=',', dtype=str)
    fatigue_data = whole_data[:, 3].astype(np.float32)
    fatigue_data = (fatigue_data - 1) / 4
    bins = np.arange(0, 1, 0.1)
    bins = np.append(bins, 1.1)
    fatigue_class = pd.cut(fatigue_data, bins, right=False, labels=np.arange(10))
    fatigue_class = fatigue_class.astype(np.int)

    sfolder = StratifiedKFold(n_splits=fold_num, random_state=0, shuffle=True)

    train_fold = []
    eval_fold = []
    for kfold, (train_subfold, eval_subfold) in enumerate(sfolder.split(whole_data, fatigue_class)):
        train_fold.append(whole_data[train_subfold, :])
        eval_fold.append(whole_data[eval_subfold, :])

    for i in range(fold_num):
        np.savetxt(os.path.join(save_dir, 'Data-fold{}-Train.txt'.format(i)), train_fold[i], fmt='%s')
        np.savetxt(os.path.join(save_dir, 'Data-fold{}-Eval.txt'.format(i)), eval_fold[i], fmt='%s')



#===========================================================================================

whole_data_txt = r'C:/Users/admin/Desktop/Data-sum.txt'
save_dir = 'C:/Users/admin/Desktop/tmp'
stratified_kfold_data(whole_data_txt, save_dir)
