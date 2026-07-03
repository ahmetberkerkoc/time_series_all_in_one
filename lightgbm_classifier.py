import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.utils import shuffle as shuffler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import PredefinedSplit
import numpy as np




def undotransform(preds,y_train,fores,y_test,val_preds=None,log=False,diff=False,cube_root=False):
    y_train = pd.Series(y_train)
    y_test  = pd.Series(y_test)
    preds = pd.Series(preds)
    fores = pd.Series(fores)
    if val_preds is not None:
        val_preds = pd.Series(val_preds)
    if log:
        if diff:
            # exp first; then mul with 1+y_{t-1}; then subtract 1
            preds = np.exp(preds)
            preds *= y_train.shift(fill_value=0).add(1)
            preds -= 1

            fores = np.exp(fores)
            fores *= y_test.shift(fill_value=y_train.iat[-1]).add(1)
            fores -= 1

            if val_preds is not None:
                val_preds = np.exp(val_preds)
                val_preds *= y_train.shift(fill_value=0).add(1)
                val_preds -= 1

        else:
            preds = np.exp(preds) - 1
            fores = np.exp(fores) - 1

            if val_preds is not None:
                val_preds = np.exp(val_preds) - 1
    elif diff:
        # reinstate predictions
        preds += y_train.shift(fill_value=0)
        fores += y_test.shift(fill_value=y_train.iat[-1])
        if val_preds is not None:
            val_preds += y_train.shift(fill_value=0)
    elif cube_root:
        preds = np.power(preds,3)
        fores = np.power(fores,3)
        if val_preds is not None:
            val_preds = np.power(val_preds, 3)
    return preds, fores, val_preds
def search_params(X_train,
                  y_train,
                  X_test,
                  y_test,
                  searching_params,
                  non_transform_y_train,
                  non_transform_y_test,
                  transformation_dic,
                  exp_name,
                  validation_tech,
                  ):
    shuffle = False


    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    model = lgb.LGBMClassifier()

    xv_cls = RandomizedSearchCV
    score_func = 'f1' # accuracy, balanced_accuracy, precision, recall


    if validation_tech == "hold_out":
        val_size = len(y_test)
        train_val_indexes = np.zeros_like(y_train)
        train_val_indexes[:-val_size] = -1
        fold_size = PredefinedSplit(test_fold=train_val_indexes)
    else: # cross validation with k fold
        fold_size = 5


    if xv_cls == GridSearchCV:
        xv = xv_cls(estimator=model, param_grid=searching_params, scoring=score_func, n_jobs=-1, cv=fold_size, verbose=-1, refit=False)
    elif xv_cls == RandomizedSearchCV:
        xv = xv_cls(estimator=model,param_distributions=searching_params, n_iter=10000,scoring=score_func, n_jobs=-1, cv=fold_size, verbose=-1, refit=False)
    xv.fit(X_train, y_train)

    best_params = xv.best_params_

    best_model = type(model)(**best_params).fit(X_train,y_train)
    preds = best_model.predict(X_train)
    fores = best_model.predict(X_test)
    val_preds = cross_val_predict(best_model,X_train,y_train,cv=5)

    X_train = X_train_orig.copy()
    y_train = y_train_orig.copy()

    preds, fores, val_preds = undotransform(preds,non_transform_y_train,fores,non_transform_y_test,val_preds,log=transformation_dic["log"],diff=transformation_dic["diff"],cube_root=transformation_dic["cube_root"])

    accuracy_score_train = accuracy_score(non_transform_y_train, preds)
    f1_score_train = f1_score(non_transform_y_train, preds)
    precision_score_train = precision_score(non_transform_y_train, preds)
    recall_score_train = recall_score(non_transform_y_train, preds)

    accuracy_score_val = accuracy_score(non_transform_y_train, val_preds)
    f1_score_val = f1_score(non_transform_y_train, val_preds)
    precision_score_val = precision_score(non_transform_y_train, val_preds)
    recall_score_val = recall_score(non_transform_y_train, val_preds)

    accuracy_score_test = accuracy_score(non_transform_y_test, fores)
    f1_score_test = f1_score(non_transform_y_test, fores)
    precision_score_test = precision_score(non_transform_y_test, fores)
    recall_score_test = recall_score(non_transform_y_test, fores)

    naive_accuracy_score_train = accuracy_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_f1_score_train = f1_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_precision_score_train = precision_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_recall_score_train = recall_score(non_transform_y_train[1:], non_transform_y_train[:-1])

    naive_accuracy_score_test = accuracy_score(non_transform_y_test, np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_f1_score_test = f1_score(non_transform_y_test, np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_precision_score_test = precision_score(non_transform_y_test, np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_recall_score_test = recall_score(non_transform_y_test, np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))


    plt.figure(figsize=(15, 5))
    plt.plot(non_transform_y_train, label='label')
    plt.plot(preds, label='preds')
    plt.legend()
    plt.title(f"accuracy_score: {accuracy_score_train}, f1_score: {f1_score_train}\n precision_score: {precision_score_train}, recall_score: {recall_score_train}")
    plt.savefig(f'training_preds_vs_label_{exp_name}_center.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(non_transform_y_train, label='label')
    plt.plot(val_preds, label='preds')
    plt.legend()
    plt.title(f"accuracy_score: {accuracy_score_val}, f1_score: {f1_score_val}\n precision_score: {precision_score_val}, recall_score: {recall_score_val}")
    plt.savefig(f'{fold_size}_fold_validation_preds_vs_label{exp_name}_center.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.plot(non_transform_y_test, label='label')
    plt.plot(fores, label='preds')
    plt.legend()
    plt.title(f"accuracy_score: {accuracy_score_test}, f1_score: {f1_score_test}\n precision_score: {precision_score_test}, recall_score: {recall_score_test}")
    plt.savefig(f'test_fores_vs_label_{exp_name}_center.png')
    plt.close()

    result_dic = {
        "best_params": best_params,
        "model": best_model,
        "scores": {
            "accuracy_score_train":accuracy_score_train,
            "f1_score_train":f1_score_train,
            "precision_score_train": precision_score_train,
            "recall_score_train": recall_score_train,


            "accuracy_score_val": accuracy_score_val,
            "f1_score_val": f1_score_val,
            "precision_score_val": precision_score_val,
            "recall_score_val": recall_score_val,

            "accuracy_score_test": accuracy_score_test,
            "f1_score_test": f1_score_test,
            "precision_score_test": precision_score_test,
            "recall_score_test": recall_score_test,

            "naive_accuracy_score_train": naive_accuracy_score_train,
            "naive_f1_score_train": naive_f1_score_train,
            "naive_precision_score_train": naive_precision_score_train,
            "naive_recall_score_train": naive_recall_score_train,

            "naive_accuracy_score_test": naive_accuracy_score_test,
            "naive_f1_score_test": naive_f1_score_test,
            "naive_precision_score_test": naive_precision_score_test,
            "naive_recall_score_test": naive_recall_score_test,
        },
        "preds":preds,
        "fores":fores,
        "val_preds":val_preds,
        "X_train":X_train,
        "X_test":X_test,
        "y_train":non_transform_y_train,
        "y_test": non_transform_y_test,

    }

    pd.to_pickle(result_dic,f"lgbm_result_{exp_name}_center_regression.pkl")

    return result_dic


def fit_predict(X_train,
               y_train,
               X_test,
               y_test,
               non_transform_y_train,
               non_transform_y_test,
               transformation_dic,
               exp_name,
               best_params
                  ):



    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    model = lgb.LGBMClassifier()


    best_model = type(model)(**best_params).fit(X_train,y_train)
    preds = best_model.predict(X_train)
    fores = best_model.predict(X_test)
    val_preds = None #cross_val_predict(best_model,X_train,y_train,cv=5)

    X_train = X_train_orig.copy()
    y_train = y_train_orig.copy()

    preds, fores, val_preds = undotransform(preds,non_transform_y_train,fores,non_transform_y_test,val_preds,log=transformation_dic["log"],diff=transformation_dic["diff"],cube_root=transformation_dic["cube_root"])



    accuracy_score_train = accuracy_score(non_transform_y_train, preds)
    f1_score_train = f1_score(non_transform_y_train, preds)
    precision_score_train = precision_score(non_transform_y_train, preds)
    recall_score_train = recall_score(non_transform_y_train, preds)

    accuracy_score_val = accuracy_score(non_transform_y_train, val_preds) if val_preds is not None else None
    f1_score_val = f1_score(non_transform_y_train, val_preds) if val_preds is not None else None
    precision_score_val = precision_score(non_transform_y_train, val_preds) if val_preds is not None else None
    recall_score_val = recall_score(non_transform_y_train, val_preds) if val_preds is not None else None

    accuracy_score_test = accuracy_score(non_transform_y_test, fores)
    f1_score_test = f1_score(non_transform_y_test, fores)
    precision_score_test = precision_score(non_transform_y_test, fores)
    recall_score_test = recall_score(non_transform_y_test, fores)






    naive_accuracy_score_train = accuracy_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_f1_score_train = f1_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_precision_score_train = precision_score(non_transform_y_train[1:], non_transform_y_train[:-1])
    naive_recall_score_train = recall_score(non_transform_y_train[1:], non_transform_y_train[:-1])

    naive_accuracy_score_test = accuracy_score(non_transform_y_test,
                                               np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_f1_score_test = f1_score(non_transform_y_test,
                                   np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_precision_score_test = precision_score(non_transform_y_test, np.concatenate(
        [[non_transform_y_train[-1]], non_transform_y_test[:-1]]))
    naive_recall_score_test = recall_score(non_transform_y_test,
                                           np.concatenate([[non_transform_y_train[-1]], non_transform_y_test[:-1]]))



    plt.figure(figsize=(15, 5))
    plt.plot(non_transform_y_train, label='label')
    plt.plot(preds, label='preds')
    plt.legend()
    plt.title(f"accuracy_score: {accuracy_score_train}, f1_score: {f1_score_train}\n precision_score: {precision_score_train}, recall_score: {recall_score_train}")
    plt.savefig(f'training_preds_vs_label_{exp_name}.png')
    plt.close()


    plt.figure(figsize=(15, 5))
    plt.plot(non_transform_y_test, label='label')
    plt.plot(fores, label='preds')
    plt.legend()
    plt.title(f"accuracy_score: {accuracy_score_test}, f1_score: {f1_score_test}")
    plt.savefig(f'test_fores_vs_label_{exp_name}.png')
    plt.close()

    result_dic = {
        "best_params": best_params,
        "model": best_model,
        "scores": {
            "accuracy_score_train": accuracy_score_train,
            "f1_score_train": f1_score_train,
            "precision_score_train": precision_score_train,
            "recall_score_train": recall_score_train,

            "accuracy_score_val": accuracy_score_val,
            "f1_score_val": f1_score_val,
            "precision_score_val": precision_score_val,
            "recall_score_val": recall_score_val,

            "accuracy_score_test": accuracy_score_test,
            "f1_score_test": f1_score_test,
            "precision_score_test": precision_score_test,
            "recall_score_test": recall_score_test,

            "naive_accuracy_score_train": naive_accuracy_score_train,
            "naive_f1_score_train": naive_f1_score_train,
            "naive_precision_score_train": naive_precision_score_train,
            "naive_recall_score_train": naive_recall_score_train,

            "naive_accuracy_score_test": naive_accuracy_score_test,
            "naive_f1_score_test": naive_f1_score_test,
            "naive_precision_score_test": naive_precision_score_test,
            "naive_recall_score_test": naive_recall_score_test,
        },
        "preds":preds,
        "fores":fores,
        "val_preds":val_preds,
        "X_train":X_train,
        "X_test":X_test,
        "y_train":non_transform_y_train,
        "y_test": non_transform_y_test,

    }

    pd.to_pickle(result_dic,f"lgbm_all_results_{exp_name}_regression.pkl")

    return result_dic