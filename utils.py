import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', None)
from skimage.transform import rescale, resize, downscale_local_mean
import warnings
from sklearn import tree
warnings.filterwarnings("ignore")

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def plot_(X_test,predicted,digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(digits.images[0].shape[0], digits.images[0].shape[1])
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


def svm_hyper_params_generator():
    # GAMMA = 0.001
    # C = 0.5
    GAMMA_list = [0.01,0.005,0.001,0.0005]
    C_list = [0.05,0.1,0.2,0.5,0.7,0.9,1,2,3]
    # GAMMA_list = [0.01]
    # C_list = [0.05,0.1]

    combination_tray = []

    for gamma_ in GAMMA_list:
        for c_val in C_list:
            combination_tray.append([gamma_,c_val])

    print("Total Combinations of hyper-parms available : ",len(combination_tray))

    return combination_tray

def decision_tree_hyper_params_generator():
    # max_depth_list = [2,4]
    # criterion_list = ["gini","entropy"]
    max_depth_list = [2,3,4,5,6,7,8]
    criterion_list = ["gini","entropy"]

    combination_tray = []

    for depth_ in max_depth_list:
        for criterion_ in criterion_list:
            combination_tray.append([depth_,criterion_])

    print("Total Combinations of hyper-parms available : ",len(combination_tray))

    return combination_tray


def model(model_req):
    if model_req=="svm":
        print("Model loaded : svm...")
        return svm.SVC()
    elif model_req=="decision_tree":
        print("Model loaded : decision_tree...")
        return tree.DecisionTreeClassifier()


def hyper_param_tuning(model_required,X_train,\
                        y_train,X_val,y_val,X_test,y_test):
    
    accuracy_df_all_comb = pd.DataFrame()
    best_val_acc = 0

    if model_required=="svm":
        combination_tray = svm_hyper_params_generator()
    elif model_required=="decision_tree":
        combination_tray = decision_tree_hyper_params_generator()

    for config in combination_tray:

        clf = model(model_req=model_required)

        #------------------------------------------------
        #PART: setting up hyperparameter
        if model_required=="svm":
            hyper_params = {'gamma':config[0], 'C':config[1]}
        elif model_required=="decision_tree":
            hyper_params = {'max_depth':config[0], 'criterion':config[1]}

        clf.set_params(**hyper_params)
        #------------------------------------------------

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        #----------------------------------------------------------
        predicted_val = clf.predict(X_val)
        accuracy_val = metrics.accuracy_score(y_val,predicted_val) 

        predicted_test = clf.predict(X_test)
        accuracy_test = metrics.accuracy_score(y_test,predicted_test) 

        predicted_train = clf.predict(X_train)
        accuracy_train = metrics.accuracy_score(y_train,predicted_train) 

        #----------------------------------------------------------
        hyper_params["train accuracy"] = accuracy_train
        hyper_params["test accuracy"] = accuracy_test
        hyper_params["dev accuracy"] = accuracy_val

        #print(hyper_params)
        temp_df = pd.DataFrame(list(hyper_params.values())).reset_index(drop=True).T
        #print(temp_df)
        temp_df.columns = list(hyper_params.keys())
        accuracy_df_all_comb = accuracy_df_all_comb.append(temp_df)

        if accuracy_val > best_val_acc:
            best_model = clf
            best_val_acc = accuracy_val.copy()
            best_combination_df = temp_df.copy() 
        else:
            pass

    return best_model,accuracy_df_all_comb,best_combination_df


def generate_random_splits(data,digits,train_frac):

    # Split data into train and rest subsets
    X_train, X_eval, y_train, y_eval = train_test_split(
        data, digits.target, train_size=train_frac, shuffle=True)

    # Split rest data into 50% test and 50% val subsets
    X_test, X_val, y_test, y_val = train_test_split(
        X_eval, y_eval, test_size=0.5, shuffle=True)

    return X_train,y_train,X_test,y_test,X_val,y_val