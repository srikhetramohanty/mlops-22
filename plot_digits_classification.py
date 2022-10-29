"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', None)
from skimage.transform import rescale, resize, downscale_local_mean
import warnings
warnings.filterwarnings("ignore")
import utils

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#

#---------------------------------------------------------
# GAMMA = 0.001
# C = 0.5
# # GAMMA_list = [0.01,0.005,0.001,0.0005]
# # C_list = [0.05,0.1,0.2,0.5,0.7,0.9,1,2,3]
# GAMMA_list = [0.01]
# C_list = [0.05,0.1]

# combination_tray = []

# for gamma_ in GAMMA_list:
#     for c_val in C_list:
#         combination_tray.append([gamma_,c_val])

# print("Total Combinations of hyper-parms available : ",len(combination_tray))

#model_required = "svm"

# if model_required=="svm":
#     combination_tray = utils.svm_hyper_params_generator()
# elif model_required=="decision_tree":
#     combination_tray = utils.decision_tree_hyper_params_generator()



train_frac = 0.9
test_frac = 0.1
dev_frac = 0.1

#---------------------------------------------------------
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

#--------------------------------------------------------------------------------
print(digits.images.shape)
#--------------------------------------------------------------------------------



###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images

print('Shape of an image : ',digits.images[0].shape)


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# # Split data into train and rest subsets
# X_train, X_eval, y_train, y_eval = train_test_split(
#     data, digits.target, train_size=train_frac, shuffle=False,
# random_state=100)

# # Split rest data into 50% test and 50% val subsets
# X_test, X_val, y_test, y_val = train_test_split(
#     X_eval, y_eval, test_size=0.5, shuffle=False,
# random_state=100)

#X_train,y_train,X_test,y_test,X_val,y_val = utils.generate_random_splits(data=data,digits=digits,train_frac=train_frac)

#val_accuracy_tray = []
# val_accuracy_df = pd.DataFrame()
# val_accuracy_df.columns = ["Hyperparam-combination","train accuracy","test accuracy","val accuracy"]

#best_model,accuracy_df_all_comb,best_combination_df = utils.hyper_param_tuning(model_required,combination_tray,\
#                                                                                X_train,y_train,X_val,y_val,X_test,y_test)

all_iteration_list = []

for idx in range(5):
    
    temp_list = [str(idx)]
    
    print('\n# iteration : {} -------------------------------------------------------------'.format(idx))
    X_train,y_train,X_test,y_test,X_val,y_val = utils.generate_random_splits(data=data,digits=digits,train_frac=train_frac)

    best_model,accuracy_df_all_comb,best_combination_df = utils.hyper_param_tuning("svm",\
                                                                                X_train,y_train,X_val,y_val,X_test,y_test)
    print(best_combination_df)
    dev_acc_svm = float(best_combination_df["dev accuracy"])
    temp_list.append(dev_acc_svm)

    best_model,accuracy_df_all_comb,best_combination_df = utils.hyper_param_tuning("decision_tree",\
                                                                                X_train,y_train,X_val,y_val,X_test,y_test)
    print(best_combination_df)
    dev_acc_dt = float(best_combination_df["dev accuracy"])
    temp_list.append(dev_acc_dt)

    all_iteration_list.append(temp_list)

all_iteration_list_df = pd.DataFrame(all_iteration_list)
all_iteration_list_df.columns = ["run","svm","decision tree"]

summary_mean = ["mean",all_iteration_list_df["svm"].mean(),all_iteration_list_df["decision tree"].mean()]
summary_std = ["std",all_iteration_list_df["svm"].std(),all_iteration_list_df["decision tree"].std()]

summary = [summary_mean,summary_std]

summary_df = pd.DataFrame(summary)
summary_df.columns = ["run","svm","decision tree"]

all_iteration_list_df = all_iteration_list_df.append(summary_df)

print("#--- results table : \n")
print(all_iteration_list_df)


# accuracy_df_all_comb = pd.DataFrame()
# best_val_acc = 0

# for config in combination_tray:

#     #temp_df = pd.DataFrame()
#     #print("---- Configuration : ",config)
#     # Create a classifier: a support vector classifier
#     #clf = svm.SVC()
#     clf = utils.model(model_req=model_required)

#     #------------------------------------------------
#     #PART: setting up hyperparameter
#     if model_required=="svm":
#         hyper_params = {'gamma':config[0], 'C':config[1]}
#     elif model_required=="decision_tree":
#         hyper_params = {'max_depth':config[0], 'criterion':config[1]}

#     clf.set_params(**hyper_params)
#     #------------------------------------------------

#     # Learn the digits on the train subset
#     clf.fit(X_train, y_train)

#     #----------------------------------------------------------
#     predicted_val = clf.predict(X_val)
#     accuracy_val = metrics.accuracy_score(y_val,predicted_val) 

#     predicted_test = clf.predict(X_test)
#     accuracy_test = metrics.accuracy_score(y_test,predicted_test) 

#     predicted_train = clf.predict(X_train)
#     accuracy_train = metrics.accuracy_score(y_train,predicted_train) 

#     #----------------------------------------------------------
#     hyper_params["train accuracy"] = accuracy_train
#     hyper_params["test accuracy"] = accuracy_test
#     hyper_params["dev accuracy"] = accuracy_val

#     #print(hyper_params)
#     temp_df = pd.DataFrame(list(hyper_params.values())).reset_index(drop=True).T
#     #print(temp_df)
#     temp_df.columns = list(hyper_params.keys())
#     accuracy_df_all_comb = accuracy_df_all_comb.append(temp_df)

#     if accuracy_val > best_val_acc:
#         best_model = clf
#         best_val_acc = accuracy_val.copy()
#         best_combination_df = temp_df.copy() 
#     else:
#         pass

    #print(temp_df)

    #val_accuracy_tray.append(accuracy_val)

    #print("---- Configuration : ",config, " --- Accuracy : ",accuracy_val)
###############################################################################
# print("--- All Combinations ---\n")
# print(accuracy_df_all_comb)
# print(type(accuracy_df_all_comb.describe()))

# print("Statistics of the accuracies : ")
# described_all_comb = accuracy_df_all_comb[["train accuracy","dev accuracy","test accuracy"]].describe().reset_index()
# described_all_comb = described_all_comb[described_all_comb["index"].isin(["mean","50%","min","max"])]
# described_all_comb["index"] = np.where(described_all_comb["index"]=="50%","median",described_all_comb["index"])
# print(described_all_comb)

# print("--- Best Combinations ---\n")
# print(best_combination_df)



predicted = best_model.predict(X_val)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

#utils.plot_(X_test,predicted,digits)
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(digits.images[0].shape[0], digits.images[0].shape[1])
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")
#     plt.show()
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {best_model}:\n"
    f"{metrics.classification_report(y_val, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
