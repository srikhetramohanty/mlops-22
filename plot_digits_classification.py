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
train_frac = 0.9
test_frac = 0.1
dev_frac = 0.1

digits = datasets.load_digits()

#--------------------------------------------------------------------------------
print(digits.images.shape)
#--------------------------------------------------------------------------------

print('Shape of an image : ',digits.images[0].shape)


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

##########################################################################################################################################

all_iteration_list = []
perc_agreement_all = []

for idx in range(5):
    
    temp_list = [str(idx)]
    
    print('\n# iteration : {} -------------------------------------------------------------'.format(idx))
    X_train,y_train,X_test,y_test,X_val,y_val = utils.generate_random_splits(data=data,digits=digits,train_frac=train_frac)

    best_model,accuracy_df_all_comb,best_combination_df,predicted_svm = utils.hyper_param_tuning("svm",\
                                                                                X_train,y_train,X_val,y_val,X_test,y_test)
    print(best_combination_df)
    dev_acc_svm = float(best_combination_df["dev accuracy"])
    temp_list.append(dev_acc_svm)

    best_model,accuracy_df_all_comb,best_combination_df,predicted_dt = utils.hyper_param_tuning("decision_tree",\
                                                                                X_train,y_train,X_val,y_val,X_test,y_test)
    print(best_combination_df)
    dev_acc_dt = float(best_combination_df["dev accuracy"])
    temp_list.append(dev_acc_dt)

    #--------------------------------------
    perc_agreement = metrics.accuracy_score(predicted_svm,predicted_dt)
    temp_list.append(perc_agreement)
    #--------------------------------------

    perc_agreement_all.append(perc_agreement)
    all_iteration_list.append(temp_list)

all_iteration_list_df = pd.DataFrame(all_iteration_list)
all_iteration_list_df.columns = ["run","svm","decision tree","perc_agreement_models"]

summary_mean = ["mean",all_iteration_list_df["svm"].mean(),all_iteration_list_df["decision tree"].mean(),all_iteration_list_df["perc_agreement_models"].mean()]
summary_std = ["std",all_iteration_list_df["svm"].std(),all_iteration_list_df["decision tree"].std(),all_iteration_list_df["perc_agreement_models"].std()]

summary = [summary_mean,summary_std]

summary_df = pd.DataFrame(summary)
summary_df.columns = ["run","svm","decision tree","perc_agreement_models"]

all_iteration_list_df = all_iteration_list_df.append(summary_df)

print("#--- results table : \n")
print(all_iteration_list_df.drop(columns=["perc_agreement_models"]))
print("#--- Percetage agreement : \n")
print(all_iteration_list_df)

###############################################################################
predicted = best_model.predict(X_val)

###############################################################################
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