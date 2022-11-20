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
import os
import utils
from joblib import dump
from joblib import load


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import argparse
# Initialize the Parser
parser = argparse.ArgumentParser()
# Adding Arguments
parser.add_argument("--clf_name", help = "Classifier name")
parser.add_argument("--random_state", help = "Set random state")  
args = parser.parse_args()
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
seed = int(args.random_state)
model = args.clf_name

print("# -------- args : \n")
print("Random seed parsed: ",seed)
print("Model parsed: ",model)
n_cv = 1
##########################################################################################################################################

for idx in range(n_cv):
    print("CV Fold :-------------------------- ",idx)
    
    temp_list = [str(idx)]
    
    print('\n# iteration : {} -------------------------------------------------------------'.format(idx))
    X_train,y_train,X_test,y_test,X_val,y_val = utils.generate_random_splits(data=data,digits=digits,\
                                                        train_frac=train_frac,seed=seed)

    best_model,accuracy_df_all_comb,best_combination_df,predicted_svm = utils.hyper_param_tuning(model,\
                                                                                X_train,y_train,X_val,y_val,X_test,y_test)

###############################################################################
#Saving model
save_path_base = os.getcwd()
#print(str(best_combination_df["C"][0]))
if os.path.exists("./models"):
    pass
else:
    os.mkdir("./models")

if model=="svm":
    save_path_partial = "models/"+ model +"_gamma_"+str(best_combination_df["gamma"][0])+"_C_"+str(best_combination_df["C"][0])+".joblib"
elif model=="tree":
    save_path_partial = "models/"+ model +"_max_depth_"+str(best_combination_df["max_depth"][0])+"_Criterion_"+str(best_combination_df["criterion"][0])+".joblib"

save_path_full = os.path.join(save_path_base,save_path_partial)

print(save_path_full)

dump(best_model, save_path_full)
print('#--- Best Model Saved ---#')

#Loading the best model
best_model = load(save_path_full)
print("#--- Best Model loaded ---#")


###############################################################################
predicted = best_model.predict(X_val)
predicted_test = best_model.predict(X_test)


print("Best hyper-param results on valset : \n",best_combination_df)
###############################################################################
print("# -------- args : \n")
print("Random seed parsed: ",seed)
print("Model parsed: ",model)

print(
    f"Classification report for classifier {best_model}:\n"
    f"{metrics.classification_report(y_val, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

if os.path.exists("./results"):
    pass
else:
    os.mkdir("./results")

txt_path_partial = "results/"+ model +"_" + str(seed) + ".txt"
txt_path_full = os.path.join(save_path_base,txt_path_partial)
print(txt_path_full)

# text_file = open(txt_path_full, "w")
# text_file.write('test accuracy: {}'.format(metrics.accuracy_score(y_test,predicted_test)))
# text_file.close()

with open(txt_path_full, "w") as file:
   lines = ['test accuracy: {}\n'.format(metrics.accuracy_score(y_test,predicted_test)),\
            'test macro-f1: {}\n'.format(metrics.f1_score(y_test,predicted_test,average='macro')),\
            'model saved at {}\n'.format("./"+save_path_partial)]
   file.writelines(lines)
   file.close()

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()