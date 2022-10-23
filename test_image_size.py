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
from skimage.transform import rescale, resize, downscale_local_mean
#pd.set_option('display.max_rows', None)

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
GAMMA = 0.001
C = 0.5
GAMMA_list = [0.01,0.005,0.001,0.0005,0.0001,0.00005]
C_list = [0.05,0.1,0.2,0.5,0.7,0.9,1,2,3,5,6,7,10,20]

combination_tray = []

for gamma_ in GAMMA_list:
    for c_val in C_list:
        combination_tray.append([gamma_,c_val])

print("Total Combinations of hyper-parms available : ",len(combination_tray))



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#---------------------------------------------------------
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits() 

#--------------------------------------------------------------------------------
print(digits.images.shape)

# Resizing through rescaling using skimage library 
modified_images = []
rescale_factor = 2 #Factor for scaling

for image in digits.images: #Iterating over images
    image_mod = resize(image, (image.shape[0] * rescale_factor, image.shape[1] * rescale_factor), anti_aliasing=True) #Rescaling image
    modified_images.append(image_mod)

digits.images = np.array(modified_images) #Passigng into array

print("Shape of array holding all images : ",digits.images.shape)
#--------------------------------------------------------------------------------


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

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
# print(digits.images.shape)
# test_image = digits.images[0]
# test_image_mod = rescale(digits.images[0], 0.25, anti_aliasing=True)
# modified_images = []
# rescale_factor = 0.25

# for image in digits.images:
#     image_mod = rescale(digits.images[0], rescale_factor, anti_aliasing=True)
#     #modified_images = np.append(modified_images,image_mod,axis=0)
#     modified_images.append(image_mod)

# modified_images = np.array(modified_images)
# print(modified_images.shape)

# plt.imshow(test_image_mod)
# plt.show()


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

# #val_accuracy_tray = []
# # val_accuracy_df = pd.DataFrame()
# # val_accuracy_df.columns = ["Hyperparam-combination","train accuracy","test accuracy","val accuracy"]
# accuracy_df_all_comb = pd.DataFrame()
# best_val_acc = 0

# for config in combination_tray:

#     #temp_df = pd.DataFrame()
#     #print("---- Configuration : ",config)
#     # Create a classifier: a support vector classifier
#     clf = svm.SVC()

#     #------------------------------------------------
#     #PART: setting up hyperparameter
#     hyper_params = {'gamma':config[0], 'C':config[1]}
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
#     hyper_params["val accuracy"] = accuracy_val

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

#     #print(temp_df)

#     #val_accuracy_tray.append(accuracy_val)

#     #print("---- Configuration : ",config, " --- Accuracy : ",accuracy_val)
# ###############################################################################
# print("--- All Combinations ---\n")
# print(accuracy_df_all_comb)

# print("--- Best Combinations ---\n")
# print(best_combination_df)

# predicted = best_model.predict(X_val)

# ###############################################################################
# # Below we visualize the first 4 test samples and show their predicted
# # digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# ###############################################################################
# # :func:`~sklearn.metrics.classification_report` builds a text report showing
# # the main classification metrics.

# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
