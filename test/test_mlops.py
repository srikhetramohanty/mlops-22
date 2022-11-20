import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
#pd.set_option('display.max_rows', None)
from skimage.transform import rescale, resize, downscale_local_mean
import warnings
warnings.filterwarnings("ignore")
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utils

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pytest



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------



def test_random_state_same():
    train_frac = 0.8

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train_1,y_train_1,X_test_1,y_test_1,X_val_1,y_val_1 = utils.generate_random_splits(data=data,digits=digits,\
                                                        train_frac=train_frac,seed=100)

    X_train_2,y_train_2,X_test_2,y_test_2,X_val_2,y_val_2 = utils.generate_random_splits(data=data,digits=digits,\
                                                        train_frac=train_frac,seed=100)
                                                        

    assert np.sum(X_train_1)==np.sum(X_train_2), "X-Train datasets not maching"
    # assert y_train_1==y_train_2, "y-Train datasets not maching"
    # assert X_test_1==X_test_2, "X-Test datasets not maching"
    # assert y_test_1==y_test_2, "y-test datasets not maching"


def test_random_state_different():
    train_frac = 0.8

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train_1,y_train_1,X_test_1,y_test_1,X_val_1,y_val_1 = utils.generate_random_splits(data=data,digits=digits,\
                                                        train_frac=train_frac,seed=100)

    X_train_2,y_train_2,X_test_2,y_test_2,X_val_2,y_val_2 = utils.generate_random_splits(data=data,digits=digits,\
                                                        train_frac=train_frac,seed=250)
                                                        

    assert np.sum(X_train_1)==np.sum(X_train_2), "X-Train datasets not maching"
    # assert y_train_1==y_train_2, "y-Train datasets not maching"
    # assert X_test_1==X_test_2, "X-Test datasets not maching"
    # assert y_test_1==y_test_2, "y-test datasets not maching"
