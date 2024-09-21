import numpy as np
import pandas as pd
import pickle
from data_preprocessing import preprocess_data

def standard_scaler(X):
    means = X.mean(0)
    stds = X.std(0)
    return (X - means)/stds, means, stds

def sign(x, first_element_zero=False):
    signs = (-1)**(x < 0)
    if first_element_zero:
        signs[0] = 0
    return signs

class RegularizedRegression:
    
    def _record_info(self, X, y, lam, intercept, standardize):
        # standardize 
        if standardize:
            X, self.means, self.stds = standard_scaler(X)
        else:
            self.means, self.stds = None, None
        
        # add intercept
        if not intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)  # column of ones 
            X = np.concatenate((ones, X), axis=1)  # concatenate
            
        # record values
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        self.lam = lam
        self.intercept = intercept
        self.standardize = standardize
        
    def fit_ridge(self, X, y, lam=0, intercept=False, standardize=True):
        # record data and dimensions
        self._record_info(X, y, lam, intercept, standardize)
        
        # estimate parameters
        XtX = np.dot(self.X.T, self.X)
        I_prime = np.eye(self.D)
        I_prime[0,0] = 0 
        XtX_plus_lam_inverse = np.linalg.inv(XtX + self.lam*I_prime)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_plus_lam_inverse, Xty)
        
        # get fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
   