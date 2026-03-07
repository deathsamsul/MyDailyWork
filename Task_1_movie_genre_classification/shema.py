from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np



class data_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['tags'] = df['title']*2 + " " + df['summary']
        df['tags'] = df['tags'].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
        df['tags'] = df['tags'].str.lower()
        text = df['tags'].tolist()
        return text
