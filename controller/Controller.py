# Import necessary libraries 
import pandas as pd
import numpy as np
from model import preprocessing, predict
from view import View


# Preprocess data
def preprocess_data(data):
    preprocessing.preprocess_data(data)
    return data


# Predict transformed data
def predict_data(data):
    data = preprocessing.__main__(data)
    pred_status = predict.predict_data(data)
    return pred_status


# View display page
def display():
    View.__main__() 