# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:14:01 2022

@author: Ravi
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import flask
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
filename =  'Model_GYM.pkl'
model = pickle.load(open (filename,'rb'))

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1,data2,data3]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)

