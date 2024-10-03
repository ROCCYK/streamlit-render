# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:00:20 2020

@author: Noopa Jagadeesh
"""


import numpy as np
import pickle
import pandas as pd
#from flask import Flask, request
#import flasgger
#from flasgger import Swagger
import streamlit as st 

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("./classifier.pkl","rb")
classifier=pickle.load(pickle_in)


#@app.route('/')
def hello():
    return "Welcome All to Week-5"

#@app.route('/predict', methods=["GET"])
def predict_class(sepal_length,sepal_width,petal_length,petal_width):
    
    """Let's predict the class for iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    #sepal_length=request.args.get('sepal_length')
    #sepal_width=request.args.get('sepal_width')
    #petal_length=request.args.get('petal_length')
    #petal_width=request.args.get('petal_width')
    prediction=classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    return str(prediction)

def main():
    st.title("Iris Flower Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Iris Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sepal_length = st.text_input("sepal_length","Type Here")
    sepal_width = st.text_input("sepal_width","Type Here")
    petal_length = st.text_input("petal_length","Type Here")
    petal_width = st.text_input("petal_width","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_class(sepal_length,sepal_width,petal_length,petal_width)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("AI powered App")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
