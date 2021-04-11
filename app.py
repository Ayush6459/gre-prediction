import gradio as gr
import numpy as np
import requests
import joblib

predictor=joblib.load('regressor.joblib')


def predict(gre_score,toefl_score,university_rating,sop,cgpa ):
    return str(predictor.predict([[gre_score,toefl_score,university_rating,sop,cgpa]]))

iface=gr.Interface(predict,inputs=['number','number','number','number','number'],outputs=['number'])
iface.launch()
