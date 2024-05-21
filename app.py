import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home() :
                    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
                    features = list(request.form.values())
                    for i in range ( 0, 5 ) :
                                        features[i] = int(features[i])
                    for i in range ( 5, 10 ) :
                                        features[i] = float(features[i])
                    features[10] = int(features[i])
                    features = np.array( features )
                    features = features.reshape(1, -1)
                    prediction = model.predict( features )
                    
                    return render_template('answer.html', prediction_text = 'loan approved or not {}' .format(prediction) )

if __name__ == "__main__":
                    app.run(debug=True)
                    
                    
#change