from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#import ai
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/agent-service')
def agent_service():
    return render_template('agent_service.html')

#Sms Fraud detection function
model_name = "austinb/fraud_text_detection"
tokenizer_sms = AutoTokenizer.from_pretrained(model_name)
model_sms = AutoModelForSequenceClassification.from_pretrained(model_name)
def predict_sms_fraud(text):
    inputs = tokenizer_sms(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sms(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 1:
        return "It is a fraud message!"
    else:
        return "The message is not fraud."
    
@app.route('/sms-fraud-detection', methods=['GET', 'POST'])
def sms_fraud_detector():
    if request.method == 'POST':
        sms_to_predict = request.form['sms']
        prediction = predict_sms_fraud(sms_to_predict)
        return render_template('sms.html', prediction=prediction)
    return render_template('sms.html')

#For Fake Transaction Detection
@app.route('/transaction-detector', methods=['GET', 'POST'])
def transaction_detector():
    if request.method == 'POST':
        # Extract input data from the form
        type_val = request.form['type']
        amount_val = float(request.form['amount'])
        oldbalanceOrg_val = float(request.form['oldbalanceorg'])
        newbalanceOrg_val = float(request.form['newbalanceorg'])
        oldbalanceDest_val = float(request.form['oldbalancedest'])
        newbalanceDest_val = float(request.form['newbalancedest'])
        if type_val=='CASH_OUT':
            type_val=1
        elif type_val=='PAYMENT': 
            type_val=2
        elif type_val=='CASH_IN':
            type_val=3
        elif type_val=='TRANSFER':
            type_val=4
        elif type_val=='DEBIT':
            type_val=5               
        # Define the input sample
        sample = np.array([[type_val, amount_val, oldbalanceOrg_val, newbalanceOrg_val, oldbalanceDest_val, newbalanceDest_val]])
        
        # Load the pickled model
        with open('new.pkl', 'rb') as f:
            model_1 = pickle.load(f)
        
        # Make predictions
        prediction = model_1.predict(sample)
        
        # Render the template with the prediction result
        return render_template('transaction_detector.html', prediction=prediction)
    else:
        # Render the empty form
        return render_template('transaction_detector.html')
    
#for csv 
with open('new.pkl', 'rb') as f:
    model_2 = pickle.load(f)

@app.route('/transaction-detector-csv', methods=['GET', 'POST'])
def transaction_detector_csv():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return render_template('transaction_detector_csv.html', error="No file part")
        file = request.files['csv_file']
        if file.filename == '':
            return render_template('transaction_detector_csv.html', error="No selected file")
        if file:
            try:
                data = pd.read_csv(file)
                data = np.array(data)
                predictions = []
                for i in range(len(data)):
                    y_pred = model_2.predict([data[i]])
                    predictions.append(y_pred[0])
                return render_template('transaction_detector_csv.html', predictions=predictions)
            except Exception as e:
                return render_template('transaction_detector_csv.html', error=str(e))
    return render_template('transaction_detector_csv.html')


#Phisng section code
data=pd.read_csv('Phishing_site.csv')
X = data['URL'].astype(str)
y = data['label']
model_phising=tf.keras.models.load_model('Phising_model.h5')
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
def predict_url_phishing(url, tokenizer, model):
    url_seq = tokenizer.texts_to_sequences([url])
    url_padded =tf.keras.preprocessing.sequence.pad_sequences(url_seq, maxlen=100)
    prediction = model.predict(url_padded)
    if prediction > 0.5:
        return "Phishing Site"
    else:
        return "Not Phishing Site"
@app.route('/phishing-site-detector', methods=['GET', 'POST'])
def phishing_site_detector():
    if request.method == 'POST':
        url_to_predict = request.form['url']
        prediction = predict_url_phishing(url_to_predict, tokenizer, model_phising)
        return render_template('phishing_site_detector.html', prediction=prediction)
    return render_template('phishing_site_detector.html')

if __name__ == '__main__':
    app.run(debug=True)
