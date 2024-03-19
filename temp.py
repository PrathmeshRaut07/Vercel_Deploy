import pickle
import numpy as np
import pandas as pd
# Load the pickled model
with open('new.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your input sample
type_val = 2
amount_val = 9839.64
oldbalanceOrg_val = 170136.0
newbalanceOrg_val = 160296.36
oldbalanceDest_val = 0.0
newbalanceDest_val = 0.0

sample = np.array([[type_val, amount_val, oldbalanceOrg_val, newbalanceOrg_val, oldbalanceDest_val, newbalanceDest_val]])

data=pd.read_csv('Test_Transaction.csv')
data=np.array(data)
for i in range(10):
    y_pred=model.predict([data[i]])
    print(f'{i} transcations is '+y_pred[0])

print(y_pred)

