import numpy as np
import pandas as pd
import joblib
def discretize_state(state):
    discretized_state = []
    for s, bins in zip(state, state_bins):
        if s <= bins[0]:
            discretized_state.append(0)
        elif s >= bins[-1]:
            discretized_state.append(len(bins) - 1)
        else:
            discretized_state.append(np.digitize(s, bins) - 1)
    return tuple(discretized_state)
q_table=joblib.load('q_table.pkl')
df=pd.read_csv('PS_20174392719_1491204439457_log.csv')
state_bins = [np.linspace(-5, 5, 10) for _ in range(len(df.columns))]
custom_sample_data = [181.0, 181.0, 0.0, 0.0, 0.0]  # Example sample data
mean_values = [df['amount'].mean(),df['oldbalanceOrg'].mean(),df['newbalanceOrig'].mean(),df['oldbalanceDest'].mean(),df['newbalanceDest'].mean()]  # Fill in with mean values used during training
std_values =  [df['amount'].std(),df['oldbalanceOrg'].std(),df['newbalanceOrig'].std(),df['oldbalanceDest'].std(),df['newbalanceDest'].std()]
normalized_features = [(feature - mean_val) / std_val for feature, mean_val, std_val in zip(custom_sample_data, mean_values, std_values)]
discretized_state = discretize_state(normalized_features)
action = np.argmax(q_table[tuple(discretized_state)])
action_str = "Approve" if action == 0 else "Decline"
print("Action chosen for custom sample data:", action_str)