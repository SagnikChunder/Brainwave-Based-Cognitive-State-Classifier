# app.py

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Load the pre-trained model and data
clf = load('model.joblib')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

st.set_page_config(page_title="EEG Live Prediction", layout="wide")
st.title("🧠 EEG Attention vs. Distraction - Live Stream")

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 4))
predictions = []
truths = []
line1, = ax.plot([], [], label='Predicted', color='blue')
line2, = ax.plot([], [], label='Actual', color='green', linestyle='--')

ax.set_ylim(-0.5, 1.5)
ax.set_xlim(0, 50)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Distraction', 'Attention'])
ax.set_xlabel('Time Window')
ax.set_ylabel('Cognitive State')
ax.set_title('EEG Attention vs. Distraction - Live Animation')
ax.legend()
ax.grid(True)

# Function to update the plot in the animation
def update(frame):
    sample = X_test[frame].reshape(1, -1)
    pred = clf.predict(sample)[0]
    actual = y_test[frame]

    predictions.append(pred)
    truths.append(actual)

    start = max(0, len(predictions) - 50)
    line1.set_data(range(len(predictions[start:])), predictions[start:])
    line2.set_data(range(len(truths[start:])), truths[start:])
    ax.set_xlim(0, max(50, len(predictions[start:])))
    return line1, line2

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(X_test), interval=200, blit=True)

# Render the animation on the Streamlit app
st.pyplot(fig)

# Optional: To control animation playback
st.slider("Frame slider", min_value=0, max_value=len(X_test)-1, value=0, step=1)