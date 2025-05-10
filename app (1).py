import gradio as gr
import mne
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# Load trained model
model = load("best_model.joblib")
def predict_eeg_states(uploaded_file):
    try:
        file_path = uploaded_file.name
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)

        if raw.info['nchan'] < 4:
            return "❌ EEG file must have at least 4 EEG channels.", None, None, None

        raw.resample(128)
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True, verbose=False)
        data = epochs.get_data()

        if data.shape[2] < 48:
            return "❌ Not enough time samples per epoch (need at least 48).", None, None, None

        X = data[:, :4, :48]
        X_flat = X.reshape(X.shape[0], -1)

        if X_flat.shape[1] != 192:
            return f"❌ Feature shape mismatch: expected 192, got {X_flat.shape[1]}", None, None, None

        predictions = model.predict(X_flat)

        # Separate attention and distraction epochs
        attention_epochs = X[predictions == 1]
        distraction_epochs = X[predictions == 0]

        def save_plot(epochs, label):
            fig, axs = plt.subplots(min(3, len(epochs)), 1, figsize=(7, 2 * min(3, len(epochs))))
            if len(epochs) == 0:
                return None
            if min(3, len(epochs)) == 1:
                axs = [axs]
            for i in range(min(3, len(epochs))):
                axs[i].plot(epochs[i][0])  # Plot channel 0
                axs[i].set_title(f"{label} Epoch {i+1}")
                axs[i].set_ylim([-100e-6, 100e-6])
                axs[i].set_xlabel("Time")
                axs[i].set_ylabel("EEG Amplitude")
            plt.tight_layout()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name)
            plt.close()
            return temp_file.name

        attention_plot = save_plot(attention_epochs, "Attention")
        distraction_plot = save_plot(distraction_epochs, "Distraction")

        summary = f"✅ Prediction complete. {sum(predictions)} Attention out of {len(predictions)} epochs."
        return summary, attention_plot, distraction_plot

    except Exception as e:
        return f"❌ Error occurred: {str(e)}", None, None

iface = gr.Interface(
    fn=predict_eeg_states,
    inputs=gr.File(file_types=[".edf"], label="Upload EEG File"),
    outputs=[
        gr.Text(label="Prediction Summary"),
        gr.Image(type="filepath", label="🟢 Attention Epochs"),
        gr.Image(type="filepath", label="🔴 Distraction Epochs"),
    ],
    title="🧠 EEG Attention vs Distraction Classifier",
    description="Upload a .edf EEG file. The model separates attention and distraction states and plots them."
)
if __name__ == "__main__":
    iface.launch()