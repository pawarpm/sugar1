# Sugarcane Age Classifier — Streamlit App

This repository runs a Streamlit web app that loads a MobileNetV2-based classifier and predicts sugarcane age classes from images.

## Files
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies

## Weights
The app downloads the weights from a Google Drive file ID embedded in `app.py`. The weights file should be a Keras `model.save_weights(...)` HDF5 (as used during training).

Drive link used in the app (example):
