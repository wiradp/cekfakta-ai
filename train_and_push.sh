#!/bin/bash

echo "Training model..."
python model/train_model.py

echo "Model training selesai."

echo "Commit dan push model ke GitHub..."
git add model/vectorizer.pkl model/classifier.pkl
git commit -m "update model"
git push

echo "Done."

