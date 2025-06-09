#!/bin/bash

# Install requirements locally into site-packages (home/site/wwwroot)
pip install --upgrade pip
pip install -r requirements.txt

# Start gunicorn
gunicorn --bind=0.0.0.0:8000 app:app
