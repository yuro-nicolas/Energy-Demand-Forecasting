#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running full pipeline..."
python run.py
