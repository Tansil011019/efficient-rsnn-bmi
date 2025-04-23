#!/bin/bash

# Stop script on error
set -e

echo "Start Setting Up"

VENV_DIR="venvs/efficient-rsnn-bmi"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "No virtual environment found, creating one..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Using virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found!"
    exit 1
fi

python3 -m ensurepip --upgrade
pip install --upgrade pip
python3 -m pip install --require-virtualenv -e "."

echo "Project is completely set"