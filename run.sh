#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_CMD="python3"  # Command to invoke Python 3
VENV_DIR="venv"       # Name of the virtual environment directory
REQUIREMENTS_FILE="requirements.txt"
APP_SCRIPT="app.py"

# --- Sanity Checks ---
# Check if Python 3 command exists
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: $PYTHON_CMD command not found. Please install Python 3."
    exit 1
fi

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found in the current directory."
    exit 1
fi

# Check if app script exists
if [ ! -f "$APP_SCRIPT" ]; then
    echo "Error: $APP_SCRIPT not found in the current directory."
    exit 1
fi

# --- Environment Setup ---
# Check if virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
fi

# --- Install Dependencies ---
echo "Installing dependencies from $REQUIREMENTS_FILE..."
# Use pip from the virtual environment
"$VENV_DIR/bin/pip" install --upgrade pip # Upgrade pip first
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi
echo "Dependencies installed successfully."

# --- Run Application ---
echo "Starting Streamlit application ($APP_SCRIPT)..."
# Use streamlit from the virtual environment
"$VENV_DIR/bin/streamlit" run "$APP_SCRIPT"

echo "Application finished."
exit 0