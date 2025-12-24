Overfitting & Underfitting Visual Lab
This project is an interactive machine learning application designed to demonstrate and compare underfitting, good fit, and overfitting behaviors using different classification models.
A graphical user interface (GUI) allows users to load datasets, configure preprocessing steps, train models, and visually analyze performance differences between training and test sets.
The project was developed as part of CENG 465 – Machine Learning.

Features
Load any CSV dataset with a target column
Graphical User Interface (GUI) built with Tkinter
Preprocessing options:
Standard Scaling
Min-Max Scaling
One-Hot Encoding
Supported classification models:
Perceptron
Multi-Layer Perceptron (MLP)
Decision Tree
Adjustable train/test split ratio
Automatic detection of:
🟡 Underfitting
🟢 Good Fit
🔴 Overfitting
Performance evaluation with:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Visualization of Train vs Test Accuracy

Requirements
Python 3.9+
Required libraries are listed in requirements.txt
Install dependencies:
pip install -r requirements.txt

From the project root directory:
python src/gui.py
or (recommended, guarantees correct imports):
python -m src.gui

The GUI will open, allowing you to:
Select a CSV file
Choose preprocessing options
Select a model
Train and evaluate results interactively

Run Preset Experiments (CLI Demo)
To automatically run Underfitting → Good Fit → Overfitting scenarios:
python -m scripts.run_presets

This script runs predefined configurations and prints:
Train/Test metrics
Fit type (Underfitting, Good Fit, Overfitting)
Accuracy gap analysis
This is useful for quick verification and live demonstration.

Reset / Clean Run
If you want a clean state before running again:
1-Stop the program
Just close the GUI window or press CTRL + C in terminal.
2-Recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3- Re-run the project
python -m src.gui
