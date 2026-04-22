Overview

SynthEval is an end-to-end system for generating synthetic tabular data and evaluating its quality using multiple metrics.

This project combines:

A Python-based evaluation pipeline
An interactive web portal (in-browser tool)

The objective is to assess synthetic data across four critical aspects:

Utility → Does synthetic data work like real data?
Fidelity → Does it preserve distributions and relationships?
Privacy → Does it leak real data?
Detectability → Can models distinguish it from real data?
 Features
 Synthetic Data Generators
Bootstrap Sampling — random row sampling (baseline)
Gaussian Noise — adds controlled perturbations
Marginal Distribution Sampling — independent column sampling
SMOTE-like Interpolation — generates realistic intermediate samples
 Evaluation Metrics
 Utility (TSTR)
Train on Synthetic, Test on Real
Models used:
Logistic Regression
KNN
 Fidelity
KS Statistic
Wasserstein Distance
Jensen-Shannon Divergence (JSD)
DPCM (Difference in Pairwise Correlation Matrix)
 Privacy
DCR (Distance to Closest Record)
Detects memorization risk
 Detection
Logistic classifier trained to distinguish:
Real vs Synthetic
Ideal score ≈ 50%
 Web Portal

The project includes a fully interactive browser-based UI for easy experimentation.

 Features:
Upload CSV file or load dataset via URL
Select generation method
Configure synthetic data size and target column
Run full evaluation pipeline instantly
Visualize results using charts and tables
Download generated synthetic dataset
 Project Structure
├── data/                          # Input datasets
├── models/                        # ML models
├── evaluation.py                  # Metrics implementation
├── model.py                       # Model training logic
├── main.py                        # Main execution script
├── web_portal/
│   └── synthetic_eval_portal.html # Web interface
├── requirements.txt               # Dependencies
└── README.md
 Dependencies
Python Requirements
Python 3.8+
numpy
pandas
scikit-learn
scipy

Install dependencies:

pip install -r requirements.txt
 How to Run
 Option 1: Python Pipeline

Run the full pipeline:

python main.py
Steps:
Load dataset
Preprocess data
Generate synthetic data
Evaluate metrics
Print results
 Option 2: Web Portal 
Open the file:
web_portal/synthetic_eval_portal.html
Steps:
Upload CSV or paste dataset URL
Choose generator method
Select target column
Click Run Evaluation
Output:
Metrics dashboard
Graphical visualizations
Downloadable synthetic dataset
 Metrics Interpretation
 Utility (TSTR)
High score → synthetic data is useful
Close to baseline → good quality
 Fidelity
Lower values → distributions match better
KS ≈ 0 → identical distributions
 Privacy (DCR)
Higher distance → safer data
Very low values → risk of data leakage
 Detection
~50% → ideal (indistinguishable)
High accuracy → poor synthetic quality
 Design Principles
Model-independent evaluation
Modular architecture
Easy to extend with new metrics or generators
Visual + analytical approach
