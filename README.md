# Detecting_fake_job_postings
Fake Job Posting Detection — Model Comparison Project

Executive Summary
This project evaluates multiple machine learning techniques to detect fraudulent job postings using a Kaggle dataset of 17,880 listings, of which only 4.8% are labeled as fake. The work spans the full data‑science lifecycle — from exploratory analysis and feature engineering to model tuning and final model selection.

The dataset required substantial preprocessing due to extensive missingness, highly unstandardized categorical fields (e.g., 11,000+ unique job titles and 3,100+ locations), and sparse category distributions. Features were engineered through standardization, parsing of location and salary fields, consolidation of rare categories, and transformation of text‑heavy fields into numerical and thematic representations.
Multiple models were trained and tuned, including logistic regression, random forest, and XGBoost, with class imbalance addressed via weighting. Models were evaluated across alternative feature sets and probability thresholds. The champion model — XGBoost using detailed text‑derived features — delivered the strongest performance on the holdout test set using the default classification threshold.

Project Objective
The goal of this project is to evaluate a range of supervised learning models and identify the best‑performing approach for detecting fraudulent job postings. The dataset contains a mix of structured categorical variables, binary indicators, and several text‑heavy fields, making it a realistic and challenging classification problem.

Dataset source:
Real or Fake? Fake Job Posting Prediction
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction (kaggle.com)

Dataset Summary
- Total job postings: 17,880
- Fraudulent postings: 866
- Class imbalance: ~4.8% fraudulent
- Feature types include:
- Semi‑structured categorical fields
- Multiple free‑text fields
- Binary flags

Project Workflow
1. Data Exploration and Standardization
- Identified missingness across 10+ fields, each with over 1,000 missing values
- Observed highly unstandardized categorical fields (11,000+ job titles, 3,100+ locations)
- Noted sparsity issues, with many categories appearing fewer than 50 times

2. Feature Engineering
- Standardized categorical fields (uppercasing, whitespace removal)
- Parsed location field to extract a clean country indicator
- Parsed salary ranges into low, high, and difference components
- Collapsed rare categorical levels into an “other” category
- Converted text‑heavy fields into numerical and thematic features

3. Model Fitting and Tuning
- Fit logistic regression, random forest, and XGBoost models
- Applied class weighting to address imbalance
- Tuned hyperparameters for each model
- Evaluated alternative probability cut‑off thresholds
- Tested reduced vs. expanded feature sets derived from text fields

4. Champion Model Selection
- Selected the best model based on performance on the holdout test set
- Winner: XGBoost using detailed text‑derived features at the default threshold

