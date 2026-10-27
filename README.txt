# Resume Classification Project

## Overview
This project involves preprocessing a dataset of resumes and building a machine learning model to classify them into predefined categories.

It is divided into two main notebooks:
1. **DataCleaning.ipynb** – Handles dataset loading, cleaning, and preparation for modeling.
2. **ResumeModel.ipynb** – Trains, evaluates, and exports a classification model.

---

## 1. DataCleaning.ipynb

### Purpose
This notebook focuses on preparing the raw dataset for machine learning. The goal is to clean the data, handle missing values, and transform textual information into a usable form.

### Key Steps
- **Import Libraries**: Pandas, NumPy, regex, and visualization libraries.
- **Load Dataset**: Reads `UpdatedResumeDataSet.csv`.
- **Exploratory Data Analysis (EDA)**:
  - Checks dataset shape, column names, and sample rows.
  - Displays class distribution using a bar chart.
- **Text Preprocessing**:
  - Removes URLs, punctuation, special characters, and extra spaces.
  - Converts text to lowercase.
  - Removes HTML tags and non-ASCII characters.
  - Tokenizes and stems words.
- **Stopword Removal**: Filters common English stopwords to improve model performance.
- **Final Output**: A cleaned dataset ready for feature extraction.

---

## 2. ResumeModel.ipynb

### Purpose
This notebook builds and evaluates a text classification model to predict resume categories.

### Key Steps
- **Import Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, joblib.
- **Load Cleaned Data**: Reads processed dataset from the previous step.
- **Feature Extraction**:
  - Uses `TfidfVectorizer` to convert text into numerical feature vectors.
- **Model Training**:
  - Uses algorithms like Logistic Regression for classification.
  - Splits data into training and testing sets.
- **Evaluation**:
  - Calculates accuracy, precision, recall, and F1-score.
  - Displays confusion matrix.
- **Model Saving**:
  - Saves the trained model and vectorizer using `joblib` for future predictions.

---

## Requirements
Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

---

## How to Run
1. Run **DataCleaning.ipynb** to clean and preprocess the dataset.
2. Run **ResumeModel.ipynb** to train, evaluate, and save the model.
3. Use the saved model and vectorizer to make predictions on new resumes.

---

## Output Files
- `cleaned_data.csv` – Preprocessed dataset.
- `resume_classifier_model.pkl` – Saved trained model.
- `vectorizer.pkl` – Saved TF-IDF vectorizer.

---

