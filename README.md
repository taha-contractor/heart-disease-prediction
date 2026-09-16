# Heart Disease Prediction

A machine learning practice project that demonstrates the full ML lifecycle, from data understanding and preprocessing to model training, evaluation, and deployment as a web application.

## Project Overview

This project predicts whether a patient is likely to have heart disease based on clinical attributes such as age, sex, blood pressure, cholesterol, heart rate, and exercise-related indicators. In the notebook, several classification algorithms were evaluated and the best-performing model was selected before being saved and deployed through a Streamlit-based web interface.

The main purpose of this project is educational: to show how a real-world ML workflow is built step by step, including model comparison, selection, validation, and deployment.

## ML Lifecycle Demonstrated

This project covers the core stages of the machine learning lifecycle:

1. Problem Definition
   - Predict the presence or absence of heart disease.
   - Use patient medical and lifestyle attributes as input features.

2. Data Collection and Understanding
   - Dataset stored in `data/heart.csv`.
   - Exploratory analysis is performed to inspect feature distributions, correlations, and target balance.

3. Data Cleaning and Preprocessing
   - Handle missing or inconsistent values.
   - Encode categorical features.
   - Ensure the input data is in a format suitable for training.

4. Feature Engineering and Selection
   - Select clinically relevant features.
   - Keep only the columns required by the trained model.

5. Model Training and Comparison
   - Train and compare multiple classification models in the notebook.
   - Evaluate models such as Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and SVM.
   - Select the best-performing model based on evaluation metrics.

6. Model Evaluation
   - Use metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
   - Compare model performance to ensure the solution is reliable.

7. Model Saving and Reuse
   - Save the selected trained model and feature columns for later use in the app.

8. Deployment
   - Package the chosen model into a Streamlit application for interactive prediction.

## Repository Structure

```text
HeartDiseasePrediction/
├── README.md
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   └── heart.csv              # Heart disease dataset
├── models/
│   ├── RF_model.pkl           # Trained Random Forest model
│   └── columns.pkl            # Ordered feature columns used during training
├── notebooks/
│   └── heart_disease_prediction.ipynb
└── .gitignore
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Matplotlib / Seaborn (for analysis and visualization)

## Dataset

The dataset used in this project is a heart disease dataset with features such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- ST slope

The target variable indicates whether heart disease is present or absent.

## Setup Instructions

### 1. Clone the project

```bash
git clone <repository-url>
cd HeartDiseasePrediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn notebook
```

## Run the Application

From the project root, run:

```bash
streamlit run app/app.py
```

This will launch the prediction interface in your browser.

## Model Workflow

The notebook in `notebooks/heart_disease_prediction.ipynb` demonstrates the full model selection workflow, including:

- Loading the dataset
- Performing data exploration
- Cleaning and preprocessing
- Splitting data into training and testing sets
- Training multiple classification algorithms
- Comparing performance using evaluation metrics
- Choosing the best model
- Saving the selected model and feature columns

Once the best model is selected, the app loads the saved artifacts from the `models/` directory and uses them for prediction.

## How the App Works

1. User enters patient details in the Streamlit form.
2. The app converts the input into a DataFrame.
3. It reorders the input columns to match the training model.
4. The saved Random Forest model predicts the class.
5. The result is displayed with probability information.

## Notes

This project is intended for learning and demonstration purposes. It is not a substitute for medical diagnosis and should not be used for clinical decision-making.

## Future Improvements

Potential next steps for the project include:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV for the selected model
- Adding cross-validation and more robust evaluation metrics
- Deploying the app to a cloud platform
- Creating a proper `requirements.txt` file for easier setup
- Comparing additional ensemble methods for further performance gains

## Conclusion

This project is a simple but complete example of how a machine learning model is developed, tested, and deployed in a real workflow. It highlights both the technical and practical aspects of the ML lifecycle in a beginner-friendly way.
