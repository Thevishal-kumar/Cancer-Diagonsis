# Cancer-Diagonsis
Personalized Cancer Diagnosis
A machine learning project focused on building models for personalized cancer treatment prediction using genetic mutation and textual data. This repository contains modular Python scripts for preprocessing, feature engineering, training classifiers, and evaluating performance metrics.


Models Implemented
| Model                  | File                          | Description                                    |
| ---------------------- | ----------------------------- | ---------------------------------------------- |
| K-Nearest Neighbors    | cancerKNN.py                  | Distance-based classification using KNN        |
| Logistic Regression    | cancerLogisticReg.py          | Linear model using log loss                    |
| Naive Bayes            | cancerNB_Model.py             | Probabilistic classifier with calibration      |
| Support Vector Machine | cancerSVM.py                  | Margin-based classifier                        |
| Random Forest          | cancerRandomForest.py         | Tree ensemble model                            |
| Stacking Classifier    | cancerStackingClassifier.p`   | Combines multiple models using meta-classifier |
| Majority Baseline      | cancerMajorityClassifier.p`   | Always predicts most frequent class            |
| Random Model           | cancerRandomModel.py          | Random prediction baseline                     |


Feature Engineering
| Script                         | Feature Used          | Description                               |
| ------------------------------ | -------------------   | ----------------------------------------- |
| cancerUnivariateText.py        | Text-based features   | Extracted using TF-IDF or CountVectorizer |
| cancerUnivariateGene.py        | Gene features         | One-hot or response encoding              |
| cancerUnivariateVariation.py   | Variation features    | One-hot or response encoding              |



Requirements:
pip install -r requirements.txt

Typical Libraries:
numpy,pandas,scikit-learn,matplotlib,seaborn,nltk



How to Run

1. Clone the repository:
git clone https://github.com/Thevishal-kumar/Cancer-Diagonsis.git
cd personalised-cancer-diagnosis

2. Run individual model scripts:
python cancerLogisticReg.py

3. Each script will:
   Load and preprocess the data
   Train the model
   Calibrate the classifier (if needed)
   Evaluate using Log Loss and Confusion Matrix


Evaluation Metrics:
Log Loss
Confusion Matrix
Precision, Recall, F1-Score
t-SNE Visualization for class separation


Dataset:
This project uses data provided by the MSK-IMPACT Kaggle competition. It includes:
Patient gene and mutation information
Variation descriptions
Text evidence (clinical notes)
