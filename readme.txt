Machine Learning Model Comparison for Breast Cancer Classification
1. Project Overview
This project implements a complete machine learning workflow to classify breast tumors as either Malignant or Benign. The primary goals are:
Train multiple classification models on the Breast Cancer Wisconsin (Diagnostic) dataset.
Evaluate and compare their baseline performance using key metrics (Accuracy, Precision, Recall, F1-Score).
Implement hyperparameter tuning techniques (GridSearchCV and RandomizedSearchCV) to optimize model parameters.
Analyze the results to select the best-performing model for this specific classification task.
2. Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset.
Source: Originally from the UCI Machine Learning Repository, commonly available on platforms like Kaggle.
Features: 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image (e.g., radius, texture, perimeter, area).
Target Variable: diagnosis (M = Malignant, B = Benign).
Instances: 569
Problem Type: Binary Classification.
3. Project Workflow
The project follows these sequential steps:
Data Loading and Preprocessing: The data is loaded from data.csv. Unnecessary columns (id, Unnamed: 32) are dropped. The categorical diagnosis column is encoded into a numerical format (Malignant: 1, Benign: 0).
Data Splitting and Scaling: The dataset is split into training (70%) and testing (30%) sets. Feature scaling is applied using StandardScaler to normalize the data, which is crucial for models like Logistic Regression and SVM.
Baseline Model Training: Four different models are trained with their default parameters to establish a performance baseline:
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Random Forest
Hyperparameter Tuning:
GridSearchCV is used to find the optimal parameters for the Support Vector Machine from a predefined grid.
RandomizedSearchCV is used to efficiently search a large parameter space for the Random Forest model.
Final Evaluation and Analysis: All models (baseline and tuned) are compared side-by-side. The best model is selected based on the F1-Score, and a detailed classification report is generated for in-depth analysis.
4. How to Run
Prerequisites
Python 3.x
The following Python libraries are required. You can install them using pip:
Generated bash
pip install pandas numpy scikit-learn
Use code with caution.
Bash
Running the Script
Clone this repository or place the project_script.py and data.csv in the same directory.
Open your terminal or command prompt.
Navigate to the project directory.
Run the script:
Generated bash
python project_script.py
Use code with caution.
Bash
5. Results and Analysis
The script produced the following results on the test set.
Baseline Model Performance
Initially, all models performed well, with accuracies above 95%. Notably, KNN, SVM, and Random Forest achieved perfect precision (1.0), meaning they made no false positive predictions for the malignant class. However, their recall was lower than that of Logistic Regression, indicating they missed more actual malignant cases.
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.970760	0.983607	0.937500	0.960000
K-Nearest Neighbors	0.964912	1.000000	0.906250	0.950820
Random Forest	0.964912	1.000000	0.906250	0.950820
Support Vector Machine	0.959064	1.000000	0.890625	0.942149
Hyperparameter Tuning
The tuning process identified the following optimal parameters:
SVM (GridSearchCV): {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
Random Forest (RandomizedSearchCV): {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 162}
Final Model Comparison
After including the tuned models, we compared all contenders. The models are sorted by their F1-Score, which provides a balanced measure of precision and recall.
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.970760	0.983607	0.937500	0.960000
Random Forest (Tuned)	0.970760	1.000000	0.921875	0.959350
K-Nearest Neighbors	0.964912	1.000000	0.906250	0.950820
Random Forest	0.964912	1.000000	0.906250	0.950820
Support Vector Machine	0.959064	1.000000	0.890625	0.942149
SVM (Tuned)	0.953216	1.000000	0.875000	0.933333
6. Conclusion and Final Model Selection
The best-performing model for this task is the baseline Logistic Regression model, achieving the highest F1-Score of 0.96.
This is an interesting outcome, as more complex models like a tuned Random Forest are often expected to outperform simpler ones. It suggests that the decision boundary in this dataset is relatively linear and that the simpler, well-regularized Logistic Regression model generalizes better than the more complex models, which may have slightly overfit or whose tuned parameters were not globally optimal.
It is also noteworthy that the tuned SVM performed worse than its baseline version on this particular test set, which can sometimes occur due to the specific cross-validation splits during the search.
Detailed Report for the Best Model: Logistic Regression
The classification report provides a deeper look into its performance:
Generated code
Classification Report:
              precision    recall  f1-score   support

      Benign       0.96      0.99      0.98       107
   Malignant       0.98      0.94      0.96        64

    accuracy                           0.97       171
   macro avg       0.97      0.96      0.97       171
weighted avg       0.97      0.97      0.97       171
Use code with caution.
Interpretation:
For Malignant cases (the critical class): The model has a recall of 0.94, meaning it correctly identifies 94% of all actual malignant tumors. This is a very strong result, as minimizing false negatives is paramount in medical diagnoses.
The precision of 0.98 means that when the model predicts a tumor is malignant, it is correct 98% of the time, minimizing false alarms.