# Customer Churn Prediction with Random Forest

This project demonstrates a complete machine learning workflow for predicting customer churn using a Random Forest Classifier in Python. The workflow includes data preprocessing, feature engineering, handling class imbalance, model training, and evaluation using a real-world telecom dataset.

---

## Features

- **Data Exploration & Visualization:**
  - Loads and inspects a customer churn CSV dataset.
  - Visualizes feature distributions and churn patterns using Seaborn.

- **Data Preprocessing:**
  - Identifies categorical (object) columns and encodes them using OneHotEncoder.
  - Checks and handles missing values.

- **Class Imbalance Handling:**
  - Analyzes class distribution of churn.
  - Applies SMOTE oversampling to balance churn vs. non-churn classes.

- **Feature Scaling:**
  - Standardizes features for optimal model performance.

- **Model Building:**
  - Splits data into training and test sets.
  - Trains a RandomForestClassifier on the processed data.

- **Evaluation Metrics:**
  - Computes and prints accuracy, precision, recall, F1-score, ROC AUC, and confusion matrix on the test set.

---

## Usage Instructions

1. **Requirements:**
   - Python 3.x
   - pandas
   - numpy
   - seaborn
   - scikit-learn
   - imbalanced-learn

   Install dependencies:
   ```bash
   pip install pandas numpy seaborn scikit-learn imbalanced-learn
   ```

2. **Run the Notebook:**
   - Place `customer_churn.csv` in the same directory.
   - Open `customer_churn.ipynb` in Jupyter Notebook.
   - Execute cells sequentially to reproduce the workflow.

---

## Workflow Summary

1. **Import libraries:** Load all necessary Python libraries.
2. **Load data:** Read the CSV into a DataFrame and inspect.
3. **EDA & Visualization:** Explore numerical and categorical features, visualize churn distribution.
4. **Encoding:** Apply one-hot encoding to categorical variables.
5. **Class Imbalance:** Use SMOTE to balance the target classes.
6. **Train/Test Split:** Separate data for training and validation.
7. **Scaling:** Normalize features with StandardScaler.
8. **Model Training:** Fit a Random Forest model on the training data.
9. **Evaluation:** Print accuracy, classification report, ROC AUC, and confusion matrix.

---

## Example Output

```
Accuracy: 0.99
              precision    recall  f1-score   support
         0.0       0.97      1.00      0.99      1021
         1.0       1.00      0.97      0.99      1049
   macro avg       0.99      0.99      0.99      2070
weighted avg       0.99      0.99      0.99      2070

ROC_AUC_Score: 0.99
Confusion Matrix:
[[1021    0]
 [  30 1019]]
```

---

## Notes

- The full encoding step results in a high-dimensional dataset due to one-hot encoding of categorical and numerical features.
- SMOTE is used to ensure fair representation of both churn and non-churn customers for model training.
- The workflow is modular and can be adapted for other classifiers or advanced feature engineering.

---

## License

This project is released under the MIT License.

---

**For educational purposes. Adapt and extend for your own business use cases!**