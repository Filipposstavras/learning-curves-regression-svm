# Learning Curves, Polynomial & Ridge Regression, and SVM Classification

This project explores **bias–variance tradeoffs** and **Support Vector Machines** through two problems:

1. **Regression on synthetic sinusoidal data**  
   - Linear Regression  
   - Polynomial Regression (degree 50)  
   - Ridge Regression (α = 0.001)  
   - Learning curves (training vs validation RMSE)  
   - 10-fold Cross Validation  
   - Analysis of underfitting, overfitting, and regularization  

2. **Classification with the Breast Cancer Dataset**  
   - Features: *Worst Area* & *Mean Concave Points*  
   - Linear SVM with different C values (0.1 and 1000)  
   - Visualizing support vectors and decision boundaries  
   - RBF SVM with hyperparameter tuning (GridSearchCV)  
   - Performance evaluated with F1-score  

---

## How to Run
Install dependencies:
Run the script:

python regression_and_svm.py


Open the notebook for detailed steps and plots:

jupyter notebook regression_and_svm.ipynb

Results

Regression:

Linear Regression → consistent but high error (underfitting)

Polynomial Regression (deg=50) → severe overfitting

Ridge Regression → best performance, balanced training/validation error

SVM:

Linear SVM → decision boundaries shift with C, number of support vectors changes

RBF SVM (Grid Search) → best parameters: C=10, γ=1 with F1 ≈ 0.96

Dataset Sources

Synthetic sinusoidal data (generated)

Breast Cancer Dataset — scikit-learn
