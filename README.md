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
```bash
pip install -r requirements.txt
