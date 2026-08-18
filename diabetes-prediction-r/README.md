# Diabetes Classification in R

A machine-learning model comparison project using the Pima Indians Diabetes dataset. The analysis focuses on data quality, missing-value treatment, class imbalance, repeated cross-validation, and comparison of multiple classification approaches in R.

![Project poster](output/poster_preview.png)

## Project objective

Build and compare classification models that predict diabetes status from eight clinical and demographic predictors, then evaluate the tradeoffs between overall predictive performance and classification errors.

## What the analysis covers

- Exploratory analysis of distributions, missing values, outliers, and class balance
- Conversion of implausible zero measurements to missing values
- Predictive mean matching for missing-data imputation
- Stratified train/test splitting and up-sampling of the training data
- Model-specific preprocessing including centering, scaling, Box-Cox, and spatial-sign transformations
- Repeated 10-fold cross-validation
- Comparison of linear, distance-based, SVM, neural-network, discriminant-analysis, bagging, boosting, and random-forest classifiers
- Evaluation with accuracy, Cohen's kappa, ROC AUC, confusion matrices, and error tradeoffs

## Results

Random Forest produced the strongest overall performance in the final comparison.

| Model | Accuracy | Kappa | ROC AUC |
|---|---:|---:|---:|
| Random Forest | 0.8121 | 0.5827 | 0.8793 |
| KNN | 0.7852 | 0.5514 | 0.8727 |
| SVM Radial | 0.7718 | 0.5314 | 0.8715 |
| QDA | 0.7852 | 0.5315 | 0.8656 |
| Gradient Boosting | 0.7919 | 0.5521 | 0.8633 |

The full model comparison is available in [`output/model_performance.csv`](output/model_performance.csv).

Although Random Forest had the best aggregate metrics, the analysis also considered false-negative behavior. That distinction matters for a screening-oriented classification problem, where model selection should consider error costs rather than accuracy alone.

## Repository structure

```text
.
├── data/
│   └── diabetes.csv
├── src/
│   └── final_analysis.Rmd
├── output/
│   ├── diabetes_prediction_poster.pdf
│   ├── diabetes_prediction_poster.pptx
│   ├── model_performance.csv
│   └── poster_preview.png
└── README.md
```

`src/final_analysis.Rmd` is a cleaned version of the final R Markdown analysis used for the project. Scratch files, duplicate drafts, local RStudio state, and personal metadata were removed while retaining the modeling workflow used for the final deliverable.

## Running the analysis

The project was built in R with `caret` as the primary modeling framework. Core packages include:

```r
install.packages(c(
  "tidyverse", "caret", "mice", "DataExplorer", "pROC",
  "themis", "VIM", "reshape2", "GGally", "klaR", "mda",
  "gbm", "randomForest", "kernlab", "RSNNS", "pls", "ipred"
))
```

From RStudio, open `src/final_analysis.Rmd` and knit the document. The source expects the repository structure above and reads the dataset from `data/diabetes.csv`.

## Notes

This is an educational modeling project and is not intended for clinical diagnosis or medical decision-making.
