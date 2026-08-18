# Diabetes Classification in R

Machine-learning model comparison using the Pima Indians Diabetes dataset. The project covers data cleaning, missing-value treatment, class imbalance, repeated cross-validation, model comparison, and classification-error tradeoffs.

## Objective

Predict diabetes status from eight clinical and demographic predictors and compare multiple classifiers using accuracy, Cohen's kappa, ROC AUC, and confusion-matrix behavior.

## Workflow

- Treat implausible zero values as missing for glucose, blood pressure, skin thickness, insulin, and BMI
- Use predictive mean matching for remaining missing values
- Create a stratified 80/20 train/test split
- Use up-sampling within training to address class imbalance
- Use repeated 10-fold cross-validation
- Compare logistic regression, KNN, discriminant analysis, SVMs, a neural network, PLS, bagging, gradient boosting, and Random Forest
- Evaluate held-out performance and consider false-negative tradeoffs in addition to aggregate metrics

## Submitted results

Random Forest produced the strongest overall performance in the final submitted comparison.

| Model | Accuracy | Kappa | ROC AUC |
|---|---:|---:|---:|
| Random Forest | 0.8121 | 0.5827 | 0.8793 |
| Bagged Trees | 0.8054 | 0.5619 | 0.8577 |
| Gradient Boosting | 0.7919 | 0.5521 | 0.8633 |
| RDA | 0.7987 | 0.5488 | 0.8557 |
| KNN | 0.7852 | 0.5514 | 0.8727 |
| SVM Radial | 0.7718 | 0.5314 | 0.8715 |

The full submitted model comparison is preserved in [`output/model_performance.csv`](output/model_performance.csv).

Although Random Forest had the strongest overall metrics, the original analysis also considered KNN because of its false-negative behavior. For a screening-oriented problem, model selection should consider the cost of different error types rather than accuracy alone.

## Repository structure

```text
diabetes-prediction-r/
├── data/
│   ├── diabetes_part1.csv
│   ├── diabetes_part2.csv
│   ├── diabetes_part3.csv
│   └── diabetes_part4.csv
├── src/
│   ├── prepare_data.R
│   └── final_analysis.R
├── output/
│   └── model_performance.csv
├── .gitignore
└── README.md
```

The dataset is stored in four source partitions. `src/prepare_data.R` combines them into the single `data/diabetes.csv` file expected by the analysis.

`src/final_analysis.R` is a cleaned portfolio version of the final modeling workflow. Scratch code, duplicate drafts, local RStudio state, and personal metadata were removed while preserving the core analysis approach.

## Reproduce the analysis

Install the required packages in R:

```r
install.packages(c(
  "tidyverse", "caret", "mice", "pROC", "klaR", "mda",
  "gbm", "randomForest", "kernlab", "RSNNS", "pls", "ipred"
))
```

From the `diabetes-prediction-r` directory, run:

```bash
Rscript src/prepare_data.R
Rscript src/final_analysis.R
```

The first command reconstructs `data/diabetes.csv`. The second runs the cleaned modeling workflow and writes a fresh comparison to `output/model_performance_reproduced.csv`.

Because this project was originally completed in an earlier R environment, reproduced metrics may vary slightly with package versions and tuning behavior. `output/model_performance.csv` contains the results from the submitted project.

## Notes

This is an educational modeling project and is not intended for clinical diagnosis or medical decision-making.
