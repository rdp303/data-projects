# Modeling Practice: Regression & Regularization in R

A compact regression modeling exercise comparing **ordinary least squares, ridge regression, and elastic net** on the `College` dataset from the `ISLR` package.

## Objective

Predict the number of college applications received (`Apps`) using the remaining institutional characteristics in the dataset. The response is log-transformed and models are evaluated on a held-out test set.

## Methods

- 80/20 train/test split
- 10-fold cross-validation
- Centering and scaling
- Near-zero variance filtering
- Box-Cox transformations where appropriate
- Linear regression baseline
- Ridge regression with cross-validated regularization strength
- Elastic net with cross-validated tuning
- Held-out evaluation using RMSE, R-squared, and MAE

## Reported results

The original submitted analysis reported the following held-out metrics:

| Model | RMSE | R-squared | MAE |
|---|---:|---:|---:|
| Ridge Regression | **0.2651** | **0.9534** | **0.2075** |
| Linear Regression | 0.3166 | 0.9206 | 0.2532 |
| Elastic Net | 0.6720 | 0.9480 | 0.5492 |

On that split, **ridge regression had the lowest RMSE and MAE**.

The submitted metrics are preserved in [`output/reported_model_performance.csv`](output/reported_model_performance.csv). The cleaned script writes a separate `reproduced_model_performance.csv` when rerun so the original reported results are not overwritten.

## Repository structure

```text
modeling-practice/
├── README.md
├── .gitignore
├── src/
│   └── modeling_practice.R
└── output/
    ├── modeling-practice.pdf
    └── reported_model_performance.csv
```

## Run the analysis

Install the required packages:

```r
install.packages(c("caret", "ISLR", "elasticnet"))
```

Then run from the project root:

```r
source("src/modeling_practice.R")
```

## Notes

`output/modeling-practice.pdf` is a sanitized copy of the original project report with identifying information removed. The R script has been cleaned for readability and uses a conventional 10-fold cross-validation setup while retaining the same modeling objective and model families as the original exercise.
