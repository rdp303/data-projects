# Regression model comparison using the College dataset
# Portfolio-cleaned version of the original modeling exercise.

library(caret)
library(ISLR)
library(elasticnet)

set.seed(100)

# -----------------------------------------------------------------------------
# 1. Load data and define the modeling target
# -----------------------------------------------------------------------------

data(College)

# Predict the number of applications received (Apps) from all other variables.
# The original exercise modeled log(Apps) to stabilize the response.
predictors <- College[, setdiff(names(College), "Apps")]
response <- log(College$Apps)

# Stratified 80/20 split for a continuous outcome.
train_index <- createDataPartition(response, p = 0.80, list = FALSE)

x_train <- predictors[train_index, ]
y_train <- response[train_index]
x_test <- predictors[-train_index, ]
y_test <- response[-train_index]

# Use 10-fold cross-validation consistently across candidate models.
ctrl <- trainControl(method = "cv", number = 10)

# -----------------------------------------------------------------------------
# 2. Train candidate models
# -----------------------------------------------------------------------------

linear_model <- train(
  x = x_train,
  y = y_train,
  method = "lm",
  preProcess = c("center", "scale", "nzv", "corr", "BoxCox"),
  trControl = ctrl
)

ridge_grid <- expand.grid(lambda = seq(0, 0.20, length.out = 10))

ridge_model <- train(
  x = x_train,
  y = y_train,
  method = "ridge",
  tuneGrid = ridge_grid,
  preProcess = c("center", "scale", "nzv", "BoxCox"),
  trControl = ctrl
)

enet_grid <- expand.grid(
  lambda = c(0, 0.01, 0.10),
  fraction = seq(0.05, 1, length.out = 20)
)

enet_model <- train(
  x = x_train,
  y = y_train,
  method = "enet",
  tuneGrid = enet_grid,
  preProcess = c("center", "scale", "nzv", "BoxCox"),
  trControl = ctrl
)

# -----------------------------------------------------------------------------
# 3. Evaluate on the held-out test set
# -----------------------------------------------------------------------------

evaluate_regression <- function(model, x, y, model_name) {
  predictions <- predict(model, x)
  metrics <- postResample(predictions, y)

  data.frame(
    model = model_name,
    RMSE = unname(metrics["RMSE"]),
    R_squared = unname(metrics["Rsquared"]),
    MAE = unname(metrics["MAE"])
  )
}

results <- rbind(
  evaluate_regression(linear_model, x_test, y_test, "Linear Regression"),
  evaluate_regression(ridge_model, x_test, y_test, "Ridge Regression"),
  evaluate_regression(enet_model, x_test, y_test, "Elastic Net")
)

results <- results[order(results$RMSE), ]
print(results)

# Compare cross-validated resampling distributions across models.
resampling_results <- resamples(list(
  Linear = linear_model,
  Ridge = ridge_model,
  ElasticNet = enet_model
))

print(summary(resampling_results))

# Save reproduced test-set metrics when the script is run from the project root.
dir.create("output", showWarnings = FALSE)
write.csv(results, "output/reproduced_model_performance.csv", row.names = FALSE)
