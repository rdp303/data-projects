# Diabetes classification model comparison
# Cleaned portfolio version of the final school analysis.

library(tidyverse)
library(caret)
library(mice)
library(pROC)
library(klaR)
library(mda)
library(gbm)
library(randomForest)
library(kernlab)
library(RSNNS)
library(pls)
library(ipred)

set.seed(100)

# -----------------------------------------------------------------------------
# 1. Load and prepare data
# -----------------------------------------------------------------------------

diabetes <- read.csv("data/diabetes.csv")
diabetes$Outcome <- factor(diabetes$Outcome, levels = c(0, 1), labels = c("no", "yes"))

# Zero is not a plausible value for these clinical measurements, so treat it as
# missing before imputation.
diabetes_clean <- diabetes %>%
  mutate(
    Glucose = na_if(Glucose, 0),
    BloodPressure = na_if(BloodPressure, 0),
    SkinThickness = na_if(SkinThickness, 0),
    Insulin = na_if(Insulin, 0),
    BMI = na_if(BMI, 0)
  ) %>%
  filter(!is.na(Glucose), !is.na(BMI))

# Predictive mean matching for remaining missing predictor values.
imputed <- mice(diabetes_clean, method = "pmm", seed = 100, printFlag = FALSE)
diabetes_model <- complete(imputed)

# Stratified 80/20 split.
idx <- createDataPartition(diabetes_model$Outcome, p = 0.80, list = FALSE)
train <- diabetes_model[idx, ]
test <- diabetes_model[-idx, ]

# Repeated cross-validation with up-sampling to address class imbalance.
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE,
  sampling = "up"
)

x_train <- train[, 1:8]
y_train <- train$Outcome
x_test <- test[, 1:8]
y_test <- test$Outcome

# -----------------------------------------------------------------------------
# 2. Train candidate models
# -----------------------------------------------------------------------------

models <- list(
  logistic = train(
    x_train, y_train,
    method = "glm",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "spatialSign")
  ),

  knn = train(
    x_train, y_train,
    method = "knn",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "nzv", "BoxCox"),
    tuneGrid = expand.grid(k = c(3, 5, 7, 9, 25, 50, 75, 100))
  ),

  qda = train(
    x_train, y_train,
    method = "qda",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "nzv", "BoxCox")
  ),

  rda = train(
    x_train, y_train,
    method = "rda",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale")
  ),

  svm_radial = train(
    x_train, y_train,
    method = "svmRadial",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "BoxCox")
  ),

  svm_poly = train(
    x_train, y_train,
    method = "svmPoly",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "BoxCox")
  ),

  pls = train(
    x_train, y_train,
    method = "pls",
    metric = "ROC",
    trControl = ctrl,
    tuneGrid = expand.grid(ncomp = 1:8)
  ),

  mlp = train(
    x_train, y_train,
    method = "mlp",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center", "scale", "spatialSign", "corr")
  ),

  treebag = train(
    x_train, y_train,
    method = "treebag",
    metric = "ROC",
    trControl = ctrl,
    nbagg = 50
  ),

  gbm = train(
    x_train, y_train,
    method = "gbm",
    metric = "ROC",
    trControl = ctrl,
    verbose = FALSE
  ),

  random_forest = train(
    x_train, y_train,
    method = "rf",
    metric = "ROC",
    trControl = ctrl,
    ntree = 200
  )
)

# -----------------------------------------------------------------------------
# 3. Evaluate on the held-out test set
# -----------------------------------------------------------------------------

evaluate_model <- function(model, name) {
  pred_class <- predict(model, x_test)
  pred_prob <- predict(model, x_test, type = "prob")[, "yes"]

  cm <- confusionMatrix(pred_class, y_test, positive = "yes")
  roc_obj <- roc(y_test, pred_prob, levels = c("no", "yes"), direction = "<")

  tibble(
    model = name,
    accuracy = unname(cm$overall["Accuracy"]),
    kappa = unname(cm$overall["Kappa"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    roc_auc = as.numeric(auc(roc_obj))
  )
}

results <- bind_rows(Map(evaluate_model, models, names(models))) %>%
  arrange(desc(roc_auc))

print(results)

# Save a reproducible model comparison table.
dir.create("output", showWarnings = FALSE)
write.csv(results, "output/model_performance_reproduced.csv", row.names = FALSE)

# Inspect Random Forest variable importance.
rf_importance <- varImp(models$random_forest)
print(rf_importance)

# Final project conclusion: Random Forest was the strongest overall model in the
# submitted comparison, while KNN was also considered because of its error mix.
