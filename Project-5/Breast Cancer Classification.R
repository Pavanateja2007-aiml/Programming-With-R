# ============================================================
# Breast Cancer Classification Using Random Forest
# Dataset: Wisconsin Breast Cancer (mlbench::BreastCancer)
# Goal: Predict Malignant vs Benign
# Libraries: ggplot2, corrplot, caret, randomForest, dplyr
# ============================================================

# Install required packages if missing
required_packages <- c("mlbench", "ggplot2", "corrplot", "caret", "randomForest", "dplyr", "ROCR")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# -------------------------------
# 1. Load and clean dataset
# -------------------------------
data("BreastCancer", package = "mlbench")
df <- BreastCancer
str(df)

# Remove ID column (not a feature)
df <- df[, -1]

# Convert factor columns to numeric (except Class)
for (i in 1:(ncol(df)-1)) {
  df[, i] <- as.numeric(as.character(df[, i]))
}

# Remove rows with missing values
df <- na.omit(df)

# Convert target to factor (Benign = "benign", Malignant = "malignant")
df$Class <- factor(df$Class, levels = c("benign", "malignant"))

# Check class distribution
table(df$Class)

# -------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------
# Class distribution plot
ggplot(df, aes(x = Class, fill = Class)) +
  geom_bar() +
  labs(title = "Class Distribution (Benign vs Malignant)") +
  theme_minimal()

# Feature distributions (first 4 features)
features <- names(df)[1:4]
df_long <- tidyr::pivot_longer(df, cols = all_of(features), names_to = "Feature", values_to = "Value")
ggplot(df_long, aes(x = Value, fill = Class)) +
  geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Feature Distributions by Class") +
  theme_minimal()

# Correlation heatmap (first 10 features)
library(corrplot)
cor_matrix <- cor(df[, 1:10])
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, title = "Feature Correlations")

# -------------------------------
# 3. Train/Test split (stratified)
# -------------------------------
set.seed(42)
trainIndex <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test  <- df[-trainIndex, ]

# Separate features and target
X_train <- train[, -ncol(train)]
y_train <- train$Class
X_test  <- test[, -ncol(test)]
y_test  <- test$Class

# -------------------------------
# 4. Feature scaling (standardize)
# -------------------------------
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled  <- predict(preproc, X_test)

# -------------------------------
# 5. Train Random Forest model
# -------------------------------
set.seed(42)
rf_model <- randomForest(X_train_scaled, y_train, ntree = 100, importance = TRUE)

# -------------------------------
# 6. Model Evaluation
# -------------------------------
# Predictions
y_pred <- predict(rf_model, X_test_scaled)

# Accuracy
accuracy <- confusionMatrix(y_pred, y_test)$overall["Accuracy"]
cat("Test Accuracy:", round(accuracy, 4), "\n")

# Cross-validation
ctrl <- trainControl(method = "cv", number = 5)
rf_cv <- train(X_train_scaled, y_train, method = "rf", trControl = ctrl, ntree = 100)
cat("5-Fold CV Accuracy:", round(max(rf_cv$results$Accuracy), 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(y_pred, y_test)
print(cm$table)

# ROC Curve and AUC
library(ROCR)
pred_prob <- predict(rf_model, X_test_scaled, type = "prob")[, "malignant"]
pred_obj <- prediction(pred_prob, y_test)
perf <- performance(pred_obj, "tpr", "fpr")
plot(perf, col = "red", main = "ROC Curve")
auc <- performance(pred_obj, "auc")@y.values[[1]]
legend("bottomright", legend = paste("AUC =", round(auc, 3)), col = "red", lty = 1)

# -------------------------------
# 7. Feature Importance
# -------------------------------
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

# Top 10 features
top10 <- head(importance_df, 10)
ggplot(top10, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Feature Importances (Random Forest)",
       x = "Features", y = "Mean Decrease Gini") +
  theme_minimal()

# Print top 5
print(head(importance_df, 5))

# -------------------------------
# End of script
# -------------------------------