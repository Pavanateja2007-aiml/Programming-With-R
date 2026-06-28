# ============================================================================
# Heart Disease Prediction (Binary Classification) - R Implementation
# Dataset: heart.csv (UCI Heart Disease) from Kaggle
# Goal: Predict presence of heart disease (1 = disease, 0 = no disease)
# ============================================================================

# ------------------------- 1. Load Required Libraries -----------------------
# If any package is missing, install it with: install.packages("package_name")
library(tidyverse)    # data manipulation & ggplot2
library(caret)        # train/test split, preprocessing, GridSearchCV, cross-validation
library(randomForest) # Random Forest model
library(pROC)         # ROC curve & AUC
library(gridExtra)    # arrange multiple plots
library(factoextra)   # PCA visualization (optional)

# ------------------------- 2. Load & Explore Dataset ------------------------

heart <- read.csv("C:\\Users\\My world\\OneDrive\\Desktop\\R\\project6\\heart.csv", stringsAsFactors = TRUE)

# Check structure and missing values
str(heart)
summary(heart)
anyNA(heart)   # UCI dataset typically has no missing values

# Convert target to factor (required for classification in caret)
heart$target <- as.factor(heart$target)

# ------------------------- 3. Exploratory Data Analysis ----------------------
# Histograms & boxplots for age, cholesterol (chol), maximum heart rate (thalach)
p1 <- ggplot(heart, aes(x = age)) + 
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Age Distribution")

p2 <- ggplot(heart, aes(x = chol)) + 
  geom_histogram(bins = 30, fill = "salmon", color = "black") +
  labs(title = "Cholesterol Distribution")

p3 <- ggplot(heart, aes(x = thalach)) + 
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  labs(title = "Max Heart Rate (thalach) Distribution")

# Boxplots (can also use facet wrap)
p4 <- ggplot(heart, aes(y = age, x = target)) + geom_boxplot(fill = "skyblue") +
  labs(title = "Age by Disease Status")
p5 <- ggplot(heart, aes(y = chol, x = target)) + geom_boxplot(fill = "salmon") +
  labs(title = "Cholesterol by Disease Status")
p6 <- ggplot(heart, aes(y = thalach, x = target)) + geom_boxplot(fill = "lightgreen") +
  labs(title = "Max Heart Rate by Disease Status")

# Arrange histograms and boxplots in one figure
grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3, nrow = 2,
             top = "EDA: Histograms and Boxplots")

# ------------------------- 4. Outlier Handling (IQR Capping) ----------------
# Identify numeric columns (exclude target)
numeric_cols <- sapply(heart, is.numeric) & !(names(heart) %in% "target")
for (col in names(heart)[numeric_cols]) {
  Q1 <- quantile(heart[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(heart[[col]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  # Cap (winsorize) outliers
  heart[[col]] <- ifelse(heart[[col]] < lower_bound, lower_bound, heart[[col]])
  heart[[col]] <- ifelse(heart[[col]] > upper_bound, upper_bound, heart[[col]])
}
cat("Outliers capped using IQR method.\n")

# ------------------------- 5. Train/Test Split & Standardization ------------
set.seed(42)  # reproducibility
trainIndex <- createDataPartition(heart$target, p = 0.7, list = FALSE)
train_data <- heart[trainIndex, ]
test_data  <- heart[-trainIndex, ]

# Standardize numeric predictors (center + scale) using training data parameters
preproc <- preProcess(train_data[, numeric_cols], method = c("center", "scale"))
train_data[, numeric_cols] <- predict(preproc, train_data[, numeric_cols])
test_data[, numeric_cols]  <- predict(preproc, test_data[, numeric_cols])

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# ------------------------- 6. Model Training & Tuning (GridSearchCV) --------
# Define cross-validation control (5-fold CV)
ctrl <- trainControl(method = "cv", number = 5, 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE,   # needed for ROC
                     savePredictions = TRUE)

# ----- Logistic Regression (no hyperparameters, but included for comparison) -----
set.seed(42)
lr_model <- train(target ~ ., data = train_data, method = "glm", 
                  family = "binomial", trControl = ctrl)

# ----- K-Nearest Neighbors (tune k from 3 to 20) -----
knn_grid <- expand.grid(k = seq(3, 20, by = 2))
set.seed(42)
knn_model <- train(target ~ ., data = train_data, method = "knn", 
                   tuneGrid = knn_grid, trControl = ctrl,
                   preProcess = NULL)   # already standardized

# ----- Random Forest (tune mtry = sqrt(p) +/- ) -----
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
set.seed(42)
rf_model <- train(target ~ ., data = train_data, method = "rf", 
                  tuneGrid = rf_grid, trControl = ctrl,
                  ntree = 500, importance = TRUE)

# Compare models using resamples
model_list <- list(Logistic = lr_model, KNN = knn_model, RF = rf_model)
resamps <- resamples(model_list)
summary(resamps)
# Plot cross-validation performance comparison
bwplot(resamps, metric = "ROC")

# ------------------------- 7. Evaluation on Test Set ------------------------
# Predict class and probabilities for each model
lr_pred <- predict(lr_model, newdata = test_data)
knn_pred <- predict(knn_model, newdata = test_data)
rf_pred <- predict(rf_model, newdata = test_data)

# Probabilities for ROC
lr_prob <- predict(lr_model, newdata = test_data, type = "prob")[, "1"]
knn_prob <- predict(knn_model, newdata = test_data, type = "prob")[, "1"]
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

# Confusion matrix & accuracy
confusionMatrix(lr_pred, test_data$target)
confusionMatrix(knn_pred, test_data$target)
confusionMatrix(rf_pred, test_data$target)

# ------------------------- 8. ROC Curves & AUC ------------------------------
roc_lr <- roc(response = test_data$target, predictor = lr_prob, levels = rev(levels(test_data$target)))
roc_knn <- roc(test_data$target, knn_prob, levels = rev(levels(test_data$target)))
roc_rf <- roc(test_data$target, rf_prob, levels = rev(levels(test_data$target)))

# Plot ROC curves
plot(roc_lr, col = "blue", lwd = 2, main = "ROC Curves for Heart Disease Prediction")
plot(roc_knn, col = "red", lwd = 2, add = TRUE)
plot(roc_rf, col = "darkgreen", lwd = 2, add = TRUE)
legend("bottomright", legend = c(paste("Logistic (AUC =", round(auc(roc_lr), 3), ")"),
                                 paste("KNN (AUC =", round(auc(roc_knn), 3), ")"),
                                 paste("RF (AUC =", round(auc(roc_rf), 3), ")")),
       col = c("blue", "red", "darkgreen"), lwd = 2)

# Print AUC values
cat("\nTest Set AUC:\n")
cat("Logistic Regression:", auc(roc_lr), "\n")
cat("KNN:", auc(roc_knn), "\n")
cat("Random Forest:", auc(roc_rf), "\n")

# ------------------------- 9. Decision Boundary Visualization Using PCA -----
# Project all predictors onto first two principal components (on training data)
# We'll use the best model (e.g., Random Forest) to illustrate decision boundary

# Prepare data: keep only numeric predictors (exclude target)
train_X <- train_data[, numeric_cols]
test_X  <- test_data[, numeric_cols]

# Perform PCA on training data
pca_res <- prcomp(train_X, center = TRUE, scale. = TRUE)  # already scaled, but safe
train_pca <- as.data.frame(pca_res$x[, 1:2])  # first two PCs
train_pca$target <- train_data$target

# Transform test data using the same PCA rotation
test_pca <- as.data.frame(predict(pca_res, newdata = test_X)[, 1:2])
test_pca$target <- test_data$target

# Train a Random Forest on the 2D PCA features (for visualization only)
set.seed(42)
rf_pca <- randomForest(target ~ PC1 + PC2, data = train_pca, ntree = 200)

# Create a fine grid over the PC1-PC2 space
grid_pc1 <- seq(min(train_pca$PC1), max(train_pca$PC1), length = 100)
grid_pc2 <- seq(min(train_pca$PC2), max(train_pca$PC2), length = 100)
grid <- expand.grid(PC1 = grid_pc1, PC2 = grid_pc2)

# Predict class probabilities for each grid point
grid$prob <- predict(rf_pca, newdata = grid, type = "prob")[, "1"]

# Plot decision boundary (contour) + test points
ggplot() +
  geom_raster(data = grid, aes(x = PC1, y = PC2, fill = prob), alpha = 0.8) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5,
                       name = "Predicted\nProbability") +
  geom_point(data = test_pca, aes(x = PC1, y = PC2, color = target), size = 2) +
  scale_color_manual(values = c("0" = "darkblue", "1" = "darkred"), name = "True Class") +
  labs(title = "Decision Boundary of Random Forest (PCA Projection)",
       subtitle = "Background: predicted probability of heart disease. Points: test set.") +
  theme_minimal()

# Optional: For Logistic Regression or KNN boundary, replace rf_pca with:
#   glm_pca <- glm(target ~ PC1 + PC2, data = train_pca, family = binomial)
#   grid$prob <- predict(glm_pca, newdata = grid, type = "response")
# ----------------------------------------------------------------------------
cat("\nAnalysis complete. All plots have been displayed.\n")