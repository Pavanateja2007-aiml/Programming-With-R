# ==============================
# Titanic Survival Prediction
# Binary Classification in R
# ==============================

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(gbm)
library(pROC)
library(gridExtra)

# ------------------------------
# 1. Load Dataset
# ------------------------------
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df <- read.csv(url)
cat("Initial data shape:", dim(df), "\n")
head(df)

# ------------------------------
# 2. Handle Missing Values
# ------------------------------
# Age: fill with median grouped by Pclass and Sex
df <- df %>%
  group_by(Pclass, Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

# Embarked: fill with mode
df$Embarked[is.na(df$Embarked)] <- names(sort(table(df$Embarked), decreasing = TRUE))[1]

# Cabin: create new feature 'HasCabin' and extract deck letter
df$HasCabin <- as.integer(!is.na(df$Cabin))
df$Deck <- ifelse(is.na(df$Cabin), "U", substr(df$Cabin, 1, 1))

# Drop original Cabin and other irrelevant columns
df <- df %>% select(-Cabin, -PassengerId, -Ticket)

# ------------------------------
# 3. Feature Engineering
# ------------------------------
df$FamilySize <- df$SibSp + df$Parch + 1
df$IsAlone <- as.integer(df$FamilySize == 1)

# Extract Title from Name
get_title <- function(name) {
  title <- str_trim(str_split(name, ",")[[1]][2])
  title <- str_split(title, "\\.")[[1]][1]
  title <- str_trim(title)
  if (title %in% c("Mr", "Mrs", "Miss", "Master")) {
    return(title)
  } else if (title %in% c("Mme", "Ms")) {
    return("Mrs")
  } else if (title %in% c("Mlle", "Lady", "Dona")) {
    return("Miss")
  } else {
    return("Rare")
  }
}
df$Title <- sapply(df$Name, get_title)
df <- df %>% select(-Name)

# ------------------------------
# 4. Encode Categorical Variables
# ------------------------------
# Sex: label encode (female = 0, male = 1)
df$Sex <- ifelse(df$Sex == "female", 0, 1)

# One-hot encoding for Embarked, Title, Deck
dummies <- dummyVars(~ Embarked + Title + Deck, data = df)
dummy_mat <- predict(dummies, newdata = df)
df <- cbind(df, dummy_mat)
df <- df %>% select(-Embarked, -Title, -Deck)

# ------------------------------
# 5. Prepare Features and Target
# ------------------------------
X <- df %>% select(-Survived)
y <- df$Survived

# Split (stratified by y)
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Scale numeric features
numeric_cols <- c("Age", "Fare", "SibSp", "Parch", "FamilySize")
scaler <- preProcess(X_train[, numeric_cols], method = c("center", "scale"))
X_train[, numeric_cols] <- predict(scaler, X_train[, numeric_cols])
X_test[, numeric_cols]  <- predict(scaler, X_test[, numeric_cols])

# ------------------------------
# 6. Visualizations (saved as PNG)
# ------------------------------
# Re-load clean data for meaningful plots (before encoding but after imputation)
df_plot <- read.csv(url)
df_plot <- df_plot %>%
  group_by(Pclass, Sex) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()
df_plot$FamilySize <- df_plot$SibSp + df_plot$Parch + 1

p1 <- ggplot(df_plot, aes(x = Sex, y = Survived)) +
  stat_summary(fun = mean, geom = "bar", fill = "steelblue") +
  labs(title = "Survival Rate by Sex", y = "Survival Rate")

p2 <- ggplot(df_plot, aes(x = as.factor(Pclass), y = Survived)) +
  stat_summary(fun = mean, geom = "bar", fill = "darkorange") +
  labs(title = "Survival Rate by Passenger Class", x = "Pclass")

p3 <- ggplot(df_plot, aes(x = as.factor(Survived), y = Age, fill = as.factor(Survived))) +
  geom_violin(trim = FALSE) +
  labs(title = "Age Distribution by Survival", x = "Survived") +
  theme(legend.position = "none")

p4 <- ggplot(df_plot, aes(x = as.factor(FamilySize), y = Survived)) +
  stat_summary(fun = mean, geom = "bar", fill = "darkgreen") +
  labs(title = "Survival Rate by Family Size", x = "FamilySize")

png("titanic_visualizations.png", width = 1200, height = 1000, res = 150)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()

# ------------------------------
# 7. Train Models & Evaluate
# ------------------------------
models <- list(
  "Logistic Regression" = glm(y_train ~ ., data = as.data.frame(X_train), family = binomial),
  "Random Forest" = randomForest(x = X_train, y = as.factor(y_train), ntree = 100, seed = 42),
  "Gradient Boosting" = gbm(y_train ~ ., data = as.data.frame(X_train), distribution = "bernoulli",
                            n.trees = 100, interaction.depth = 3, shrinkage = 0.1, verbose = FALSE)
)

results <- data.frame(Model = character(), Accuracy = numeric(), Precision = numeric(),
                      Recall = numeric(), F1 = numeric(), ROC_AUC = numeric())

# Prepare ROC plot
roc_plot <- ggplot() + geom_abline(slope = 1, intercept = 0, linetype = "dashed")

for (name in names(models)) {
  model <- models[[name]]
  
  if (name == "Logistic Regression") {
    pred_prob <- predict(model, newdata = as.data.frame(X_test), type = "response")
    pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  } else if (name == "Random Forest") {
    pred_prob <- predict(model, newdata = X_test, type = "prob")[, 2]
    pred_class <- predict(model, newdata = X_test)
  } else { # Gradient Boosting
    pred_prob <- predict(model, newdata = as.data.frame(X_test), n.trees = 100, type = "response")
    pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  }
  
  acc <- confusionMatrix(as.factor(pred_class), as.factor(y_test), positive = "1")$overall["Accuracy"]
  prec <- posPredValue(as.factor(pred_class), as.factor(y_test), positive = "1")
  rec <- sensitivity(as.factor(pred_class), as.factor(y_test), positive = "1")
  f1 <- (2 * prec * rec) / (prec + rec)
  roc_val <- roc(y_test, pred_prob, quiet = TRUE)$auc
  
  results <- rbind(results, data.frame(Model = name, Accuracy = acc, Precision = prec,
                                       Recall = rec, F1 = f1, ROC_AUC = roc_val))
  
  # ROC curve data
  roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
  roc_df <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)
  roc_plot <- roc_plot +
    geom_line(data = roc_df, aes(x = fpr, y = tpr, color = name), linewidth = 1.2)
}

roc_plot <- roc_plot +
  labs(x = "False Positive Rate", y = "True Positive Rate", title = "ROC Curves") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

ggsave("roc_curves.png", roc_plot, width = 6, height = 5, dpi = 150)

# Print results table
cat("\n===== MODEL PERFORMANCE =====\n")
print(results)

# Confusion matrix for Random Forest (example)
rf_pred <- predict(models[["Random Forest"]], newdata = X_test)
cm <- confusionMatrix(as.factor(rf_pred), as.factor(y_test), positive = "1")
cat("\nConfusion Matrix (Random Forest):\n")
print(cm$table)

# ------------------------------
# 8. Feature Importance (Random Forest)
# ------------------------------
rf_imp <- importance(models[["Random Forest"]])
imp_df <- data.frame(Feature = rownames(rf_imp), Importance = rf_imp[, "MeanDecreaseGini"])
imp_df <- imp_df[order(-imp_df$Importance), ]

png("feature_importance.png", width = 800, height = 500, res = 120)
ggplot(imp_df[1:10, ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance", x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()
dev.off()

cat("\nAnalysis complete. Plots saved:\n")
cat("- titanic_visualizations.png\n- roc_curves.png\n- feature_importance.png\n")