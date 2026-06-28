


# ============================================================
# SECTION 0: INSTALL AND LOAD REQUIRED PACKAGES
# ============================================================

# Install packages if not already installed
required_packages <- c("tm", "e1071", "caret", "ggplot2", "wordcloud",
                        "RColorBrewer", "dplyr", "tidyr", "scales",
                        "reshape2", "SnowballC")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

# Load all libraries
library(tm)           # Text Mining: corpus creation, preprocessing
library(e1071)        # Naive Bayes classifier
library(caret)        # Confusion matrix and model evaluation
library(ggplot2)      # Data visualization
library(wordcloud)    # Word cloud visualization
library(RColorBrewer) # Color palettes
library(dplyr)        # Data manipulation
library(tidyr)        # Data reshaping
library(scales)       # Scale functions for ggplot2
library(reshape2)     # Melt/cast for heatmaps
library(SnowballC)    # Word stemming

cat("All packages loaded successfully!\n")


# ============================================================
# SECTION 1: LOAD AND INSPECT DATASET
# ============================================================

cat("\n========== SECTION 1: DATASET UNDERSTANDING ==========\n")

# Load the dataset
emails <- read.csv("C:\\Users\\My world\\OneDrive\\Desktop\\R\\emails.csv", stringsAsFactors = FALSE)

# --- 1.1 Basic Inspection ---
cat("\n--- Dataset Dimensions ---\n")
cat("Rows:", nrow(emails), "\n")
cat("Columns:", ncol(emails), "\n")

cat("\n--- Column Names ---\n")
print(names(emails))

cat("\n--- First 6 Rows ---\n")
print(head(emails))

cat("\n--- Dataset Structure ---\n")
str(emails)

cat("\n--- Summary Statistics ---\n")
print(summary(emails))

# --- 1.2 Column Explanation ---
cat("\n--- Column Descriptions ---\n")
cat("text : The full email content (subject + body). This is our predictor variable.\n")
cat("spam : Binary target label. 1 = Spam email, 0 = Ham (legitimate) email.\n")

# --- 1.3 Missing Values ---
cat("\n--- Missing Values per Column ---\n")
print(colSums(is.na(emails)))

# --- 1.4 Duplicate Records ---
cat("\n--- Duplicate Records ---\n")
cat("Number of duplicate rows:", sum(duplicated(emails)), "\n")

# --- 1.5 Data Cleaning ---
cat("\n--- Data Cleaning ---\n")

# Remove duplicate rows
emails <- emails[!duplicated(emails), ]
cat("Rows after removing duplicates:", nrow(emails), "\n")

# Remove rows with empty text
emails <- emails[nchar(trimws(emails$text)) > 0, ]
cat("Rows after removing empty text:", nrow(emails), "\n")

# Convert spam to factor
emails$spam <- factor(emails$spam, levels = c(0, 1), labels = c("Ham", "Spam"))

cat("\n--- Class Distribution After Cleaning ---\n")
print(table(emails$spam))
print(prop.table(table(emails$spam)))


# ============================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

cat("\n========== SECTION 2: EXPLORATORY DATA ANALYSIS ==========\n")

# --- 2.1 Class Distribution Bar Chart ---
class_df <- as.data.frame(table(emails$spam))
names(class_df) <- c("Label", "Count")
class_df$Percentage <- round(class_df$Count / sum(class_df$Count) * 100, 1)

p1 <- ggplot(class_df, aes(x = Label, y = Count, fill = Label)) +
  geom_bar(stat = "identity", width = 0.6, color = "white", size = 0.8) +
  geom_text(aes(label = paste0(Count, "\n(", Percentage, "%)")),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("Ham" = "#2ECC71", "Spam" = "#E74C3C")) +
  labs(title = "Email Class Distribution",
       subtitle = "Spam vs Ham (Non-Spam) Emails",
       x = "Email Category", y = "Number of Emails",
       fill = "Category") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        legend.position = "none") +
  ylim(0, max(class_df$Count) * 1.15)

ggsave("plot_01_class_distribution_bar.png", p1, width = 7, height = 5, dpi = 150)
cat("Saved: plot_01_class_distribution_bar.png\n")

# --- 2.2 Pie Chart ---
p2 <- ggplot(class_df, aes(x = "", y = Count, fill = Label)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 1.2) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(Label, "\n", Percentage, "%")),
            position = position_stack(vjust = 0.5),
            size = 5, fontface = "bold", color = "white") +
  scale_fill_manual(values = c("Ham" = "#2ECC71", "Spam" = "#E74C3C")) +
  labs(title = "Proportion of Spam vs Ham Emails",
       subtitle = "Email Spam Classification Dataset",
       fill = "Category") +
  theme_void(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"))

ggsave("plot_02_class_distribution_pie.png", p2, width = 6, height = 6, dpi = 150)
cat("Saved: plot_02_class_distribution_pie.png\n")

# --- 2.3 Email Text Length Distribution ---
emails$text_length <- nchar(emails$text)

p3 <- ggplot(emails, aes(x = text_length, fill = spam)) +
  geom_histogram(bins = 50, alpha = 0.75, color = "white", size = 0.2) +
  facet_wrap(~spam, scales = "free_y", labeller = labeller(spam = c("Ham" = "Ham Emails",
                                                                     "Spam" = "Spam Emails"))) +
  scale_fill_manual(values = c("Ham" = "#2ECC71", "Spam" = "#E74C3C")) +
  labs(title = "Distribution of Email Text Length",
       subtitle = "Comparing character length for Spam vs Ham",
       x = "Number of Characters", y = "Frequency",
       fill = "Category") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        legend.position = "none")

ggsave("plot_03_text_length_distribution.png", p3, width = 10, height = 5, dpi = 150)
cat("Saved: plot_03_text_length_distribution.png\n")

# --- 2.4 Word Frequency for Spam and Ham ---
get_top_words <- function(text_vec, n = 15) {
  words <- unlist(strsplit(tolower(paste(text_vec, collapse = " ")), "\\W+"))
  stop_words_list <- c(stopwords("en"), "subject", "re", "fw", "the", "a", "an",
                        "this", "that", "will", "can", "get", "just", "like",
                        "also", "from", "your", "you", "to", "of", "is", "in",
                        "for", "it", "be", "on", "with", "we", "are", "not",
                        "have", "has", "was", "as", "or", "but", "if", "all",
                        "so", "at", "do", "by", "our", "my", "s", "t", "don")
  words <- words[!words %in% stop_words_list & nchar(words) > 2]
  word_freq <- sort(table(words), decreasing = TRUE)[1:n]
  data.frame(word = names(word_freq), freq = as.integer(word_freq))
}

spam_words  <- get_top_words(emails$text[emails$spam == "Spam"])
ham_words   <- get_top_words(emails$text[emails$spam == "Ham"])

spam_words$category <- "Spam"
ham_words$category  <- "Ham"
all_words <- rbind(spam_words, ham_words)

p4 <- ggplot(all_words, aes(x = reorder(word, freq), y = freq, fill = category)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~category, scales = "free") +
  coord_flip() +
  scale_fill_manual(values = c("Ham" = "#27AE60", "Spam" = "#C0392B")) +
  labs(title = "Top 15 Most Frequent Words",
       subtitle = "Spam vs Ham — after removing stop words",
       x = "Word", y = "Frequency") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"))

ggsave("plot_04_word_frequency.png", p4, width = 12, height = 6, dpi = 150)
cat("Saved: plot_04_word_frequency.png\n")

# --- 2.5 Box Plot: Text Length by Class ---
p5 <- ggplot(emails, aes(x = spam, y = text_length, fill = spam)) +
  geom_boxplot(alpha = 0.8, outlier.colour = "gray60", outlier.size = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "white") +
  scale_fill_manual(values = c("Ham" = "#2ECC71", "Spam" = "#E74C3C")) +
  labs(title = "Email Length by Category",
       subtitle = "Box plot with mean marker (diamond)",
       x = "Email Category", y = "Text Length (Characters)",
       fill = "Category") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        legend.position = "none")

ggsave("plot_05_boxplot_text_length.png", p5, width = 7, height = 5, dpi = 150)
cat("Saved: plot_05_boxplot_text_length.png\n")

cat("\n--- EDA Summary ---\n")
cat("Ham emails:", sum(emails$spam == "Ham"),
    "| Spam emails:", sum(emails$spam == "Spam"), "\n")
cat("Average text length - Ham:", round(mean(emails$text_length[emails$spam == "Ham"])),
    "| Spam:", round(mean(emails$text_length[emails$spam == "Spam"])), "chars\n")


# ============================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================

cat("\n========== SECTION 3: DATA PREPROCESSING ==========\n")

# --- 3.1 Create Text Corpus ---
cat("Step 1: Creating text corpus...\n")
corpus <- VCorpus(VectorSource(emails$text))
cat("Corpus created with", length(corpus), "documents.\n")

# --- 3.2 Text Normalization ---
cat("Step 2: Converting to lowercase...\n")
corpus <- tm_map(corpus, content_transformer(tolower))

cat("Step 3: Removing URLs...\n")
remove_url <- function(x) gsub("http[s]?://\\S+|www\\.\\S+", " ", x)
corpus <- tm_map(corpus, content_transformer(remove_url))

cat("Step 4: Removing email addresses...\n")
corpus <- tm_map(corpus, content_transformer(
  function(x) gsub("[[:alnum:]._%+-]+@[[:alnum:].-]+\\.[a-zA-Z]{2,}", " ", x)))

cat("Step 5: Removing punctuation...\n")
corpus <- tm_map(corpus, removePunctuation)

cat("Step 6: Removing numbers...\n")
corpus <- tm_map(corpus, removeNumbers)

cat("Step 7: Removing stop words...\n")
corpus <- tm_map(corpus, removeWords, c(stopwords("english"),
                                         "subject", "re", "fw", "fwd",
                                         "will", "can", "get", "just",
                                         "also", "from", "your", "you"))

cat("Step 8: Applying word stemming...\n")
corpus <- tm_map(corpus, stemDocument)

cat("Step 9: Stripping extra whitespace...\n")
corpus <- tm_map(corpus, stripWhitespace)

cat("Preprocessing complete!\n")

# --- 3.3 Create Document-Term Matrix (TF-IDF) ---
cat("Step 10: Creating TF-IDF weighted Document-Term Matrix...\n")

dtm_tfidf <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting    = weightTfIdf,
    bounds       = list(global = c(5, Inf)),  # word must appear in >=5 docs
    wordLengths  = c(3, Inf)                  # words >= 3 chars
  )
)

cat("DTM dimensions (TF-IDF):", dim(dtm_tfidf)[1], "documents x",
    dim(dtm_tfidf)[2], "terms\n")

# Remove sparse terms (keep terms appearing in >= 0.5% of documents)
dtm_tfidf <- removeSparseTerms(dtm_tfidf, 0.995)
cat("After removing sparse terms:", dim(dtm_tfidf)[1], "x", dim(dtm_tfidf)[2], "\n")

# --- 3.4 Convert to Data Frame ---
cat("Step 11: Converting DTM to data frame...\n")
emails_df <- as.data.frame(as.matrix(dtm_tfidf))
emails_df$label <- emails$spam
cat("Final feature matrix:", nrow(emails_df), "rows x", ncol(emails_df) - 1, "features (+label)\n")


# ============================================================
# SECTION 4: TRAIN-TEST SPLIT
# ============================================================

cat("\n========== SECTION 4: TRAIN-TEST SPLIT ==========\n")

set.seed(42)  # Reproducible seed

train_idx <- createDataPartition(emails_df$label, p = 0.80, list = FALSE)
train_data <- emails_df[train_idx, ]
test_data  <- emails_df[-train_idx, ]

train_labels <- train_data$label
test_labels  <- test_data$label
train_data$label <- NULL
test_data$label  <- NULL

cat("Total samples  :", nrow(emails_df), "\n")
cat("Training set   :", nrow(train_data), "rows (80%)\n")
cat("Testing set    :", nrow(test_data),  "rows (20%)\n")

cat("\n--- Class Distribution in Training Set ---\n")
print(table(train_labels))
cat("\n--- Class Distribution in Testing Set ---\n")
print(table(test_labels))


# ============================================================
# SECTION 5: NAIVE BAYES MODEL
# ============================================================

cat("\n========== SECTION 5: NAIVE BAYES MODEL ==========\n")

cat("
ALGORITHM OVERVIEW - NAIVE BAYES:
----------------------------------
Naive Bayes is a probabilistic supervised learning algorithm based on Bayes' Theorem.
It classifies by computing the posterior probability of each class and picks the highest.

Bayes' Theorem:
  P(Class | Features) = P(Features | Class) * P(Class) / P(Features)

'Naive' assumption: all features are conditionally independent given the class label.

For text classification:
  P(Spam | w1, w2, ..., wn) ∝ P(Spam) * ∏ P(wi | Spam)

Why Supervised?
  The model LEARNS from labelled training data (spam=1, ham=0) to map input
  features (word frequencies/TF-IDF) to output classes.\n")

cat("Training Naive Bayes model...\n")

# Train model with Laplace smoothing (prevents zero probabilities)
nb_model <- naiveBayes(
  x          = train_data,
  y          = train_labels,
  laplace    = 1       # Laplace smoothing parameter
)

cat("Model trained successfully!\n")
cat("Classes:", levels(train_labels), "\n")

# Predictions on test set
cat("Predicting on test set...\n")
predictions <- predict(nb_model, newdata = test_data)

cat("Prediction complete. Sample predictions:\n")
print(head(data.frame(Actual = test_labels, Predicted = predictions), 10))


# ============================================================
# SECTION 6: MODEL EVALUATION
# ============================================================

cat("\n========== SECTION 6: MODEL EVALUATION ==========\n")

# --- 6.1 Confusion Matrix ---
cm <- confusionMatrix(predictions, test_labels, positive = "Spam")
print(cm)

# --- 6.2 Extract Metrics ---
accuracy  <- cm$overall["Accuracy"]
precision <- cm$byClass["Precision"]
recall    <- cm$byClass["Recall"]
f1_score  <- cm$byClass["F1"]
specificity <- cm$byClass["Specificity"]

cat("\n============ PERFORMANCE METRICS SUMMARY ============\n")
cat(sprintf("Accuracy    : %.4f (%.2f%%)\n", accuracy,  accuracy  * 100))
cat(sprintf("Precision   : %.4f (%.2f%%)\n", precision, precision * 100))
cat(sprintf("Recall      : %.4f (%.2f%%)\n", recall,    recall    * 100))
cat(sprintf("F1 Score    : %.4f (%.2f%%)\n", f1_score,  f1_score  * 100))
cat(sprintf("Specificity : %.4f (%.2f%%)\n", specificity, specificity * 100))
cat("=====================================================\n")

cat("
METRIC INTERPRETATIONS:
-----------------------
Accuracy    : % of all emails correctly classified (Spam + Ham)
Precision   : Of all predicted Spam, how many were truly Spam?
Recall      : Of all actual Spam, how many did we catch?
F1 Score    : Harmonic mean of Precision and Recall (balanced metric)
Specificity : Of all actual Ham, how many did we correctly identify?
")


# ============================================================
# SECTION 7: RESULT VISUALIZATIONS
# ============================================================

cat("\n========== SECTION 7: VISUALIZATIONS ==========\n")

# --- 7.1 Confusion Matrix Heatmap ---
cm_table <- as.data.frame(cm$table)
names(cm_table) <- c("Predicted", "Actual", "Count")

# Add percentage labels
total <- sum(cm_table$Count)
cm_table$Pct <- paste0(round(cm_table$Count / total * 100, 1), "%")

p6 <- ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = paste0(Count, "\n", Pct)),
            size = 6, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#AED6F1", high = "#1A5276") +
  labs(title = "Confusion Matrix Heatmap",
       subtitle = "Naive Bayes — Email Spam Classification",
       x = "Actual Label", y = "Predicted Label",
       fill = "Count") +
  theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5, size = 14),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        axis.text     = element_text(size = 12, face = "bold"))

ggsave("plot_06_confusion_matrix.png", p6, width = 6, height = 5, dpi = 150)
cat("Saved: plot_06_confusion_matrix.png\n")

# --- 7.2 Performance Metrics Bar Chart ---
metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "Specificity"),
  Value  = c(as.numeric(accuracy), as.numeric(precision),
             as.numeric(recall),   as.numeric(f1_score),
             as.numeric(specificity))
)
metrics_df$Label <- paste0(round(metrics_df$Value * 100, 1), "%")
metrics_df$Color <- ifelse(metrics_df$Value >= 0.90, "#27AE60",
                    ifelse(metrics_df$Value >= 0.75, "#F39C12", "#E74C3C"))

p7 <- ggplot(metrics_df, aes(x = reorder(Metric, Value), y = Value, fill = Metric)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = Label), hjust = -0.15, fontface = "bold", size = 4.5) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1.12), labels = percent_format()) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Metrics",
       subtitle = "Naive Bayes Classifier Evaluation",
       x = "Metric", y = "Score") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"))

ggsave("plot_07_performance_metrics.png", p7, width = 8, height = 5, dpi = 150)
cat("Saved: plot_07_performance_metrics.png\n")

# --- 7.3 Prediction Result Comparison ---
pred_df <- data.frame(
  Type  = c("Actual Ham", "Actual Spam",
            "Predicted Ham", "Predicted Spam"),
  Count = c(sum(test_labels == "Ham"), sum(test_labels == "Spam"),
            sum(predictions == "Ham"), sum(predictions == "Spam")),
  Group = c("Actual", "Actual", "Predicted", "Predicted")
)

p8 <- ggplot(pred_df, aes(x = Type, y = Count, fill = Group)) +
  geom_col(width = 0.6, color = "white") +
  geom_text(aes(label = Count), vjust = -0.5, fontface = "bold", size = 4.5) +
  scale_fill_manual(values = c("Actual" = "#2980B9", "Predicted" = "#8E44AD")) +
  labs(title = "Actual vs Predicted Class Distribution",
       subtitle = "Test Set — Naive Bayes",
       x = "", y = "Count", fill = "Type") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        axis.text.x   = element_text(angle = 15, hjust = 1))

ggsave("plot_08_actual_vs_predicted.png", p8, width = 8, height = 5, dpi = 150)
cat("Saved: plot_08_actual_vs_predicted.png\n")

# --- 7.4 Train vs Test Split Visualization ---
split_df <- data.frame(
  Set     = c("Training (80%)", "Testing (20%)"),
  Count   = c(nrow(train_data), nrow(test_data)),
  Ham     = c(sum(train_labels == "Ham"), sum(test_labels == "Ham")),
  Spam    = c(sum(train_labels == "Spam"), sum(test_labels == "Spam"))
)
split_long <- melt(split_df[, c("Set", "Ham", "Spam")], id.vars = "Set",
                   variable.name = "Class", value.name = "Count")

p9 <- ggplot(split_long, aes(x = Set, y = Count, fill = Class)) +
  geom_col(position = "dodge", width = 0.5, color = "white") +
  geom_text(aes(label = Count), position = position_dodge(width = 0.5),
            vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = c("Ham" = "#2ECC71", "Spam" = "#E74C3C")) +
  labs(title = "Train-Test Split: Class Balance",
       subtitle = "80/20 Stratified Split",
       x = "Dataset Split", y = "Number of Samples", fill = "Class") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"))

ggsave("plot_09_train_test_split.png", p9, width = 8, height = 5, dpi = 150)
cat("Saved: plot_09_train_test_split.png\n")


# ============================================================
# SECTION 8: FINAL SUMMARY
# ============================================================

cat("\n========== FINAL PROJECT SUMMARY ==========\n")
cat(sprintf(
"
Project   : Email Spam Classification using Naive Bayes
Dataset   : %d emails (%d Ham, %d Spam)
Features  : %d TF-IDF weighted terms
Split     : 80%% Train (%d) / 20%% Test (%d)
Algorithm : Naive Bayes with Laplace Smoothing (laplace=1)

--- Final Model Performance ---
Accuracy    : %.2f%%
Precision   : %.2f%%
Recall      : %.2f%%
F1 Score    : %.2f%%
Specificity : %.2f%%

--- Output Files Generated ---
 R Script        : naive_bayes_spam.R
 Plots (9 total) : plot_01 to plot_09 (PNG files)
============================================\n",
  nrow(emails), sum(emails$spam == "Ham"), sum(emails$spam == "Spam"),
  ncol(train_data),
  nrow(train_data), nrow(test_data),
  accuracy * 100, precision * 100, recall * 100, f1_score * 100, specificity * 100
))
