# ============================================================================
# PROJECT: House Price Analysis
#   Part A — K-Means Clustering  (Unsupervised Learning)
#   Part B — Multiple Linear Regression (Supervised  Learning)
# Language : R
# Dataset  : house_prices.csv
# ============================================================================

# ============================================================================
# SECTION 1: INSTALL AND LOAD REQUIRED PACKAGES
# ============================================================================

# Uncomment to install (run once):
# install.packages(c("ggplot2","dplyr","cluster","factoextra","corrplot",
#                    "caret","reshape2","scales","gridExtra","GGally",
# "tidyr","ggcorrplot"))



library(ggplot2)     # Visualisation
library(dplyr)       # Data wrangling
library(cluster)     # K-means & silhouette
library(factoextra)  # Cluster visualisation helpers
library(corrplot)    # Correlation matrix plot
library(caret)       # Pre-processing & evaluation
library(reshape2)    # Data reshaping (melt)
library(scales)      # Axis label formatting
library(gridExtra)   # Multi-panel plots
library(GGally)      # Pair-plot matrix
library(tidyr)       # Pivoting helpers
library(ggcorrplot)  # ggplot2-based correlation plot

cat("=== All libraries loaded successfully ===\n\n")

# ============================================================================
# SECTION 2: LOAD AND INSPECT THE DATASET
# ============================================================================

df <- read.csv("house_prices.csv", stringsAsFactors = FALSE)

cat("=== DATASET OVERVIEW ===\n")
cat("Dimensions (rows x cols):", nrow(df), "x", ncol(df), "\n\n")

cat("Column names:\n")
print(colnames(df))

cat("\nFirst 6 rows:\n")
print(head(df))

cat("\nData structure:\n")
str(df)

cat("\nSummary statistics:\n")
print(summary(df))

# ----------------------------------------------------------------------------
# COLUMN DICTIONARY
# ----------------------------------------------------------------------------
# area_sqft   : Total floor area of the house in square feet  (numeric)
# bedrooms    : Number of bedrooms                            (integer 1-5)
# bathrooms   : Number of bathrooms                           (integer 1-3)
# age_years   : Age of the house in years                     (integer 1-49)
# garage      : Binary — 1 if garage present, 0 otherwise     (binary)
# floors      : Number of floors                              (integer 1-3)
# garden      : Binary — 1 if garden present, 0 otherwise     (binary)
# distance_km : Distance to city centre in kilometres         (numeric)
# price       : Sale price of the house in USD               (target, numeric)

# ============================================================================
# SECTION 3: MISSING VALUES AND DUPLICATE DETECTION
# ============================================================================

cat("\n=== MISSING VALUES ===\n")
print(colSums(is.na(df)))

cat("\n=== DUPLICATE RECORDS ===\n")
cat("Number of duplicate rows:", sum(duplicated(df)), "\n")

# Clean (already clean, but applied for robustness)
df <- df[!duplicated(df), ]
df <- df[complete.cases(df), ]
cat("Cleaned dataset size:", nrow(df), "rows x", ncol(df), "cols\n")

# ============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# ── 4.1  Price Distribution ─────────────────────────────────────────────────
p1 <- ggplot(df, aes(x = price)) +
  geom_histogram(bins = 30, fill = "#1565C0", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(df$price), linetype = "dashed",
             color = "#F44336", linewidth = 1) +
  annotate("text", x = mean(df$price) * 1.05, y = 45,
           label = paste0("Mean: $", comma(round(mean(df$price)))),
           color = "#F44336", size = 3.8, fontface = "bold") +
  scale_x_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title = "House Price Distribution",
       subtitle = "Red dashed line = Mean price",
       x = "Sale Price (USD)", y = "Count") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_01_price_distribution.png", p1, width = 9, height = 6, dpi = 150)
print(p1)
cat("[Saved] plot_01_price_distribution.png\n")

# ── 4.2  Area vs Price (scatter) ────────────────────────────────────────────
p2 <- ggplot(df, aes(x = area_sqft, y = price, color = as.factor(bedrooms))) +
  geom_point(alpha = 0.65, size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "#212121", linewidth = 0.8) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  scale_x_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Area vs House Price",
       subtitle = "Coloured by number of bedrooms",
       x = "Area (sq ft)", y = "Sale Price",
       color = "Bedrooms") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_02_area_vs_price.png", p2, width = 9, height = 6, dpi = 150)
print(p2)
cat("[Saved] plot_02_area_vs_price.png\n")

# ── 4.3  Price by Bedrooms (box plot) ───────────────────────────────────────
p3 <- ggplot(df, aes(x = as.factor(bedrooms), y = price,
                     fill = as.factor(bedrooms))) +
  geom_boxplot(alpha = 0.8, outlier.color = "gray50", outlier.size = 1.5) +
  scale_fill_brewer(palette = "Blues") +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title = "House Price by Number of Bedrooms",
       x = "Bedrooms", y = "Sale Price", fill = "Bedrooms") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

ggsave("plot_03_price_by_bedrooms.png", p3, width = 8, height = 6, dpi = 150)
print(p3)
cat("[Saved] plot_03_price_by_bedrooms.png\n")

# ── 4.4  Garage / Garden bar chart ──────────────────────────────────────────
garage_df <- df %>%
  mutate(Garage = ifelse(garage == 1, "Has Garage", "No Garage"),
         Garden = ifelse(garden == 1, "Has Garden", "No Garden")) %>%
  group_by(Garage, Garden) %>%
  summarise(Mean_Price = mean(price), Count = n(), .groups = "drop")

p4 <- ggplot(garage_df, aes(x = Garage, y = Mean_Price, fill = Garden)) +
  geom_col(position = "dodge", color = "white", width = 0.6) +
  geom_text(aes(label = paste0("n=", Count)),
            position = position_dodge(0.6), vjust = -0.4, size = 3.5) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  scale_fill_manual(values = c("Has Garden" = "#43A047", "No Garden" = "#EF9A9A")) +
  labs(title = "Mean House Price by Garage and Garden",
       x = "Garage Status", y = "Mean Sale Price", fill = "Garden") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("plot_04_garage_garden_price.png", p4, width = 9, height = 6, dpi = 150)
print(p4)
cat("[Saved] plot_04_garage_garden_price.png\n")

# ── 4.5  Correlation Heatmap ─────────────────────────────────────────────────
cor_mat <- round(cor(df), 3)
p5 <- ggcorrplot(cor_mat,
                 method = "square",
                 type   = "lower",
                 lab    = TRUE,
                 lab_size = 3.2,
                 colors = c("#B71C1C", "white", "#0D47A1"),
                 title  = "Correlation Matrix — house_prices.csv",
                 ggtheme = theme_minimal(base_size = 11))

ggsave("plot_05_correlation_heatmap.png", p5, width = 9, height = 8, dpi = 150)
print(p5)
cat("[Saved] plot_05_correlation_heatmap.png\n")

# ── 4.6  Pair plot (key variables) ───────────────────────────────────────────
p6 <- ggpairs(df[, c("area_sqft","bedrooms","age_years","distance_km","price")],
              title = "Pair Plot — Key Variables vs Price",
              upper = list(continuous = wrap("cor", size = 3)),
              lower = list(continuous = wrap("points", alpha = 0.4, size = 0.8)),
              diag  = list(continuous = wrap("densityDiag", fill = "#90CAF9")))

ggsave("plot_06_pairplot.png", p6, width = 10, height = 9, dpi = 150)
print(p6)
cat("[Saved] plot_06_pairplot.png\n")

# ── 4.7  Age vs Price ────────────────────────────────────────────────────────
p7 <- ggplot(df, aes(x = age_years, y = price)) +
  geom_point(alpha = 0.5, color = "#5C6BC0", size = 1.8) +
  geom_smooth(method = "loess", se = TRUE, color = "#E53935", linewidth = 1) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title = "House Age vs Sale Price",
       x = "Age (years)", y = "Sale Price") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("plot_07_age_vs_price.png", p7, width = 8, height = 6, dpi = 150)
print(p7)
cat("[Saved] plot_07_age_vs_price.png\n")

# ── 4.8  Distance vs Price ────────────────────────────────────────────────────
p8 <- ggplot(df, aes(x = distance_km, y = price)) +
  geom_point(alpha = 0.5, color = "#26A69A", size = 1.8) +
  geom_smooth(method = "lm", se = TRUE, color = "#BF360C", linewidth = 1) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title = "Distance to City Centre vs Sale Price",
       x = "Distance (km)", y = "Sale Price") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("plot_08_distance_vs_price.png", p8, width = 8, height = 6, dpi = 150)
print(p8)
cat("[Saved] plot_08_distance_vs_price.png\n")

# ── 4.9  Pie chart: Floors distribution ──────────────────────────────────────
floors_df <- df %>%
  count(floors) %>%
  mutate(pct   = n / sum(n),
         label = paste0(floors, " floor(s)\n", round(pct * 100, 1), "%"))

p9 <- ggplot(floors_df, aes(x = "", y = n, fill = as.factor(floors))) +
  geom_col(width = 1, color = "white", linewidth = 1) +
  coord_polar("y") +
  scale_fill_brewer(palette = "Set2") +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 4, fontface = "bold", color = "white") +
  labs(title = "House Floors Distribution",
       fill = "Floors") +
  theme_void(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
        legend.position = "bottom")

ggsave("plot_09_floors_pie.png", p9, width = 7, height = 7, dpi = 150)
print(p9)
cat("[Saved] plot_09_floors_pie.png\n")

# ============================================================================
# PART A — UNSUPERVISED LEARNING: K-MEANS CLUSTERING
# ============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS K-MEANS CLUSTERING?
#   K-Means is an iterative unsupervised learning algorithm that partitions
#   n observations into k clusters, where each observation belongs to the
#   cluster whose centroid (mean) is closest.
#
# ALGORITHM STEPS:
#   1. Randomly initialise k centroids.
#   2. Assign each point to the nearest centroid (Euclidean distance).
#   3. Recompute centroid as mean of all assigned points.
#   4. Repeat steps 2-3 until centroids stabilise (convergence).
#
# OBJECTIVE FUNCTION (Within-Cluster Sum of Squares – WCSS):
#   Minimise:  Σ_k Σ_{x∈Ck} ||x − μ_k||²
#   where μ_k is the centroid of cluster k.
#
# WHY UNSUPERVISED?
#   No labels are provided. The algorithm discovers hidden groupings
#   purely from the feature structure of the data.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n\n=== PART A: K-MEANS CLUSTERING (UNSUPERVISED LEARNING) ===\n")

# ── A.1  Feature selection and scaling ──────────────────────────────────────
# Use the most informative numeric features for clustering
cluster_features <- df[, c("area_sqft", "bedrooms", "bathrooms",
                            "age_years", "distance_km", "price")]

# Standardise: (x - mean) / sd  — critical for K-means (distance-based)
cluster_scaled <- scale(cluster_features)
cat("Features selected for clustering:", colnames(cluster_scaled), "\n")
cat("Scaling applied: zero mean, unit variance\n")

# ── A.2  Elbow method: determine optimal k ──────────────────────────────────
set.seed(42)
wcss <- sapply(1:10, function(k) {
  kmeans(cluster_scaled, centers = k, nstart = 25, iter.max = 300)$tot.withinss
})

elbow_df <- data.frame(k = 1:10, WCSS = wcss)

p10 <- ggplot(elbow_df, aes(x = k, y = WCSS)) +
  geom_line(color = "#1565C0", linewidth = 1.2) +
  geom_point(color = "#1565C0", size = 3.5) +
  geom_vline(xintercept = 3, linetype = "dashed",
             color = "#F44336", linewidth = 1) +
  annotate("text", x = 3.3, y = max(wcss) * 0.9,
           label = "Optimal k = 3", color = "#F44336",
           size = 4, fontface = "bold") +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "Elbow Method — Optimal Number of Clusters",
       subtitle = "Look for the 'elbow' where WCSS decrease slows",
       x = "Number of Clusters (k)", y = "Within-Cluster Sum of Squares (WCSS)") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_10_elbow_method.png", p10, width = 9, height = 6, dpi = 150)
print(p10)
cat("[Saved] plot_10_elbow_method.png\n")

# ── A.3  Silhouette analysis ──────────────────────────────────────────────────
sil_scores <- sapply(2:8, function(k) {
  km  <- kmeans(cluster_scaled, centers = k, nstart = 25, iter.max = 300)
  sil <- silhouette(km$cluster, dist(cluster_scaled))
  mean(sil[, 3])
})

sil_df <- data.frame(k = 2:8, Silhouette = sil_scores)
best_k_sil <- sil_df$k[which.max(sil_df$Silhouette)]

p11 <- ggplot(sil_df, aes(x = k, y = Silhouette)) +
  geom_line(color = "#6A1B9A", linewidth = 1.2) +
  geom_point(color = "#6A1B9A", size = 3.5) +
  geom_vline(xintercept = best_k_sil, linetype = "dashed",
             color = "#F44336", linewidth = 1) +
  scale_x_continuous(breaks = 2:8) +
  labs(title = "Silhouette Score vs Number of Clusters",
       subtitle = paste("Best k =", best_k_sil, "(highest silhouette)"),
       x = "Number of Clusters (k)", y = "Average Silhouette Score") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_11_silhouette_scores.png", p11, width = 9, height = 6, dpi = 150)
print(p11)
cat("[Saved] plot_11_silhouette_scores.png\n")

# ── A.4  Fit final K-Means model (k = 3) ─────────────────────────────────────
set.seed(42)
k_final  <- 3
km_model <- kmeans(cluster_scaled, centers = k_final,
                   nstart = 50, iter.max = 500)

df$Cluster <- as.factor(km_model$cluster)

cat("\n=== K-MEANS RESULTS (k =", k_final, ") ===\n")
cat("Cluster sizes:\n"); print(table(df$Cluster))
cat("\nWithin-cluster sum of squares:\n"); print(km_model$withinss)
cat("Total WCSS           :", km_model$tot.withinss, "\n")
cat("Between SS / Total SS:", round(km_model$betweenss / km_model$totss * 100, 2), "%\n")

# ── A.5  Cluster Profiles ─────────────────────────────────────────────────────
cat("\n=== CLUSTER PROFILES (mean values per cluster) ===\n")
cluster_profile <- df %>%
  group_by(Cluster) %>%
  summarise(across(c(area_sqft, bedrooms, bathrooms, age_years,
                     distance_km, price), mean, .names = "avg_{.col}"),
            Count = n())
print(cluster_profile)

# Label clusters based on price
price_order  <- order(cluster_profile$avg_price)
cluster_labels <- c("Budget Homes", "Mid-Range Homes", "Premium Homes")
label_map <- setNames(cluster_labels, as.character(price_order))
df$Cluster_Label <- label_map[as.character(as.integer(df$Cluster))]
df$Cluster_Label <- factor(df$Cluster_Label,
                           levels = c("Budget Homes","Mid-Range Homes","Premium Homes"))

# ── A.6  Cluster Visualisations ───────────────────────────────────────────────

# 2-D PCA scatter plot via factoextra
p12 <- fviz_cluster(km_model, data = cluster_scaled,
                    geom = "point", ellipse.type = "convex",
                    palette = c("#1565C0","#F57F17","#B71C1C"),
                    ggtheme = theme_minimal(base_size = 12),
                    main = "K-Means Clusters (PCA 2D Projection)")

ggsave("plot_12_cluster_pca.png", p12, width = 9, height = 7, dpi = 150)
print(p12)
cat("[Saved] plot_12_cluster_pca.png\n")

# Area vs Price coloured by cluster
p13 <- ggplot(df, aes(x = area_sqft, y = price,
                      color = Cluster_Label, shape = Cluster_Label)) +
  geom_point(alpha = 0.7, size = 2.2) +
  scale_color_manual(values = c("Budget Homes"   = "#1565C0",
                                 "Mid-Range Homes" = "#F57F17",
                                 "Premium Homes"   = "#B71C1C")) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  scale_x_continuous(labels = comma) +
  labs(title  = "K-Means Clusters: Area vs Price",
       subtitle = "Each cluster represents a distinct market segment",
       x = "Area (sq ft)", y = "Sale Price",
       color = "Cluster", shape = "Cluster") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        legend.position = "bottom")

ggsave("plot_13_cluster_area_price.png", p13, width = 9, height = 6, dpi = 150)
print(p13)
cat("[Saved] plot_13_cluster_area_price.png\n")

# Box plot: price per cluster
p14 <- ggplot(df, aes(x = Cluster_Label, y = price, fill = Cluster_Label)) +
  geom_boxplot(alpha = 0.85, outlier.size = 1.5, outlier.color = "gray60") +
  scale_fill_manual(values = c("Budget Homes"   = "#90CAF9",
                                "Mid-Range Homes" = "#FFE082",
                                "Premium Homes"   = "#EF9A9A")) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title = "Price Distribution per Cluster",
       x = "Market Segment", y = "Sale Price", fill = "Segment") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

ggsave("plot_14_cluster_price_box.png", p14, width = 9, height = 6, dpi = 150)
print(p14)
cat("[Saved] plot_14_cluster_price_box.png\n")

# Radar / spider chart — cluster centroid profiles (via reshaped data)
centroid_df <- cluster_profile %>%
  select(Cluster, avg_area_sqft, avg_bedrooms,
         avg_bathrooms, avg_age_years, avg_distance_km) %>%
  mutate(across(-Cluster, ~ as.numeric(scale(.)))) %>%
  pivot_longer(-Cluster, names_to = "Feature", values_to = "Z_Score") %>%
  mutate(Feature = gsub("avg_", "", Feature))

p15 <- ggplot(centroid_df, aes(x = Feature, y = Z_Score,
                               group = Cluster, color = Cluster, fill = Cluster)) +
  geom_polygon(alpha = 0.15, linewidth = 1) +
  geom_point(size = 3) +
  coord_polar() +
  scale_color_manual(values = c("1"="#1565C0","2"="#F57F17","3"="#B71C1C"),
                     labels = c("Budget","Mid-Range","Premium")) +
  scale_fill_manual(values  = c("1"="#1565C0","2"="#F57F17","3"="#B71C1C"),
                     labels = c("Budget","Mid-Range","Premium")) +
  labs(title    = "Cluster Centroid Profiles (Standardised)",
       subtitle = "Each axis represents a standardised feature mean",
       color = "Cluster", fill = "Cluster") +
  theme_minimal(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        axis.text.x   = element_text(size = 10, face = "bold"))

ggsave("plot_15_cluster_radar.png", p15, width = 8, height = 8, dpi = 150)
print(p15)
cat("[Saved] plot_15_cluster_radar.png\n")

# ── A.7  Silhouette plot for chosen k ─────────────────────────────────────────
sil_final <- silhouette(km_model$cluster, dist(cluster_scaled))
cat("\nMean Silhouette Score (k=3):", round(mean(sil_final[,3]), 4), "\n")

# ============================================================================
# PART B — SUPERVISED LEARNING: MULTIPLE LINEAR REGRESSION
# ============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# Multiple Linear Regression models the relationship between a continuous
# target variable (price) and multiple predictor variables:
#
#   ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
#
# Estimation method: Ordinary Least Squares (OLS)
#   Minimise: Σ(yᵢ − ŷᵢ)²   (Residual Sum of Squares)
# ─────────────────────────────────────────────────────────────────────────────

cat("\n\n=== PART B: MULTIPLE LINEAR REGRESSION (SUPERVISED LEARNING) ===\n")

# ── B.1  Train / Test Split (80 / 20) ────────────────────────────────────────
set.seed(42)
train_idx <- createDataPartition(df$price, p = 0.80, list = FALSE)
train_df  <- df[train_idx, ]
test_df   <- df[-train_idx, ]

cat("Training samples:", nrow(train_df), "\n")
cat("Testing  samples:", nrow(test_df),  "\n")

# ── B.2  Feature engineering: add Cluster as predictor ───────────────────────
# Use core numeric features + cluster membership
model_cols <- c("area_sqft","bedrooms","bathrooms","age_years",
                "garage","floors","garden","distance_km","price")

train_model <- train_df[, model_cols]
test_model  <- test_df[,  model_cols]

# ── B.3  Train model ──────────────────────────────────────────────────────────
lm_model <- lm(price ~ ., data = train_model)

cat("\n=== LINEAR REGRESSION SUMMARY ===\n")
print(summary(lm_model))

# ── B.4  Predictions ──────────────────────────────────────────────────────────
predictions_lm <- predict(lm_model, newdata = test_model)

# ── B.5  Regression Evaluation Metrics ───────────────────────────────────────
actual <- test_model$price
resid  <- actual - predictions_lm

MAE    <- mean(abs(resid))
MSE    <- mean(resid^2)
RMSE   <- sqrt(MSE)
SS_res <- sum(resid^2)
SS_tot <- sum((actual - mean(actual))^2)
R2     <- 1 - SS_res / SS_tot
Adj_R2 <- summary(lm_model)$adj.r.squared
MAPE   <- mean(abs(resid / actual)) * 100

cat("\n=== REGRESSION PERFORMANCE METRICS ===\n")
cat(sprintf("MAE          : $%s\n",    comma(round(MAE))))
cat(sprintf("MSE          : $%s\n",    comma(round(MSE)))  )
cat(sprintf("RMSE         : $%s\n",    comma(round(RMSE))))
cat(sprintf("R²           : %.4f  (%.2f%% variance explained)\n", R2, R2*100))
cat(sprintf("Adjusted R²  : %.4f\n",   Adj_R2))
cat(sprintf("MAPE         : %.2f%%\n", MAPE))

# ── B.6  Regression Visualisations ───────────────────────────────────────────

results_df <- data.frame(Actual = actual, Predicted = predictions_lm,
                         Residual = resid)

# Actual vs Predicted
p16 <- ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "#1565C0", size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "#F44336",
              linewidth = 1.2, linetype = "dashed") +
  scale_x_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title    = "Actual vs Predicted House Prices",
       subtitle = "Points near the red line = accurate predictions",
       x = "Actual Price", y = "Predicted Price") +
  annotate("text", x = max(actual)*0.2, y = max(predictions_lm)*0.92,
           label = paste0("R² = ", round(R2, 4)),
           color = "#1565C0", size = 4.5, fontface = "bold") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_16_actual_vs_predicted.png", p16, width = 8, height = 7, dpi = 150)
print(p16)
cat("[Saved] plot_16_actual_vs_predicted.png\n")

# Residuals distribution
p17 <- ggplot(results_df, aes(x = Residual)) +
  geom_histogram(bins = 30, fill = "#7B1FA2", color = "white", alpha = 0.8) +
  geom_vline(xintercept = 0, color = "#F44336", linewidth = 1.2, linetype = "dashed") +
  scale_x_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title    = "Residual Distribution",
       subtitle = "Residuals should be normally distributed around 0",
       x = "Residual (Actual − Predicted)", y = "Count") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_17_residuals_distribution.png", p17, width = 8, height = 6, dpi = 150)
print(p17)
cat("[Saved] plot_17_residuals_distribution.png\n")

# Residuals vs Fitted
p18 <- ggplot(results_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.55, color = "#00838F", size = 2) +
  geom_hline(yintercept = 0, color = "#F44336", linewidth = 1.2, linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "#212121", linewidth = 0.8) +
  scale_x_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  scale_y_continuous(labels = dollar_format(scale = 1e-3, suffix = "K")) +
  labs(title    = "Residuals vs Fitted Values",
       subtitle = "Random scatter around 0 indicates a good fit",
       x = "Fitted (Predicted) Price", y = "Residual") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_18_residuals_vs_fitted.png", p18, width = 9, height = 6, dpi = 150)
print(p18)
cat("[Saved] plot_18_residuals_vs_fitted.png\n")

# Feature Importance (absolute standardised coefficients)
coef_df <- data.frame(
  Feature     = names(coef(lm_model))[-1],
  Coefficient = coef(lm_model)[-1]
) %>%
  mutate(Abs_Coef  = abs(Coefficient),
         Direction = ifelse(Coefficient > 0, "Positive", "Negative"))

p19 <- ggplot(coef_df, aes(x = reorder(Feature, Abs_Coef),
                            y = Abs_Coef, fill = Direction)) +
  geom_col(color = "white", width = 0.65) +
  coord_flip() +
  scale_fill_manual(values = c("Positive" = "#1565C0", "Negative" = "#B71C1C")) +
  labs(title    = "Feature Importance (|Coefficient| from OLS)",
       subtitle = "Larger bar = stronger effect on price",
       x = "Feature", y = "Absolute Coefficient",
       fill = "Effect") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_19_feature_importance.png", p19, width = 9, height = 6, dpi = 150)
print(p19)
cat("[Saved] plot_19_feature_importance.png\n")

# ── B.7  Performance bar chart ────────────────────────────────────────────────
metrics_df <- data.frame(
  Metric = c("MAE", "RMSE", "R² ×100k", "MAPE (%)"),
  Value  = c(MAE, RMSE, R2 * 100000, MAPE)
)

p20 <- ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_col(width = 0.55, color = "white", show.legend = FALSE) +
  geom_text(aes(label = comma(round(Value, 1))), vjust = -0.4,
            size = 4, fontface = "bold") +
  scale_fill_brewer(palette = "Set2") +
  labs(title    = "Linear Regression — Performance Metrics",
       subtitle = "Lower MAE/RMSE and higher R² indicate better fit",
       x = "Metric", y = "Value") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"))

ggsave("plot_20_regression_metrics.png", p20, width = 8, height = 6, dpi = 150)
print(p20)
cat("[Saved] plot_20_regression_metrics.png\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat("\n")
cat("================================================================\n")
cat("        HOUSE PRICE ML PROJECT — FINAL SUMMARY                 \n")
cat("================================================================\n")
cat(sprintf("  Dataset           : %d houses × %d features\n",
            nrow(df), ncol(df) - 2))
cat("--- Part A: K-Means Clustering --------------------------------\n")
cat(sprintf("  Clusters (k)      : %d\n", k_final))
cat(sprintf("  Cluster sizes     : %s\n",
            paste(table(df$Cluster), collapse=" | ")))
cat(sprintf("  Between SS / Total: %.2f%%\n",
            km_model$betweenss / km_model$totss * 100))
cat(sprintf("  Avg Silhouette    : %.4f\n",
            mean(sil_final[, 3])))
cat("--- Part B: Linear Regression ---------------------------------\n")
cat(sprintf("  MAE               : $%s\n",    comma(round(MAE))))
cat(sprintf("  RMSE              : $%s\n",    comma(round(RMSE))))
cat(sprintf("  R²                : %.4f (%.1f%% variance explained)\n",
            R2, R2 * 100))
cat(sprintf("  MAPE              : %.2f%%\n", MAPE))
cat("================================================================\n")
cat("  All 20 plots saved as PNG in working directory.\n")
cat("================================================================\n")
