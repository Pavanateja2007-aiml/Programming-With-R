# ==============================================================================
# Movie Recommendation System using Collaborative Filtering with SVD
# ==============================================================================

# Load required packages
library(tidyverse)
library(reshape2)
library(matrixStats)
library(ggplot2)

# --- 1. Dataset Downloading and Extraction ---
cat("Downloading MovieLens 100k dataset...\n")
url <- "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
zip_file <- "ml-100k.zip"

if (!file.exists(zip_file)) {
  download.file(url, destfile = zip_file, mode = "wb")
}
unzip(zip_file, overwrite = TRUE)

# --- 2. Data Loading ---
# Load ratings (u.data) -> columns: user_id, movie_id, rating, timestamp
ratings <- read_delim("ml-100k/u.data", delim = "\t", 
                      col_names = c("user_id", "movie_id", "rating", "timestamp"),
                      show_col_types = FALSE)

# Load movie titles (u.item) -> columns: movie_id, title
movies <- read_delim("ml-100k/u.item", delim = "|", 
                     col_names = c("movie_id", "title"), 
                     select = c(1, 2),
                     locale = locale(encoding = "Latin1"),
                     show_col_types = FALSE)

cat(sprintf("Ratings shape: (%d, %d)\n", nrow(ratings), ncol(ratings)))
cat(sprintf("Movies shape: (%d, %d)\n", nrow(movies), ncol(movies)))

# --- 3. Train/Test Split (80/20) ---
set.seed(42)
sample_indices <- sample(1:nrow(ratings), size = 0.8 * nrow(ratings))
train_data <- ratings[sample_indices, ]
test_data  <- ratings[-sample_indices, ]

cat(sprintf("Train size: %d, Test size: %d\n", nrow(train_data), nrow(test_data)))

# --- 4. Pivot User-Item Matrix & Mean Imputation ---
# Convert long table format to wide user-item matrix layout
user_item_matrix <- train_data %>%
  acast(user_id ~ movie_id, value.var = "rating", fill = NA)

cat(sprintf("User-Item matrix shape: (%d, %d)\n", nrow(user_item_matrix), ncol(user_item_matrix)))

# Compute the global mean rating from existing elements
global_mean <- mean(user_item_matrix, na.rm = TRUE)
cat(sprintf("Global mean rating: %.3f\n", global_mean))

# Replace NAs with global mean (replicating dense SVD approximation behavior)
user_item_imputed <- user_item_matrix
user_item_imputed[is.na(user_item_imputed)] <- global_mean

# Center the matrix around its mean (Standard TruncatedSVD pre-processing)
user_item_centered <- user_item_imputed - global_mean

# --- 5. Singular Value Decomposition (SVD) ---
n_components <- 50
svd_decomp <- svd(user_item_centered, nu = n_components, nv = n_components)

# Extract low-rank components
U_k <- svd_decomp$u
D_k <- diag(svd_decomp$d[1:n_components])
V_k <- svd_decomp$v

# Reconstruct predictions matrix and add back the baseline mean shift
predicted_ratings_matrix <- (U_k %*% D_k %*% t(V_k)) + global_mean
rownames(predicted_ratings_matrix) <- rownames(user_item_matrix)
colnames(predicted_ratings_matrix) <- colnames(user_item_matrix)

# --- 6. Top 5 Recommendations for User 1 ---
target_user <- "1"

# Identify movies user 1 has already watched in the training dataset
already_rated_ids <- train_data %>%
  filter(user_id == as.numeric(target_user)) %>%
  pull(movie_id) %>%
  as.character()

# Extract all row predictions for target user
user_predictions <- predicted_ratings_matrix[target_user, ]

# Filter watched entries, map names, sort, and select top 5
recommendations <- tibble(
  movie_id = as.numeric(names(user_predictions)),
  predicted_rating = user_predictions
) %>%
  left_join(movies, by = "movie_id") %>%
  filter(!as.character(movie_id) %in% already_rated_ids) %>%
  arrange(desc(predicted_rating)) %>%
  slice(1:5) %>%
  select(movie_id, title, predicted_rating)

cat(sprintf("\nTop 5 recommendations for User %s:\n", target_user))
print(recommendations)

# --- 7. Model Evaluation (RMSE on Test Set) ---
test_predictions <- apply(test_data, 1, function(row) {
  u <- as.character(row["user_id"])
  m <- as.character(row["movie_id"])
  
  # Ensure the user and item indices exist inside our reconstructed matrix limits
  if (u %in% rownames(predicted_ratings_matrix) && m %in% colnames(predicted_ratings_matrix)) {
    return(predicted_ratings_matrix[u, m])
  } else {
    return(global_mean) # Baseline global mean fallback for missing boundaries
  }
})

rmse_value <- sqrt(mean((test_data$rating - test_predictions)^2))
cat(sprintf("\nTest RMSE: %.4f\n", rmse_value))

# --- 8. Variance Explained Visualization ---
# Calculate the total variance of singular values
total_variance <- sum(svd_decomp$d^2)
explained_variance_ratio <- (svd_decomp$d^2) / total_variance

# Dynamic calculation array up to maximum available components length
components_range <- 1:length(svd_decomp$d)
cumulative_explained_var <- cumsum(explained_variance_ratio)

plot_data <- tibble(
  Components = components_range,
  CumulativeVariance = cumulative_explained_var
)

# Plotting with ggplot2
variance_plot <- ggplot(plot_data, aes(x = Components, y = CumulativeVariance)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(color = "blue", size = 1.5) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "red", linewidth = 0.8) +
  labs(
    title = "Explained Variance vs. Number of SVD Components",
    x = "Number of Components (k)",
    y = "Cumulative Explained Variance"
  ) +
  theme_minimal()

print(variance_plot)

# Output summary metric console blocks
cat("\nVariance explained by first 10 components:\n")
for (i in 1:10) {
  cat(sprintf("Component %d: %.4f\n", i, explained_variance_ratio[i]))
}
cat(sprintf("Total variance with %d components: %.4f\n", 
            n_components, cumulative_explained_var[n_components]))
