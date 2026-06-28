# R Machine Learning Projects Suite

A comprehensive collection of five advanced machine learning projects implemented in R, demonstrating supervised learning, unsupervised learning, collaborative filtering, and advanced model evaluation techniques.

## 📋 Table of Contents

- [Projects Overview](#projects-overview)
- [Installation & Setup](#installation--setup)
- [Project Descriptions](#project-descriptions)
- [Dependencies](#dependencies)
- [Usage Instructions](#usage-instructions)
- [Project Highlights](#project-highlights)
- [Contact & Support](#contact--support)

---

## 🎯 Projects Overview

| Project | Algorithm | Type | Dataset | Accuracy |
|---------|-----------|------|---------|----------|
| Email Spam Classification | Naive Bayes | Binary Classification | emails.csv | - |
| House Prices Analysis | K-Means + Linear Regression | Unsupervised + Supervised | house_prices.csv | - |
| Heart Disease Prediction | Logistic Regression, k-NN, Random Forest | Multi-model Comparison | Heart Disease Dataset | AUC ≈ 0.91 |
| Movie Recommendations | Collaborative Filtering (SVD) | Matrix Factorization | MovieLens 100k | - |
| Breast Cancer Classification | Random Forest | Binary Classification | Wisconsin Breast Cancer | 96.35% |

---

## 🛠️ Installation & Setup

### Prerequisites
- **R** (version 4.0 or higher)
- **RStudio** (recommended for ease of use)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/R-ML-Projects.git
cd R-ML-Projects
```

### Step 2: Install Required Packages
Run the following in R console:

```r
# Install required packages
packages <- c("caret", "ggplot2", "dplyr", "tidyr", "e1071", "class", 
              "randomForest", "pROC", "Matrix", "recosystem", "cluster")

install.packages(packages)
```

### Step 3: Load Project-Specific Dependencies
Each project may have specific requirements. Check individual project directories for additional setup.

---

## 📊 Project Descriptions

### **Project-1: Email Spam Classification Using Naive Bayes**

**Objective:** Build an automated binary text classifier to distinguish spam emails from legitimate (ham) emails using the Naive Bayes algorithm.

**Key Features:**
- Load, inspect, and clean email text data
- Comprehensive Exploratory Data Analysis with visualizations
- Complete text preprocessing pipeline (tokenization, stopword removal, stemming)
- TF-IDF feature extraction
- Naive Bayes classifier implementation and training
- Performance evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1 Score

**Technologies:** `e1071`, `tm`, `caret`

**Expected Output:**
- Trained Naive Bayes model
- Performance metrics and confusion matrix
- Text preprocessing insights

---

### **Project-2: House Prices ML Project**

**Objective:** Discover market segments in residential properties and predict house prices using clustering and regression techniques.

**Key Features:**
- 9 exploratory visualizations using `ggplot2`
- K-Means Clustering for property segmentation
- Elbow Method and Silhouette Analysis for optimal k selection
- Multiple Linear Regression for price prediction
- Feature importance analysis
- 20+ professional visualizations
- Complete reproducible R implementation

**Dataset:** 500 residential properties with 8 features (area, bedrooms, bathrooms, age, garage, floors, garden, distance)

**Technologies:** `ggplot2`, `dplyr`, `caret`, `cluster`

**Key Insights:**
- Natural market tiers identification
- Feature importance ranking
- Price prediction model accuracy

---

### **Project-3: Heart Disease Prediction**

**Objective:** Compare and benchmark multiple classification algorithms to predict heart disease with high accuracy.

**Key Features:**
- Model Selection & Benchmarking: Compare Logistic Regression, k-NN, and Random Forest
- 5-fold cross-validation for robust performance estimation
- Performance optimization achieving AUC ≈ 0.91
- Feature importance analysis (Maximum heart rate as key predictor)
- Residual analysis to verify OLS assumptions
- PCA-based decision boundary visualization
- Cross-validation and generalization validation

**Technologies:** `caret`, `pROC`, `randomForest`, `class`

**Best Performing Model:** Random Forest (AUC ≈ 0.91)

**Key Predictor:** Maximum heart rate (thalach)

---

### **Project-4: Movie Recommendation System Using Collaborative Filtering with SVD**

**Objective:** Implement a matrix factorization-based collaborative filtering system to provide personalized movie recommendations.

**Key Features:**
- Construct user-item rating matrix from historical data
- Singular Value Decomposition (SVD) with 50 latent factors
- Missing rating imputation using global mean strategy
- Predicted rating reconstruction for all user-movie pairs
- Top 5 unwatched movie recommendations for target users
- Latent factor interpretation and analysis

**Dataset:** MovieLens 100k Dataset
- 100,000 ratings from 943 users
- 1,682 movies
- 1-5 rating scale

**Technologies:** `recosystem`, `Matrix`, `dplyr`

**Output:**
- Trained SVD matrix factorization model
- User-specific recommendations
- Rating predictions for validation

---

### **Project-5: Breast Cancer Classification**

**Objective:** Develop an automated diagnostic system to classify breast tumors as benign or malignant using Random Forest classification.

**Key Features:**
- Binary classification of breast tumor malignancy
- Random Forest classifier with optimized hyperparameters
- 5-fold cross-validation consistency verification
- High-accuracy predictions supporting clinical decision-making
- Feature importance ranking identifying key biomarkers
- Comprehensive performance metrics

**Dataset:** Wisconsin Breast Cancer Dataset
- 683 samples (after removing 16 incomplete records)
- 9 morphological features
- 65% benign, 35% malignant distribution

**Performance Metrics:**
- **Test Accuracy:** 96.35% (5-fold CV mean: 96.42%)
- **ROC AUC:** 0.99 (near-perfect discrimination)
- **Precision/Recall (Benign):** 98%/97%
- **Precision/Recall (Malignant):** 94%/96%

**Top Biomarkers:**
1. Bare Nuclei (85.2)
2. Cell Size (76.5)
3. Cell Shape (74.1)

**Technologies:** `randomForest`, `caret`, `pROC`

---

## 📦 Dependencies

### Core Libraries
```r
caret          # Classification and Regression Training
ggplot2        # Advanced data visualization
dplyr          # Data manipulation
tidyr          # Data tidying
e1071          # Naive Bayes and SVM
class          # k-Nearest Neighbors
randomForest   # Random Forest implementation
pROC           # ROC curve analysis
Matrix         # Sparse matrices for SVD
recosystem     # Collaborative filtering
cluster        # Clustering algorithms
```

### Installation Command
```r
install.packages(c("caret", "ggplot2", "dplyr", "tidyr", "e1071", 
                   "class", "randomForest", "pROC", "Matrix", 
                   "recosystem", "cluster"))
```

---

## 🚀 Usage Instructions

### General Setup for All Projects
1. Set your working directory:
   ```r
   setwd("path/to/R-ML-Projects")
   ```

2. Source project-specific R scripts:
   ```r
   source("Project-1/spam_classification.R")
   ```

### Running Individual Projects

**Project-1:**
```r
source("Project-1/email_spam_classification.R")
# Output: Trained model and performance metrics
```

**Project-2:**
```r
source("Project-2/house_prices_analysis.R")
# Output: Clustering results, regression model, 20+ visualizations
```

**Project-3:**
```r
source("Project-3/heart_disease_prediction.R")
# Output: Model comparisons, ROC curves, feature importance plots
```

**Project-4:**
```r
source("Project-4/movie_recommendations.R")
# Output: Personalized recommendations for target users
```

**Project-5:**
```r
source("Project-5/breast_cancer_classification.R")
# Output: Classification results, ROC curve, feature importance
```

---

## ✨ Project Highlights

### Advanced Techniques Demonstrated
- ✅ **Text Preprocessing:** Tokenization, stemming, TF-IDF vectorization
- ✅ **Unsupervised Learning:** K-Means clustering with elbow method
- ✅ **Supervised Learning:** Multiple regression and classification algorithms
- ✅ **Model Evaluation:** Cross-validation, ROC-AUC, confusion matrices
- ✅ **Dimensionality Reduction:** PCA visualization, SVD matrix factorization
- ✅ **Feature Engineering:** Feature importance analysis and biomarker identification
- ✅ **Data Visualization:** ggplot2 for professional publication-quality graphics
- ✅ **Statistical Validation:** Residual analysis and OLS assumption verification

### Model Performance Summary
| Project | Best Model | Performance |
|---------|-----------|-------------|
| Project-1 | Naive Bayes | Detailed metrics provided |
| Project-2 | Linear Regression + K-Means | Feature importance ranked |
| Project-3 | Random Forest | AUC = 0.91 (5-fold CV) |
| Project-4 | SVD Collaborative Filtering | 50 latent factors |
| Project-5 | Random Forest | Accuracy = 96.35%, AUC = 0.99 |

---

## 📁 Repository Structure

```
R-ML-Projects/
├── Project-1/
│   ├── emails.csv
│   ├── email_spam_classification.R
│   └── README.md
├── Project-2/
│   ├── house_prices.csv
│   ├── house_prices_analysis.R
│   └── README.md
├── Project-3/
│   ├── heart_disease_data.csv
│   ├── heart_disease_prediction.R
│   └── README.md
├── Project-4/
│   ├── movielens_100k/
│   ├── movie_recommendations.R
│   └── README.md
├── Project-5/
│   ├── breast_cancer.csv
│   ├── breast_cancer_classification.R
│   └── README.md
└── README.md (this file)
```

---

## 🔍 Key Technologies & Concepts

### Machine Learning Algorithms
- Naive Bayes (probabilistic classifier)
- Logistic Regression (linear model for classification)
- k-Nearest Neighbors (instance-based learning)
- Random Forest (ensemble learning)
- K-Means Clustering (unsupervised partitioning)
- Multiple Linear Regression (supervised regression)
- Singular Value Decomposition (matrix factorization)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Curve
- Cross-Validation (5-fold, k-fold)
- Mean Squared Error (MSE)
- R-squared (R²)

### Data Science Workflow
1. Data Loading & Inspection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning & Preprocessing
4. Feature Engineering & Extraction
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation & Validation
8. Results Interpretation & Visualization

---

## 📖 How to Use This Repository

1. **For Learning:** Work through projects sequentially (1→5) to build ML expertise
2. **For Reference:** Refer to specific project implementations for algorithm patterns
3. **For Production:** Adapt code and models to your own datasets and business problems
4. **For Collaboration:** Fork, modify, and contribute improvements back

---

## ⚠️ Important Notes

- Ensure all datasets (CSV files) are in the correct project directories before running scripts
- Some computations may take time on large datasets (Project-4)
- Cross-validation and ensemble methods are computationally intensive; results may vary slightly due to random seeds
- For reproducibility, consider setting seed values in R scripts: `set.seed(123)`

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project suite is provided as-is for educational and professional purposes.

---

## 📧 Contact & Support

For questions, suggestions, or issues:
- Create an issue in the GitHub repository
- Reach out with specific project inquiries

---

## 🙏 Acknowledgments

- Datasets sourced from public ML repositories and research institutions
- Algorithms implemented using established R packages and best practices
- Inspired by real-world machine learning applications in healthcare, finance, and digital pathology

---

**Last Updated:** June 2026

**Status:** ✅ All projects complete and validated

---

## Quick Start Guide

```r
# 1. Set working directory
setwd("path/to/R-ML-Projects")

# 2. Install all dependencies
packages <- c("caret", "ggplot2", "dplyr", "tidyr", "e1071", "class", 
              "randomForest", "pROC", "Matrix", "recosystem", "cluster")
install.packages(packages)

# 3. Run your first project
source("Project-5/breast_cancer_classification.R")

# 4. Explore results and visualizations
# Review generated plots and model performance metrics
```

---

**Happy Machine Learning! 🚀**
