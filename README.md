# 🚀 R Machine Learning Projects Suite

A comprehensive collection of **5 advanced machine learning projects** implemented in R, showcasing end-to-end implementations from data preprocessing to model evaluation with detailed visualizations and reports.

> **Status:** ✅ All projects complete and validated | **Language:** R 4.0+ | **Includes:** Datasets, Scripts, Reports & Visualizations

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Projects Overview](#-projects-overview)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#️-installation--setup)
- [Project Descriptions](#-project-descriptions)
- [Usage Instructions](#-usage-instructions)
- [Project Highlights](#-project-highlights)
- [Troubleshooting](#-troubleshooting--common-issues)
- [Contributing & Support](#-contributing--support)

---

## ⚡ Quick Start

Get up and running in 3 minutes:

```r
# 1. Clone the repository
git clone https://github.com/Pavanateja2007-aiml/Programming-With-R.git
cd Programming-With-R

# 2. Install all dependencies
packages <- c("caret", "ggplot2", "dplyr", "tidyr", "e1071", "class",
              "randomForest", "pROC", "Matrix", "recosystem", "cluster")
install.packages(packages)

# 3. Run your first project (Email Spam Classification)
setwd("Project-1")
source("naive_bayes_spam.R")

# 4. View results and plots
```

---

## 🎯 Projects Overview

| # | Project | Algorithm | Type | Dataset | Report |
|---|---------|-----------|------|---------|--------|
| 1️⃣ | Email Spam Classification | Naive Bayes | Binary Classification | emails.csv | ✅ DOCX |
| 2️⃣ | House Prices ML | K-Means + Linear Regression | Clustering + Regression | house_prices.csv | ✅ DOCX |
| 3️⃣ | Titanic Survival Prediction | ML Algorithms | Binary Classification | Titanic Dataset | ✅ DOCX |
| 4️⃣ | Movie Recommendations | SVD Collaborative Filtering | Matrix Factorization | MovieLens | ✅ DOCX |
| 5️⃣ | Breast Cancer Classification | Random Forest | Binary Classification | Cancer Data | ✅ DOCX |

---

## 📁 Repository Structure

```
Programming-With-R/
│
├── README.md                                    # Main documentation
│
├── Project-1: Email Spam Classification
│   ├── naive_bayes_spam.R                      # Main script
│   ├── emails.csv                               # Dataset
│   ├── Email_Spam_Classification_Naive_Bayes_Report.docx
│   └── plots/
│       ├── plot_01_class_distribution_bar.png
│       ├── plot_02_class_distribution_pie.png
│       ├── plot_03_text_length_distribution.png
│       ├── plot_04_word_frequency.png
│       ├── plot_05_boxplot_text_length.png
│       ├── plot_06_confusion_matrix.png
│       ├── plot_07_performance_metrics.png
│       ├── plot_08_actual_vs_predicted.png
│       └── plot_09_train_test_split.png
│
├── Project-2: House Prices ML
│   ├── house_prices_ml.R                       # Main script
│   ├── house_prices.csv                         # Dataset
│   ├── HousePrices_ML_ProjectReport.docx
│   └── plots/ (20 visualizations)
│       ├── plot_01_price_distribution.png
│       ├── plot_02_area_vs_price.png
│       ├── plot_03_price_by_bedrooms.png
│       ├── plot_04_garage_garden_price.png
│       ├── plot_05_correlation_heatmap.png
│       ├── plot_06_pairplot.png
│       ├── plot_07_age_vs_price.png
│       ├── plot_08_distance_vs_price.png
│       ├── plot_09_floors_pie.png
│       ├── plot_10_elbow_method.png
│       ├── plot_11_silhouette_scores.png
│       ├── plot_12_cluster_pca.png
│       ├── plot_13_cluster_area_price.png
│       ├── plot_14_cluster_price_box.png
│       ├── plot_15_cluster_radar.png
│       ├── plot_16_actual_vs_predicted.png
│       ├── plot_17_residuals_distribution.png
│       ├── plot_18_residuals_vs_fitted.png
│       ├── plot_19_feature_importance.png
│       └── plot_20_regression_metrics.png
│
├── Project-3: Titanic Survival Prediction
│   ├── Titanic Survival Prediction.R            # Main script
│   ├── Titanic_Survival_Prediction_Report.docx
│   └── plots/ (Multiple visualizations)
│
├── Project-4: Movie Recommendation System
│   ├── Movie Recommendation System.R            # Main script
│   ├── Movie Recommendation System using Collaborative Filtering with SVD.docx
│   └── plots/ (Visualizations)
│
└── Project-5: Breast Cancer Classification
    ├── Breast Cancer Classification.R           # Main script
    ├── Breast_Cancer_Classification_Report.docx
    └── plots/ (Visualizations)
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **R** version 4.0 or higher ([Download](https://www.r-project.org/))
- **RStudio** (Recommended) ([Download](https://posit.co/download/rstudio-desktop/))
- **Git** for cloning the repository

### Step-by-Step Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Pavanateja2007-aiml/Programming-With-R.git
cd Programming-With-R
```

#### Step 2: Install Required Packages

Run this in your R console:

```r
# Install all required packages
packages <- c(
  "caret",        # ML pipeline & model training
  "ggplot2",      # Advanced visualization
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "e1071",        # Naive Bayes & SVM
  "class",        # k-NN implementation
  "randomForest", # Random Forest
  "pROC",         # ROC curves & AUC
  "Matrix",       # Sparse matrices
  "recosystem",   # Collaborative filtering
  "cluster"       # Clustering algorithms
)

install.packages(packages)
```

#### Step 3: Verify Installation

```r
# Test all packages load correctly
sapply(packages, require, character.only = TRUE)
```

---

## 📊 Project Descriptions

### **Project-1: Email Spam Classification Using Naive Bayes**

**Objective:** Build a text classification system to automatically detect spam emails using the Naive Bayes probabilistic algorithm.

**Key Features:**
- Binary text classification (Spam vs Ham)
- Text preprocessing and feature engineering
- TF-IDF vectorization
- Naive Bayes classifier implementation
- Comprehensive performance evaluation

**Dataset:** emails.csv
- Contains email text and labels
- Balanced spam/ham distribution

**Visualizations (9 plots):**
- Class distribution (bar & pie charts)
- Text length analysis (distribution & boxplot)
- Word frequency analysis
- Confusion matrix visualization
- Performance metrics comparison
- Train/test split visualization

**Technologies Used:**
```r
library(e1071)      # Naive Bayes
library(tm)         # Text mining
library(caret)      # Model evaluation
library(ggplot2)    # Visualizations
```

**Run the Project:**
```r
setwd("Project-1")
source("naive_bayes_spam.R")
```

**Output:** Detailed report in `Email_Spam_Classification_Naive_Bayes_Report.docx`

---

### **Project-2: House Prices ML Project**

**Objective:** Analyze real estate market data using clustering and regression to identify property segments and predict prices.

**Key Features:**
- Exploratory Data Analysis (EDA) with 20 visualizations
- K-Means clustering for market segmentation
- Elbow method & silhouette analysis
- Multiple linear regression for price prediction
- Feature importance ranking
- Residual analysis and model diagnostics

**Dataset:** house_prices.csv
- 500 residential properties
- 8 features: Area, Bedrooms, Bathrooms, Age, Garage, Floors, Garden, Distance
- Target: House Price

**Visualizations (20 plots):**
- Price distribution analysis
- Feature relationships (area, bedrooms, bathrooms, etc.)
- Correlation heatmap
- Pairplot matrix
- Elbow method curve
- Silhouette scores
- PCA cluster visualization
- Cluster analysis (area, price, radar charts)
- Regression predictions vs actual
- Residuals distribution & diagnostics
- Feature importance ranking
- Regression performance metrics

**Technologies Used:**
```r
library(ggplot2)    # Visualizations
library(dplyr)      # Data manipulation
library(caret)      # Model training
library(cluster)    # K-Means clustering
```

**Run the Project:**
```r
setwd("Project-2")
source("house_prices_ml.R")
```

**Output:** Detailed report in `HousePrices_ML_ProjectReport.docx`

---

### **Project-3: Titanic Survival Prediction**

**Objective:** Predict passenger survival on the Titanic using machine learning algorithms and passenger characteristics.

**Key Features:**
- Classification problem on historical data
- Multiple algorithm implementations
- Feature engineering from passenger data
- Model evaluation and comparison
- Visualization of predictions and patterns

**Dataset:** Titanic Survival Dataset
- Historical passenger data from the Titanic
- Features: Age, Sex, Class, Fare, Family size, etc.
- Target: Survived (Yes/No)

**Algorithms Used:**
- Binary classification techniques
- Model performance comparison
- Cross-validation evaluation

**Technologies Used:**
```r
library(caret)      # Model training & validation
library(ggplot2)    # Visualizations
library(dplyr)      # Data manipulation
```

**Run the Project:**
```r
setwd("Project-3")
source("Titanic Survival Prediction.R")
```

**Output:** Detailed report in `Titanic_Survival_Prediction_Report.docx`

---

### **Project-4: Movie Recommendation System Using Collaborative Filtering**

**Objective:** Build a personalized movie recommendation engine using Singular Value Decomposition (SVD) matrix factorization.

**Key Features:**
- Collaborative filtering approach
- SVD matrix factorization
- User-item rating matrix construction
- Personalized recommendations
- Latent factor analysis
- Rating prediction system

**Dataset:** MovieLens Data
- User ratings for various movies
- Sparse rating matrix (users × movies)
- Historical preference data

**Technologies Used:**
```r
library(recosystem) # Collaborative filtering
library(Matrix)     # Sparse matrices
library(dplyr)      # Data manipulation
```

**Key Highlights:**
- Recommends unwatched movies to users
- Handles sparse data effectively
- Interpretable latent factors
- Scalable matrix factorization

**Run the Project:**
```r
setwd("Project-4")
source("Movie Recommendation System.R")
```

**Output:** Detailed report in `Movie Recommendation System using Collaborative Filtering with SVD.docx`

---

### **Project-5: Breast Cancer Classification**

**Objective:** Develop a diagnostic classification system to predict breast tumor malignancy using Random Forest.

**Key Features:**
- Binary classification (Benign vs Malignant)
- Random Forest ensemble learning
- Cross-validation for robustness
- High accuracy predictions
- Feature importance biomarker identification
- Medical decision support system

**Dataset:** Breast Cancer Dataset
- Morphological features of tumors
- Binary outcome (Benign/Malignant)
- Real medical data

**Performance Metrics:**
- High accuracy classification
- ROC-AUC analysis
- Feature importance ranking
- Cross-validation results

**Technologies Used:**
```r
library(randomForest) # Ensemble learning
library(caret)        # Model training & CV
library(pROC)         # ROC curves
library(ggplot2)      # Visualizations
```

**Run the Project:**
```r
setwd("Project-5")
source("Breast Cancer Classification.R")
```

**Output:** Detailed report in `Breast_Cancer_Classification_Report.docx`

---

## 🔑 Key Technologies & Libraries

### Core Packages

| Package | Purpose | Used In |
|---------|---------|---------|
| `caret` | ML pipeline & model training | All projects |
| `ggplot2` | Publication-quality visualizations | All projects |
| `dplyr` | Data manipulation | All projects |
| `tidyr` | Data tidying | Projects 1, 2, 3 |
| `e1071` | Naive Bayes & SVM | Project 1 |
| `class` | k-NN algorithm | Project 3 |
| `randomForest` | Ensemble learning | Projects 3, 5 |
| `pROC` | ROC curves & AUC | Projects 3, 5 |
| `Matrix` | Sparse matrices | Project 4 |
| `recosystem` | Collaborative filtering | Project 4 |
| `cluster` | Clustering algorithms | Project 2 |

### Installation

```r
packages <- c("caret", "ggplot2", "dplyr", "tidyr", "e1071", "class",
              "randomForest", "pROC", "Matrix", "recosystem", "cluster")
install.packages(packages)
```

---

## 🚀 Usage Instructions

### Running Individual Projects

#### Option 1: Using setwd() + source()
```r
setwd("~/Programming-With-R/Project-1")
source("naive_bayes_spam.R")
```

#### Option 2: Full Path
```r
source("~/Programming-With-R/Project-1/naive_bayes_spam.R")
```

#### Option 3: RStudio
1. Open Programming-With-R as RStudio Project
2. Navigate to Project-X folder in Files pane
3. Open the .R file
4. Press Ctrl+Shift+S (or Cmd+Shift+S on Mac)

### Viewing Results

After running a project:

1. **Plots** - View in the Plots pane (RStudio)
2. **Console Output** - Check for model performance metrics
3. **Reports** - Open the .docx files for detailed analysis
4. **Generated Plots** - PNG files saved in project directories

### Common Commands

```r
# Set working directory
setwd("path/to/Project-X")

# View available files
list.files()

# Load dataset
data <- read.csv("filename.csv")

# Explore data
str(data)
head(data)
summary(data)

# Check missing values
colSums(is.na(data))
```

---

## ✨ Project Highlights

### Data Visualization
- 📊 20+ ggplot2 visualizations in Project-2
- 📈 Correlation heatmaps and pairplots
- 🎯 Confusion matrices and performance metrics
- 📉 Residual plots and diagnostics

### Machine Learning Techniques
- ✅ Supervised Learning (Classification & Regression)
- ✅ Unsupervised Learning (Clustering)
- ✅ Ensemble Methods (Random Forest)
- ✅ Text Processing (NLP)
- ✅ Matrix Factorization (SVD)
- ✅ Model Evaluation & Validation

### Comprehensive Reports
- 📄 Detailed .docx reports for each project
- 📊 Professional visualizations
- 📋 Results and interpretations
- 🎓 Learning insights

---

## 🐛 Troubleshooting & Common Issues

### Issue: Package Installation Fails

**Solution:**
```r
# Try with dependencies
install.packages("caret", dependencies = TRUE)

# Or specify CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("caret")
```

### Issue: File Not Found Error

**Solution:**
```r
# Check working directory
getwd()

# List files to verify
list.files()

# Set correct path
setwd("~/Programming-With-R/Project-1")
```

### Issue: Memory or Performance Problems

**Solution:**
```r
# Sample data for testing
set.seed(123)
sample_data <- data[sample(nrow(data), size = 1000), ]

# Or increase memory (Windows)
memory.limit(size = 8000)
```

### Issue: Inconsistent Results

**Solution:**
```r
# Set seed for reproducibility
set.seed(123)

# Then run your analysis
# Results will be consistent
```

### Issue: Package Loading Error

**Solution:**
```r
# Explicitly load library
library(caret)

# Check loaded packages
search()

# Reinstall if needed
remove.packages("caret")
install.packages("caret")
library(caret)
```

---

## 📖 How to Use This Repository

1. **For Learning:** Work through projects sequentially (1→5) to build ML expertise
2. **For Reference:** Refer to specific project implementations for algorithm patterns
3. **For Practice:** Adapt code to your own datasets
4. **For Production:** Use as templates for real-world applications
5. **For Collaboration:** Fork, modify, and contribute improvements

---

## 📚 Learning Path

**Beginner → Intermediate → Advanced**

1. **Start with:** Project-1 (Email Spam) - Simple, text-based
2. **Progress to:** Project-2 (House Prices) - EDA & Clustering
3. **Learn Comparison:** Project-3 (Titanic) - Multiple algorithms
4. **Advanced:** Project-4 (Recommendations) - Matrix factorization
5. **Clinical ML:** Project-5 (Cancer) - High-stakes classification

---

## 🤝 Contributing & Support

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Getting Help

- 📋 Open an issue for bugs or questions
- 💬 Discuss improvements in Pull Requests
- 📧 Contact repository maintainer
- 📚 Check project-specific reports

---

## 📄 License & Citation

This project suite is provided as-is for **educational and professional purposes**.

**Citation:**
```
@github{PavanaTejaML2024,
  title={R Machine Learning Projects Suite},
  author={Pavanateja2007-aiml},
  year={2024},
  url={https://github.com/Pavanateja2007-aiml/Programming-With-R}
}
```

---

## 🙏 Acknowledgments

- Datasets sourced from UCI ML Repository, Kaggle, MovieLens
- Algorithms implemented using established R packages
- Inspired by real-world ML applications
- Community contributions and feedback

---

## 📚 Additional Resources

### R & Machine Learning
- [R for Data Science](https://r4ds.had.co.nz/)
- [CRAN Machine Learning Task View](https://cran.r-project.org/web/views/MachineLearning.html)
- [caret Package Documentation](http://topepo.github.io/caret/)

### Datasets
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [MovieLens Data](https://movielens.org/)

### Visualization
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [R Graphics Cookbook](https://r-graphics.org/)

---

## ⭐ Quick Links

- 🔗 [GitHub Repository](https://github.com/Pavanateja2007-aiml/Programming-With-R)
- 📧 [Issues & Discussions](https://github.com/Pavanateja2007-aiml/Programming-With-R/issues)
- 💡 [Contribute](https://github.com/Pavanateja2007-aiml/Programming-With-R/pulls)

---

## 🎯 Quick Start Summary

```bash
# 1. Clone
git clone https://github.com/Pavanateja2007-aiml/Programming-With-R.git
cd Programming-With-R

# 2. Install packages (in R)
packages <- c("caret", "ggplot2", "dplyr", "tidyr", "e1071", "class",
              "randomForest", "pROC", "Matrix", "recosystem", "cluster")
install.packages(packages)

# 3. Run a project (in R)
setwd("Project-1")
source("naive_bayes_spam.R")

# 4. View results in Plots pane or open .docx reports
```

---

**Last Updated:** June 2026  
**Status:** ✅ All projects complete and validated  
**Total Projects:** 5️⃣ | **Algorithms:** Multiple ML algorithms | **Visualizations:** 50+ plots  

---

**Happy Machine Learning! 🚀**

*Built with ❤️ using R, dedicated to advancing machine learning education and practical implementations.*
