#==============================================================================================
# MANM547 - Machine Learning Coursework – Taiwanese Bankruptcy Prediction
# Script: MANM547-Group_8_Code.R
# Purpose: 
#   Implements the complete CRISP-DM process for predicting bankruptcy risk 
#   in Taiwanese companies using 95 financial ratios (1999–2009) by:
#     - Preparing and balancing data for modelling
#     - Training six machine learning models (Logistic Regression, SVM, Random Forest, XGBoost,
#       Decision Tree, Neural Network) with cross-validation
#     - Evaluating models using AUC, Accuracy, F1, Recall, Precision
#     - Visualising performance (ROC curves, metric comparison charts)
#     - Ranking models and selecting the most suitable for business needs
#     - Summarising results for stakeholders and saving the final model for deployment
# Authors: Balaji Sekar, Bhumika Dasharath, Ceeka Rohit Goud
# Date: 04/August/2025
#==============================================================================================

#==============================================================================================
# 0. Setup and Data Preparation Code for Taiwanese Bankruptcy Prediction
#==============================================================================================

library(tidyverse)
library(caret)
library(ROSE)
library(e1071)
library(corrplot)
library(psych)
library(skimr)
library(corrplot)
library(DataExplorer)
library(funModeling)
library(ranger)
library(xgboost)
library(patchwork)
library(pROC)
library(randomForest)
library(nnet)
library(moments)
library(yardstick)
library(tibble)
library(tidyr)
library(ggplot2)
library(rpart.plot)
library(explore)
library(dplyr)

# Source custom functions
source("C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/MANM547-Group_8_Functions.R")

# Load your dataset with the full path to the file.
Bankruptcy <- read_csv("C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Taiwanese Bankruptcy.csv",
                       col_types = cols(
                         `Bankrupt?` = col_logical(), 
                         `Liability-Assets Flag` = col_logical(), 
                         `Net Income Flag` = col_logical()))

# To replace spaces and special characters in column names
colnames(Bankruptcy) <- gsub("[^[:alnum:]_]", "", gsub("/", "_to_", gsub("-", "_", gsub(" ", "_", colnames(Bankruptcy)))))

#==============================================================================================
# 1. Initial Data Inspection and Understanding (CRISP-DM: Data Understanding)
#==============================================================================================

# Convert the target variable to a factor
Bankruptcy$Bankrupt <- factor(Bankruptcy$Bankrupt, levels = c("TRUE", "FALSE"), labels = c("Yes", "No"))

# 1. Glimpse + structure
glimpse(Bankruptcy)
str(Bankruptcy)

# 2. Summary statistics
summary(Bankruptcy)
skim(Bankruptcy)

# 3. Missing values
cat("Total missing values:", sum(is.na(Bankruptcy)), "\n")
missing_per_column <- colSums(is.na(Bankruptcy))
missing_per_column[missing_per_column > 0]  # Only show problematic columns

# 4. Class imbalance
cat("Class distribution:\n")
print(table(Bankruptcy$Bankrupt))
cat("Proportions:\n")
print(prop.table(table(Bankruptcy$Bankrupt)))

ggplot(Bankruptcy, aes(x = factor(Bankrupt))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Bankruptcy Class Distribution", x = "Bankrupt", y = "Count") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  )

# 5. Correlation matrix for numeric variables
# Select only numeric columns
numeric_vars <- Bankruptcy %>% select(where(is.numeric))

# Remove columns with near-zero variance or high NA
numeric_vars <- numeric_vars[, colSums(is.na(numeric_vars)) == 0]  # remove NAs

# Full matrix (up to 60 or less)
n_vars <- min(ncol(numeric_vars), 60)
cor_matrix <- cor(numeric_vars[, 1:n_vars], use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6, tl.col = "black", order = "hclust")
title(main = "Correlation Matrix of Numeric Features", 
      sub = "Ordered by Hierarchical Clustering", 
      line = -1)

# Show only zero-variance columns
nzv_info <- nearZeroVar(Bankruptcy, saveMetrics = TRUE)
zero_var_cols <- rownames(nzv_info[nzv_info$zeroVar == TRUE, ])
print(zero_var_cols)

# Liability_Assets_Flag also has zero variance
check_flag_column(Bankruptcy, "Liability_Assets_Flag")
check_flag_column(Bankruptcy, "Net_Income_Flag")

#==============================================================================================
# 2.  Data Preparation (CRISP-DM: Data Preparation)
#==============================================================================================

#=================================================
# 2.1 Remove Zero-Variance Features
#=================================================
# A. Identify and Remove Zero-Variance Features
Bankruptcy_cleaned <- Bankruptcy

manual_flags <- c("Liability_Assets_Flag", "Net_Income_Flag")
manual_flags <- manual_flags[manual_flags %in% names(Bankruptcy_cleaned)]

# Automatically detect others
nzv_cols <- nearZeroVar(Bankruptcy_cleaned, saveMetrics = TRUE)
auto_flags <- rownames(nzv_cols)[nzv_cols$zeroVar == TRUE]

# Combine all unique columns to remove
all_to_remove <- unique(c(manual_flags, auto_flags))

if (length(all_to_remove) > 0) {
  cat("Removing zero-variance columns:", paste(all_to_remove, collapse = ", "), "\n")
  Bankruptcy_cleaned <- Bankruptcy_cleaned %>% select(-all_of(all_to_remove))
} else {
  cat("No zero-variance columns to remove.\n")
}

cat("Original column count:", ncol(Bankruptcy), "\n")
cat("Cleaned column count:", ncol(Bankruptcy_cleaned), "\n")

#=================================================
# 2.2 Outlier Detection and Treatment
#=================================================

# Visualize outliers using boxplots
# --- Select first 6 numeric features ---
numeric_df <- Bankruptcy_cleaned %>% 
  select(where(is.numeric)) %>% 
  select(1:6)

# --- Convert to long format for ggplot ---
df_long <- pivot_longer(numeric_df, cols = everything(), names_to = "Feature", values_to = "Value")

# --- Plot faceted boxplots ---
ggplot(df_long, aes(x = Feature, y = Value)) +
  geom_boxplot(outlier.color = "red", fill = "skyblue", outlier.size = 1.2) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  labs(title = "Boxplots for Outlier Detection (Selected Features)", x = NULL, y = "Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    axis.title.y = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Outlier Detection and Capping using IQR method
Bankruptcy_capped <- Bankruptcy_cleaned
numeric_cols <- names(Bankruptcy_capped)[sapply(Bankruptcy_capped, is.numeric)]
for (col in numeric_cols) {
  Bankruptcy_capped[[col]] <- cap_outliers_iqr(Bankruptcy_capped[[col]])
}
cat("\nOutliers capped for numeric features.\n")
cat("Capped outliers using IQR method for", length(numeric_cols), "numeric features.\n")

#=================================================
# 2.3 Feature Scaling (Normalization/Standardization)
#=================================================

# Normalization/Standardization (Feature Scaling)
X <- Bankruptcy_capped %>% select(-Bankrupt)
y <- Bankruptcy_capped$Bankrupt

# Perform standardization (z-score normalization)
cat("\nPerforming standardization...\n")
if (any(!complete.cases(X))) {
  cat("Warning: Missing values detected in features before scaling. Consider imputation.\n")
}
preproc_params <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc_params, X)
Bankruptcy_scaled <- cbind(X_scaled, Bankrupt = y)
cat("Data scaled (centered and scaled).\n")
cat("Data scaled. Dimensions:", dim(Bankruptcy_scaled)[1], "rows x", dim(Bankruptcy_scaled)[2], "columns\n")

#=================================================
# 2.4 Skewness Detection and Transformation
#=================================================

cat("\nConsideration for skewness transformation. (Not implemented in this script).\n")

# Identify highly skewed numeric variables
skews <- sapply(X, function(x) if (is.numeric(x)) moments::skewness(x, na.rm = TRUE) else NA)
# Remove NA skewness values
skews <- skews[!is.na(skews)]
skewed_cols <- names(skews[abs(skews) > 1])  # Threshold of |1| is common

cat("Highly skewed columns:\n")
print(skewed_cols)

# Calculate skewness for numeric features
skew_df <- data.frame(Feature = names(skews), Skewness = skews)

# Select top 15 positive and 15 negative skewed variables
top_skew_df <- skew_df %>%
  filter(!is.na(Skewness)) %>%
  arrange(desc(Skewness)) %>%
  slice_head(n = 15) %>%
  bind_rows(
    skew_df %>%
      filter(!is.na(Skewness)) %>%
      arrange(Skewness) %>%
      slice_head(n = 15)
  ) %>%
  mutate(Direction = ifelse(Skewness > 0, "Right-Skewed", "Left-Skewed"))

# Visualize skewness using ggplot2
ggplot(top_skew_df, aes(x = reorder(Feature, Skewness), y = Skewness)) +
  geom_segment(aes(xend = Feature, y = 0, yend = Skewness), color = "gray70") +
  geom_point(aes(color = Direction), size = 4) +
  geom_text(aes(label = round(Skewness, 2)),
            hjust = ifelse(top_skew_df$Skewness > 0, -0.2, 1.1),
            size = 3) +
  scale_color_manual(values = c("Right-Skewed" = "darkorange", "Left-Skewed" = "steelblue")) +
  coord_flip() +
  labs(
    title = "Top 30 Skewed Numeric Features (15 Left & 15 Right Skewed)",
    x = "Feature", y = "Skewness", color = "Skew Direction"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold"),
        legend.text = element_text(face = "italic"))

# Apply log1p() (or another transformation)
X_transformed <- X
X_transformed[skewed_cols] <- lapply(X_transformed[skewed_cols], function(x) log1p(abs(x)))

# Re-scale after transformation
preproc_params <- preProcess(X_transformed, method = c("center", "scale"))
X_scaled <- predict(preproc_params, X_transformed)
Bankruptcy_scaled <- cbind(X_scaled, Bankrupt = y)

#=================================================
# 2.5 Feature Selection using Random Forest Importance
#=================================================

# 1. Train a random forest model with permutation importance

rf_model <- ranger(
  Bankrupt ~ ., 
  data = Bankruptcy_scaled,
  importance = "permutation",
  num.trees = 500,
  classification = TRUE,
  seed = 143
)
print(rf_model)

# 2. Extract and sort variable importance
importance_df <- data.frame(
  Feature = names(rf_model$variable.importance),
  Importance = rf_model$variable.importance
) %>%
  arrange(desc(Importance))

# Choose top 30 most important features
top_n <- 30
selected_features <- importance_df %>%
  slice_max(order_by = Importance, n = top_n) %>%
  pull(Feature)

# Subset data with selected features
Bankruptcy_prepared_data <- Bankruptcy_scaled %>%
  select(all_of(selected_features), Bankrupt)

summary(Bankruptcy_prepared_data)


# Take top 30 features from importance_df
top_features_plot <- importance_df %>%
  slice_max(order_by = Importance, n = top_n) %>%
  mutate(Feature = reorder(Feature, Importance))  # For sorting in plot

# Visualize the top 30 feature importances
ggplot(top_features_plot, aes(x = Feature, y = Importance, fill = Feature)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 30 Feature Importances (Random Forest)",
    x = "Feature",
    y = "Permutation Importance"
  ) +
  scale_fill_viridis_d(option = "C", direction = -1) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold")
  )
#=================================================
# 2.6 Train-Test Split
#=================================================

# Split the data into training and testing sets (70% train, 30% test)
set.seed(143) # for reproducibility
cat("\nSplitting data into training (70%) and testing (30%) sets...\n")
trainIndex <- createDataPartition(Bankruptcy_prepared_data$Bankrupt, p = 0.7, list = FALSE, times = 1)
data_train <- Bankruptcy_prepared_data[trainIndex, ]
data_test <- Bankruptcy_prepared_data[-trainIndex, ]
data_train$Bankrupt <- factor(data_train$Bankrupt, levels = c("Yes", "No"))  # Or vice versa, based on your factor order
data_test$Bankrupt <- factor(data_test$Bankrupt, levels = c("Yes", "No"))

cat("Original training data class distribution:\n")
print(prop.table(table(data_train$Bankrupt)))

#=================================================
# 2.7 Handling Class Imbalance using ROSE
#=================================================

# Visualize the class distribution before applying ROSE
ggplot(data_train, aes(x = Bankrupt)) +
  geom_bar(fill = "coral") +
  ggtitle("Class Distribution Before ROSE") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  )

# Oversample the minority class and Undersample the majority class
cat("\nApplying SMOTE to the training data...\n")
set.seed(143)
data_train_balanced <- ROSE(Bankrupt ~ ., data = data_train, seed = 143)$data
data_train_balanced$Bankrupt <- factor(data_train_balanced$Bankrupt, levels = c("Yes", "No"))
# `seed` in ROSE ensures reproducibility of the synthetic samples

cat("Balanced training data class distribution (after ROSE):\n")
print(prop.table(table(data_train_balanced$Bankrupt)))

# Visualize the class distribution after ROSE
ggplot(data_train_balanced, aes(x = Bankrupt)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Class Distribution After ROSE") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  )

# data_train_balanced and data_test are now ready for model training and evaluation
cat("\nData preparation complete.\n")
cat("Training data dimensions (balanced):", dim(data_train_balanced)[1], "rows,", dim(data_train_balanced)[2], "columns\n")
cat("Testing data dimensions:", dim(data_test)[1], "rows,", dim(data_test)[2], "columns\n")

#==============================================================================================
# 3. Model Training and Evaluation (CRISP-DM: Modelling and Evaluation)
#==============================================================================================

#=================================================
# 3.1 Logistic Regression Model
#=================================================

# Logistic Regression Model Training and Evaluation
cat("--- Training Logistic Regression Model ---\n")
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)

# Train the logistic regression model
set.seed(143)
model_logreg <- train(Bankrupt ~ ., data = data_train_balanced,
                      method = "glm",
                      family = "binomial",
                      trControl = ctrl,
                      metric = "ROC")

# Evaluate the model
results_logreg <- evaluate_model(model = model_logreg, test_data = data_test, model_name = "Logistic Regression")
plot_calibration_curve(model_logreg, data_test, "Logistic Regression")

# Tuning the Logistic Regression Model with threshold = 0.35
results_logreg_tuned <- evaluate_model(model = model_logreg, test_data = data_test, model_name = "Logistic Regression", threshold = 0.35)

#=================================================
# 3.2 Support Vector Machine (SVM) Model
#=================================================

# SVM Model Training and Evaluation
cat("\n--- Training SVM Model ---\n")
ctrl_SVM <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)
svm_grid <- expand.grid(C = c(0.25, 0.5, 1, 2), sigma = c(0.01, 0.05, 0.1))

# Train the SVM model
set.seed(143)
model_svm <- train(Bankrupt ~ ., data = data_train_balanced,
                   method = "svmRadial",
                   trControl = ctrl_SVM,
                   tuneGrid = svm_grid,
                   metric = "ROC")

# Evaluate the model
results_svm <- evaluate_model(model = model_svm, test_data = data_test, model_name = "SVM")
plot_calibration_curve(model_svm, data_test, "SVM")

#=================================================
# 3.3 Random Forest Model
#=================================================

# Random Forest Model Training and Evaluation
cat("\n--- Training Random Forest Model ---\n")
ctrl_rf <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)

# Train the Random Forest model
set.seed(143)
model_rf <- train(Bankrupt ~ ., data = data_train_balanced,
                  method = "rf",
                  trControl = ctrl_rf,
                  tuneLength = 5,
                  metric = "ROC")

# Evaluate the model
results_rf <- evaluate_model(model = model_rf, test_data = data_test, model_name = "Random Forest")
plot_calibration_curve(model_rf, data_test, "Random Forest", cuts = 15)

#=================================================
# 3.4 XGBoost Model
#=================================================

# XGBoost Model Training and Evaluation
cat("\n--- Training XGBoost Model ---\n")
ctrl_xg <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)
xgb_grid <- expand.grid(nrounds = c(50, 100), max_depth = c(2, 3), eta = c(0.1, 0.3),
                        gamma = 0, colsample_bytree = 0.8, min_child_weight = 1,
                        subsample = 0.8)

# Train the XGBoost model
set.seed(143)
model_xgb <- train(Bankrupt ~ ., data = data_train_balanced,
                   method = "xgbTree",
                   trControl = ctrl_xg,
                   tuneGrid = xgb_grid,
                   metric = "ROC",
                   verbose = FALSE)

# Evaluate the model
results_xgb <- evaluate_model(model = model_xgb, test_data = data_test, model_name = "XG Boost")
plot_calibration_curve(model_xgb, data_test, "XGBoost Model")

#=================================================
# 3.5 Neural Network Model
#=================================================

# Neural Network Model Training and Evaluation
cat("\n--- Training Neural Network Model ---\n")
ctrl_nn <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)
nnet_grid <- expand.grid(size = c(5, 10, 15), decay = c(0.01, 0.1))

# Train the Neural Network model
set.seed(143)
model_nnet <- train(Bankrupt ~ ., data = data_train_balanced,
                    method = "nnet",
                    trControl = ctrl_nn,
                    tuneGrid = nnet_grid,
                    metric = "ROC",
                    trace = FALSE)

# Evaluate the model
results_nnet <- evaluate_model(model = model_nnet, test_data = data_test, model_name = "Neural Network")
plot_calibration_curve(model_nnet, data_test, "Neural Network")

#==============================================================================================
# 4. Model Comparison and Visualization
#==============================================================================================

#=================================================
# 4.1 ROC Curve Comparison with AUC Values
#=================================================

# Create ROC Curves for All Models
roc_data_all <- bind_rows(
  get_roc_with_auc(model_logreg, data_test, "Logistic Regression"),
  get_roc_with_auc(model_svm, data_test, "SVM"),
  get_roc_with_auc(model_rf, data_test, "Random Forest"),
  get_roc_with_auc(model_xgb, data_test, "XGBoost"),
  get_roc_with_auc(model_nnet, data_test, "Neural Network")
)

# Plot ROC curves with AUC in legend
ggplot(roc_data_all, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(linewidth = 1.3) +
  geom_abline(linetype = "dashed", color = "gray50") +
  labs(
    title = "ROC Curve Comparison with AUC Values",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.text = element_text(face = "italic")
  )

roc_data_all <- bind_rows(
  get_roc_with_auc(model_logreg, data_test, "Logistic Regression"),
  get_roc_with_auc(model_svm, data_test, "SVM"),
  get_roc_with_auc(model_rf, data_test, "Random Forest"),
  get_roc_with_auc(model_xgb, data_test, "XGBoost"),
  get_roc_with_auc(model_nnet, data_test, "Neural Network")
)

#=================================================
# 4.2 Performance Metrics Summary Table
#=================================================

# Build comparison table
comparison_table <- bind_rows(
  extract_metrics(results_logreg, "Logistic Regression"),
  extract_metrics(results_svm, "SVM"),
  extract_metrics(results_rf, "Random Forest"),
  extract_metrics(results_xgb, "XGBoost"),
  extract_metrics(results_nnet, "Neural Network")
)

# Add ranking columns
comparison_table <- comparison_table %>%
  mutate(
    Rank_AUC = rank(-AUC, ties.method = "min"),
    Rank_F1 = rank(-F1, ties.method = "min"),
    Rank_Recall = rank(-Recall, ties.method = "min"),
    Rank_Precision = rank(-Precision, ties.method = "min"),
    Overall_Rank = rank(Rank_AUC + Rank_F1 + Rank_Recall + Rank_Precision, ties.method = "min")
  ) %>%
  arrange(Overall_Rank)

print(comparison_table)

# Build Comparison for Champion Model before and after tuning
comparison_table_champion <- bind_rows(
  extract_metrics(results_logreg, "Logistic Regression"),
  extract_metrics(results_logreg_tuned, "Logistic Regression After Tuning")
)

# Add ranking columns
comparison_table_champion <- comparison_table_champion %>%
  mutate(
    Rank_AUC = rank(-AUC, ties.method = "min"),
    Rank_F1 = rank(-F1, ties.method = "min"),
    Rank_Recall = rank(-Recall, ties.method = "min"),
    Rank_Precision = rank(-Precision, ties.method = "min"),
    Overall_Rank = rank(Rank_AUC + Rank_F1 + Rank_Recall + Rank_Precision, ties.method = "min")
  ) %>%
  arrange(Overall_Rank)

print(comparison_table_champion)

#=================================================
# 4.3 Visualization of Model Performance Scores
#=================================================

# Plotting the comparison table
# Convert comparison_table to long format for plotting
comparison_long <- comparison_table %>%
  select(Model, Accuracy, AUC, F1, Recall, Precision) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Plot
ggplot(comparison_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(title = "Model Performance Comparison",
       y = "Score", x = "Model") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold"),
        legend.title = element_text(face = "bold"),
        legend.text = element_text(face = "italic"))

#=================================================
# 4.4 Visualization of Confusion Matrices
#=================================================

# Logistic Regression
plot_conf_matrix(model_logreg, data_test, "Logistic Regression")
# SVM
plot_conf_matrix(model_svm, data_test, "SVM")
# Random Forest
plot_conf_matrix(model_rf, data_test, "Random Forest")
# XGBoost
plot_conf_matrix(model_xgb, data_test, "XGBoost")
# Neural Network
plot_conf_matrix(model_nnet, data_test, "Neural Network")

#==============================================================================================
# 5. Final Summary and Saving files (CRISP-DM: Deployment)
#==============================================================================================

# Save final best model to local folder
saveRDS(model_xgb, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Final_model_xgboost.rds")

# Save prepared datasets
write_csv(Bankruptcy_prepared_data, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Bankruptcy_Selected_Data.csv")
write_csv(data_train_balanced, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Train_data_prepared.csv")
write_csv(data_test, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Test_data_prepared.csv")

# Export comparison table
write_csv(comparison_table, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Model_comparison_results.csv")
write_csv(comparison_table_champion, "C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Model_comparison_Champion_results.csv")

# Summary of Best Model
cat("\n===== Final Model Summary =====\n")
cat("Best Performing Model Based on Overall Rank:\n")
print(comparison_table[1, ])  # Top ranked model

ggsave("C:/Users/balaj/Documents/MSc Business Analytics - Balaji Sekar/Semester 2/MANM574 - Machine Learning/MANM547-Machine_Learning-Group_8/Model_performance_comparison.png", width = 10, height = 6)

cat("\nProject Completed - All steps of CRISP-DM followed from Data Understanding to Deployment.\n")

#==============================================================================================
# End of the main script
#==============================================================================================

#==============================================================================================
# Appendix: Visualizing Predicted vs Actual Probabilities
#==============================================================================================
# Function to get predicted vs actual probabilities for all models
pred_all <- rbind(
  get_pred_vs_actual(model_logreg, data_test, "Logistic Regression"),
  get_pred_vs_actual(model_svm, data_test, "SVM"),
  get_pred_vs_actual(model_rf, data_test, "Random Forest"),
  get_pred_vs_actual(model_xgb, data_test, "XGBoost"),
  get_pred_vs_actual(model_nnet, data_test, "Neural Network")
)

library(scales)
ggplot(pred_all, aes(x = Actual, y = Predicted_Prob, color = Actual)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.8) +          # Show points
  geom_boxplot(alpha = 0.3, outlier.shape = NA) +               # Show boxes
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") + # Threshold line
  facet_wrap(~ Model, ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +   # Convert to %
  labs(
    title = "Predicted vs Actual Probability Comparison",
    subtitle = "Dashed red line = 0.5 decision threshold",
    x = "Actual Class",
    y = "Predicted Probability of Bankruptcy (Yes)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, face = "italic"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    legend.position = "none"
  )

ggplot(pred_all, aes(x = Predicted_Prob, fill = Actual)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ Model, ncol = 3) +
  labs(
    title = "Predicted Probability Distribution by Model",
    x = "Predicted Probability of Bankruptcy",
    y = "Density"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    legend.position = "none"
  )

library(caret)
calibration_data <- calibration(Actual ~ Predicted_Prob | Model, data = pred_all)
xyplot(calibration_data)
xyplot(cal_obj, type = "l", auto.key = list(columns = 1),
       xlab = "Predicted Probability", ylab = "Observed Event Percentage",
       main = "Calibration Curve - Logistic Regression")
#=================================================
# DataExplorer EDA Report
#=================================================

install.packages("DataExplorer")

# Load the library
library(DataExplorer)

# Check your data
str(Bankruptcy_scaled)

# Generate the EDA report for Bankruptcy_scaled
create_report(
  data = Bankruptcy_scaled,
  y = "Bankrupt",                             # Target variable
  report_title = "Bankruptcy Data EDA Report",
  output_file = "Bankruptcy_EDA_Report.html",
  output_dir = getwd()                        # Save in current working directory
)

# Generate the EDA report for Bankruptcy_prepared_data
create_report(
  data = Bankruptcy_prepared_data,
  y = "Bankrupt",                             # Target variable
  report_title = "Bankruptcy Prepared Data EDA Report",
  output_file = "Bankruptcy_Prepared_EDA_Report.html",
  output_dir = getwd()                        # Save in current working directory
)

#=================================================
# 3.5 Decision Tree Model
#=================================================

# Decision Tree Model Training and Evaluation
cat("\n--- Training Decision Tree Model ---\n")
ctrl_dt <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)

# Train the Decision Tree model
set.seed(143)
# Define a tuning grid for the complexity parameter (cp)
dt_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.005))

model_dt <- train(Bankrupt ~ ., data = data_train_balanced,
                  method = "rpart",
                  trControl = ctrl_dt,
                  tuneGrid = dt_grid,
                  metric = "ROC")

# Evaluate the model using the function you already have
results_dt <- evaluate_model(model = model_dt, test_data = data_test, model_name = "Decision Tree")

# You can also inspect the final, optimal model
print(model_dt)

#=================================================
# Visualization of Decision Tree
#=================================================

pruned_tree <- prune(model_dt$finalModel, cp = 0.02)

library(rpart.plot)
rpart.plot(pruned_tree,
           type = 2,
           extra = 106,
           under = TRUE,
           faclen = 0,
           cex = 0.8,
           main = "Pruned Decision Tree for Taiwanese Bankruptcy Prediction")

# Shallow tree (very simple)
pruned_tree_small <- prune(model_dt$finalModel, cp = 0.05)

# Medium tree
pruned_tree_medium <- prune(model_dt$finalModel, cp = 0.02)

# Full tree (less pruned but cleaner)
pruned_tree_large <- prune(model_dt$finalModel, cp = 0.005)

# Plot all three separately
rpart.plot(pruned_tree_small, type = 2, extra = 106, under = TRUE, cex = 0.8,
           main = "Small Pruned Tree")
rpart.plot(pruned_tree_medium, type = 2, extra = 106, under = TRUE, cex = 0.8,
           main = "Medium Pruned Tree")
rpart.plot(pruned_tree_large, type = 2, extra = 106, under = TRUE, cex = 0.8,
           main = "Large Pruned Tree (Readable)")

# Decision Tree
plot_conf_matrix(model_dt, data_test, "Decision Tree")


# Select top N numeric variables with highest variance
numeric_vars <- Bankruptcy_scaled %>% select(where(is.numeric))
top_vars <- names(sort(apply(numeric_vars, 2, var), decreasing = TRUE))[1:6]

# Melt the data for ggplot
library(reshape2)
Bankruptcy_melted <- cbind(numeric_vars[top_vars], Bankrupt = Bankruptcy_scaled$Bankrupt)
Bankruptcy_melted <- melt(Bankruptcy_melted, id.vars = "Bankrupt")

# Boxplot with enhancements
ggplot(Bankruptcy_melted, aes(x = Bankrupt, y = value, fill = Bankrupt)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 4, alpha = 0.7) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  labs(title = "Boxplots of Top Numeric Features by Bankruptcy Class",
       x = "Bankrupt", y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        strip.text = element_text(face = "bold"))


# Density plots of same top 6 variables split by class
ggplot(Bankruptcy_melted, aes(x = value, fill = Bankrupt, color = Bankrupt)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  labs(title = "Density Plots of Key Features by Bankruptcy Class",
       x = "Value", y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        strip.text = element_text(face = "bold"))


library(patchwork)

# --- Step 1: Select and reshape top numeric variables ---
selected_vars <- names(numeric_vars)[1:6]  # Pick top 6 for a 2x3 layout

outlier_long <- numeric_vars %>%
  select(all_of(selected_vars)) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")
p_box <- ggplot(outlier_long, aes(x = Feature, y = Value)) +
  geom_boxplot(fill = "skyblue", outlier.color = "red", outlier.size = 1.2) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  labs(title = "Boxplots for Outlier Detection", x = NULL, y = "Value") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
p_box


p_hist <- ggplot(outlier_long, aes(x = Value)) +
  geom_histogram(fill = "lightgreen", color = "black", bins = 30) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  labs(title = "Histograms of Numeric Features", x = "Value", y = "Count") +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Display together vertically using patchwork
p_box / p_hist
#==============================================================================================
# End of Appendix
#==============================================================================================