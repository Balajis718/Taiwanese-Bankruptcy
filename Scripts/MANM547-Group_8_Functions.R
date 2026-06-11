#==============================================================================================
# MANM547 - Machine Learning Coursework – Taiwanese Bankruptcy Prediction
# Script: MANM547-Group_8_Functions.R
# Purpose: To Create Functions which will be reused in MANM547-Group_8_Code.R file
# Authors: Balaji Sekar, Bhumika Dasharath, Ceeka Rohit Goud
# Date: 04/August/2025
#==============================================================================================

#=================================================
# 2.x Data Quality Check Functions
#=================================================

# Function to check the structure of a dataframe
check_flag_column <- function(df, colname) {
  cat(glue::glue("Checking '{colname}':\n"))
  if (colname %in% names(df)) {
    cat("  Unique values:\n")
    print(unique(df[[colname]]))
    cat("  Variance:", var(df[[colname]]), "\n\n")
  } else {
    cat(glue::glue("  Column '{colname}' not found.\n\n"))
  }
}

#=================================================
# 2.x Outlier Capping Function (IQR Method)
#=================================================

# Function to cap outliers using the IQR method
cap_outliers_iqr <- function(x) {
  if (is.numeric(x)) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    upper_bound <- Q3 + 1.5 * IQR_val
    lower_bound <- Q1 - 1.5 * IQR_val
    x[x < lower_bound] <- lower_bound
    x[x > upper_bound] <- upper_bound
  }
  return(x)
}

#=================================================
# 3.x Model Evaluation Function (Confusion Matrix, AUC, F1, Recall, Precision)
#=================================================

# Function to evaluate a model on test data
evaluate_model <- function(model, test_data, model_name, threshold = 0.5) {
  cat("\nEvaluation for", model_name, "on Test Data:\n")
  
  # Ensure target is a factor with proper levels
  test_data$Bankrupt <- factor(test_data$Bankrupt, levels = c("Yes", "No"))
  
  # Predict probabilities
  probabilities <- predict(model, newdata = test_data, type = "prob")
  
  # Predict class based on threshold
  predicted_classes <- ifelse(probabilities$Yes >= threshold, "Yes", "No")
  predicted_classes <- factor(predicted_classes, levels = c("Yes", "No"))
  
  # Confusion Matrix
  cm <- confusionMatrix(predicted_classes, test_data$Bankrupt, positive = "Yes")
  print(cm)
  
  # Evaluate AUC
  eval_data_auc <- data.frame(
    obs = test_data$Bankrupt,
    pred = predicted_classes,
    Yes = probabilities$Yes,
    No = probabilities$No
  )
  auc <- twoClassSummary(eval_data_auc, lev = levels(test_data$Bankrupt))
  cat("AUC-ROC:", auc[1], "\n")
  
  # Yardstick metrics
  eval_data_yardstick <- tibble(
    truth = test_data$Bankrupt,
    predicted = predicted_classes,
    prob_yes = probabilities$Yes
  )
  
  f1 <- yardstick::f_meas(eval_data_yardstick, truth = truth, estimate = predicted, event_level = "first")$.estimate
  rec <- yardstick::recall(eval_data_yardstick, truth = truth, estimate = predicted, event_level = "first")$.estimate
  prec <- yardstick::precision(eval_data_yardstick, truth = truth, estimate = predicted, event_level = "first")$.estimate
  
  cat("F1-Score:", f1, "\n")
  cat("Recall:", rec, "\n")
  cat("Precision:", prec, "\n")
  
  # Return metrics
  list(cm = cm, auc = auc, f1 = f1, recall = rec, precision = prec)
}

#=================================================
# 4.x ROC Curve Generation Function with AUC Values
#=================================================

# Function to generate ROC curve data with AUC values
get_roc_with_auc <- function(model, test_data, model_name) {
  probs <- predict(model, newdata = test_data, type = "prob")$Yes
  roc_obj <- roc(response = test_data$Bankrupt,
                 predictor = probs,
                 levels = c("No", "Yes"), direction = "<")
  auc_val <- round(auc(roc_obj), 3)
  roc_df <- data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities,
    Model = paste0(model_name, " (AUC=", auc_val, ")")
  )
  return(roc_df)
}

#=================================================
# 4.x Extracting Model Metrics into Summary Table
#=================================================

# Function to extract evaluation metrics from the model evaluation result
extract_metrics <- function(result, model_name) {
  tibble(
    Model = model_name,
    Accuracy = round(result$cm$overall["Accuracy"], 4),
    AUC = round(as.numeric(result$auc[1]), 4),
    F1 = round(result$f1, 4),
    Recall = round(result$recall, 4),
    Precision = round(result$precision, 4)
  )
}

plot_conf_matrix <- function(model, test_data, model_name) {
  preds <- predict(model, newdata = test_data)
  truth <- test_data$Bankrupt
  cm <- confusionMatrix(preds, truth, positive = "Yes")
  cm_df <- as.data.frame(cm$table)
  
  ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(
      title = paste("Confusion Matrix:", model_name),
      x = "Predicted",
      y = "Actual",
      fill = "Count"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(face = "bold"),
      legend.title = element_text(face = "bold")
    )
}

#=================================================
# 4.x Extracting Model Metrics for Confusion Matrix
#=================================================

plot_conf_matrix <- function(model, test_data, model_name) {
  preds <- predict(model, newdata = test_data)
  truth <- test_data$Bankrupt
  cm <- confusionMatrix(preds, truth, positive = "Yes")
  cm_df <- as.data.frame(cm$table)
  
  ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(
      title = paste("Confusion Matrix:", model_name),
      x = "Predicted",
      y = "Actual",
      fill = "Count"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(face = "bold"),
      legend.title = element_text(face = "bold")
    )
}

#=================================================
# 4.x Plot Calibration Curve with Enhanced Features
#=================================================
plot_calibration_curve <- function(model, test_data, model_name, cuts = 10, line_col = "steelblue") {
  # Ensure the target variable is a factor with correct levels
  test_data$Bankrupt <- factor(test_data$Bankrupt, levels = c("Yes", "No"))
  
  # Get predicted probabilities for "Yes"
  pred_probs <- predict(model, newdata = test_data, type = "prob")$Yes
  
  # Prepare calibration data
  cal_data <- data.frame(
    obs = test_data$Bankrupt,
    Yes = pred_probs
  )
  
  # Create calibration object
  cal_obj <- calibration(obs ~ Yes, data = cal_data, class = "Yes", cuts = cuts)
  
  # Enhanced plot
  xyplot(cal_obj, type = "l", auto.key = list(columns = 1, lines = TRUE, points = FALSE),
         lwd = 3,                          # Thicker line
         col = line_col,                   # Customizable line color
         xlab = "Predicted Probability", 
         ylab = "Observed Event Percentage",
         main = paste("Calibration Curve -", model_name),
         panel = function(...) {
           panel.grid(h = -1, v = -1, col.line = "lightgray", lty = 3) # Light grid
           panel.abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)   # Perfect calibration line
           panel.xyplot(...)
         })
}
#=================================================
# Appendix: Function to Get Predicted vs Actual Probabilities
#=================================================
get_pred_vs_actual <- function(model, test_data, model_name) {
  probs <- predict(model, newdata = test_data, type = "prob")$Yes
  data.frame(
    Actual = test_data$Bankrupt,
    Predicted_Prob = probs,
    Model = model_name
  )
}
#=================================================
# End of Functions
#=================================================