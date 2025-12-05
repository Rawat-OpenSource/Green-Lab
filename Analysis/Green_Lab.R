library(stringr)
library(dplyr)
library(tidyr)
library(knitr)
library(car)
library(ggpubr)
library(rstatix)
library(PMCMRplus)
library(moments)

# Load the data
energy_data <- final_energy_data
str(energy_data)

# Convert categorical variables to factors
energy_data$task_type <- factor(energy_data$task_type)
energy_data$model <- factor(energy_data$model)

# Check for missing values
sum(is.na(energy_data))

# DATA EXPLORATION 

summary_stats <- energy_data %>%
  group_by(model, task_type) %>%
  summarise(
    n = n(),
    mean = mean(gpu_energy_j, na.rm = TRUE),
    median = median(gpu_energy_j, na.rm = TRUE),
    sd = sd(gpu_energy_j, na.rm = TRUE),
    variance = var(gpu_energy_j, na.rm = TRUE),
    min = min(gpu_energy_j, na.rm = TRUE),
    max = max(gpu_energy_j, na.rm = TRUE),
    q1 = quantile(gpu_energy_j, 0.25, na.rm = TRUE),
    q3 = quantile(gpu_energy_j, 0.75, na.rm = TRUE),
    iqr = IQR(gpu_energy_j, na.rm = TRUE),
    skewness = moments::skewness(gpu_energy_j, na.rm = TRUE),
    kurtosis = moments::kurtosis(gpu_energy_j, na.rm = TRUE),
    .groups = 'drop'
  )

print(kable(summary_stats, digits = 2))

concise_summary_stats <- energy_data %>%
  group_by(model, task_type) %>%
  summarise(
    mean = round(mean(gpu_energy_j, na.rm = TRUE),2),
    median = round(median(gpu_energy_j, na.rm = TRUE),2),
    sd = round(sd(gpu_energy_j, na.rm = TRUE),2),
    variance = round(var(gpu_energy_j, na.rm = TRUE),2),
    q1 = quantile(gpu_energy_j, 0.25, na.rm = TRUE),
    q3 = quantile(gpu_energy_j, 0.75, na.rm = TRUE),
    iqr = IQR(gpu_energy_j, na.rm = TRUE),
    .groups = 'drop'
  )

# VISUALIZATION & DISTRIBUTION PLOTS

# Setting plotting theme for beautification 
theme_set(theme_minimal(base_size = 12) +
            theme(plot.title = element_text(hjust = 0.5, face = "bold"),
                  axis.title = element_text(face = "bold"),
                  legend.position = "bottom"))

# Boxplot of the energy consumed by task type and model
p1 <- ggplot(energy_data, aes(x = task_type, y = gpu_energy_j, fill = model)) +
  geom_boxplot(alpha = 0.8, outlier.shape = 21, outlier.size = 2) +
  labs(title = "GPU Energy Consumption by Task Type and Model",
       x = "Task Type",
       y = "GPU Energy (J)",
       fill = "Model") +
  scale_fill_brewer(palette = "Set2") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("boxplot_energy_task_model.png", p1, width = 12, height = 8, dpi = 300)

print(p1)

# Violin plot with boxplot
p2 <- ggplot(energy_data, aes(x = task_type, y = gpu_energy_j, fill = model)) +
  geom_violin(alpha = 0.5, trim = FALSE) +
  geom_boxplot(width = 0.1, alpha = 0.8, outlier.shape = NA) +
  labs(title = "Violin Plot of Energy Distribution",
       x = "Task Type",
       y = "GPU Energy (J)",
       fill = "Model") +
  scale_fill_brewer(palette = "Set3") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("violin_plot_energy.png", p2, width = 12, height = 8, dpi = 300)

print(p2)

# Density plots by model
p3 <- ggplot(energy_data, aes(x = gpu_energy_j, fill = task_type)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ model, scales = "free") +
  labs(title = "Density Distribution of Energy Consumption by Model",
       x = "GPU Energy (J)",
       y = "Density",
       fill = "Task Type") +
  scale_fill_brewer(palette = "Set1") +
  theme(strip.text = element_text(face = "bold"))

ggsave("density_plots_by_model.png", p3, width = 14, height = 10, dpi = 300)

print(p3)

# Faceted boxplot
p4 <- ggplot(energy_data, aes(x = task_type, y = gpu_energy_j, fill = task_type)) +
  geom_boxplot(alpha = 0.8) +
  facet_wrap(~ model, scales = "free_y") +
  labs(title = "Energy Consumption by Task Type Across Models",
       x = "Task Type",
       y = "GPU Energy (J)") +
  scale_fill_brewer(palette = "Pastel1") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

ggsave("faceted_boxplots.png", p4, width = 14, height = 10, dpi = 300)

print(p4)

# STATISTICAL ANALYSIS & ASSUMPTION CHECKING
# Create a function to check assumptions for each model
check_assumptions <- function(data, model_name) {
  cat(paste("\nASSUMPTION CHECKING FOR", model_name, "\n"))
  
  # Filter data for specific model
  model_data <- data %>% filter(model == model_name)
  
  # 1 Normality test for each task type
  normality_results <- model_data %>%
    group_by(task_type) %>%
    summarise(
      shapiro_p = shapiro.test(gpu_energy_j)$p.value,
      .groups = 'drop'
    )
  
  cat("\nNormality Tests (Shapiro-Wilk):\n")
  print(kable(normality_results, digits = 4))
  
  # 2 Homogeneity of variances (Levene's Test)
  levene_test <- leveneTest(gpu_energy_j ~ task_type, data = model_data)
  cat("\nLevene's Test for Homogeneity of Variances:\n")
  print(levene_test)
  
  # 3 Q-Q plots for visual inspection
  qq_plot <- ggplot(model_data, aes(sample = gpu_energy_j)) +
    stat_qq() + stat_qq_line() +
    facet_wrap(~ task_type, scales = "free") +
    labs(title = paste("Q-Q Plots for", model_name),
         x = "Theoretical Quantiles",
         y = "Sample Quantiles") +
    theme_minimal()
  
  ggsave(paste0("qq_plot_", gsub(" ", "_", model_name), ".png"), 
         qq_plot, width = 10, height = 8, dpi = 300)
  
  # Return test results
  return(list(
    normality = normality_results,
    homogeneity = levene_test,
    all_normal = all(normality_results$shapiro_p > 0.05),
    variances_homogeneous = levene_test$`Pr(>F)`[1] > 0.05
  ))
}

# Check assumptions for each model
models <- unique(energy_data$model)
assumption_results <- list()

for (m in models) {
  assumption_results[[m]] <- check_assumptions(energy_data, m)
}

# HYPOTHESIS TESTING
# Function to perform appropriate statistical test based on assumptions
perform_statistical_test <- function(data, model_name) {
  cat(paste("\n\n HYPOTHESIS TESTING FOR", model_name, "\n"))
  
  model_data <- data %>% filter(model == model_name)
  assumptions <- assumption_results[[model_name]]
  
  # Test choice based on assumptions
  if (assumptions$all_normal && assumptions$variances_homogeneous) {
    cat("Assumptions met: Using One-Way ANOVA\n")
    
    # One-Way ANOVA
    anova_result <- aov(gpu_energy_j ~ task_type, data = model_data)
    anova_summary <- summary(anova_result)
    print(anova_summary)
    
    # Post-hoc test (Tukey HSD)
    if (anova_summary[[1]]$`Pr(>F)`[1] < 0.05) {
      cat("\nSignificant ANOVA result - Performing Tukey HSD post-hoc test\n")
      tukey_result <- TukeyHSD(anova_result)
      print(tukey_result)
      
      # Extract p-values for correction
      tukey_pvals <- as.data.frame(tukey_result$task_type)$`p adj`
      
      return(list(
        test = "ANOVA",
        result = anova_summary,
        posthoc = tukey_result,
        p_values = tukey_pvals
      ))
    }
    
  } else {
    cat("Assumptions violated: Using Kruskal-Wallis test (non-parametric)\n")
    
    # Kruskal-Wallis test
    kruskal_result <- kruskal.test(gpu_energy_j ~ task_type, data = model_data)
    print(kruskal_result)
    
    # Post-hoc test (Dunn's test with Benjamini-Hochberg correction)
    if (kruskal_result$p.value < 0.05) {
      cat("\nSignificant Kruskal-Wallis result - Performing Dunn's post-hoc test\n")
      
      # Using PMCMRplus for Dunn's test with BH correction
      dunn_result <- kwAllPairsDunnTest(gpu_energy_j ~ task_type, 
                                        data = model_data,
                                        p.adjust.method = "BH")
      print(dunn_result)
      
      return(list(
        test = "Kruskal-Wallis",
        result = kruskal_result,
        posthoc = dunn_result
      ))
    }
  }
}

# Perform tests for each model
test_results <- list()
for (m in models) {
  test_results[[m]] <- perform_statistical_test(energy_data, m)
}

# EFFECT SIZE CALCULATION
# Function to calculate effect sizes
calculate_effect_sizes <- function(data, model_name) {
  cat(paste("\n\n EFFECT SIZE CALCULATION FOR", model_name, "\n"))
  
  model_data <- data %>% filter(model == model_name)
  
  # Eta-squared for ANOVA or epsilon-squared for Kruskal-Wallis
  if (test_results[[model_name]]$test == "ANOVA") {
    # Eta-squared
    anova_result <- test_results[[model_name]]$result
    ss_between <- anova_result[[1]]$`Sum Sq`[1]
    ss_total <- sum(anova_result[[1]]$`Sum Sq`)
    eta_squared <- ss_between / ss_total
    
    # Partial eta-squared
    df_between <- anova_result[[1]]$Df[1]
    df_error <- anova_result[[1]]$Df[2]
    partial_eta_squared <- ss_between / (ss_between + anova_result[[1]]$`Sum Sq`[2])
    
    cat(sprintf("Eta-squared: %.4f\n", eta_squared))
    cat(sprintf("Partial Eta-squared: %.4f\n", partial_eta_squared))
    
    return(list(
      eta_squared = eta_squared,
      partial_eta_squared = partial_eta_squared
    ))
    
  } else {
    # Epsilon-squared for Kruskal-Wallis
    kruskal_result <- test_results[[model_name]]$result
    n_total <- nrow(model_data)
    h_stat <- kruskal_result$statistic
    epsilon_squared <- (h_stat - (length(unique(model_data$task_type)) - 1)) / 
      (n_total - 1)
    
    cat(sprintf("Epsilon-squared (for Kruskal-Wallis): %.4f\n", epsilon_squared))
    
    return(list(
      epsilon_squared = epsilon_squared
    ))
  }
}

# Calculate effect sizes for each model
effect_sizes <- list()
for (m in models) {
  effect_sizes[[m]] <- calculate_effect_sizes(energy_data, m)
}

# MULTIPLE COMPARISONS CORRECTION
# Collect all p-values from post-hoc tests across models
all_p_values <- c()

for (m in models) {
  if (!is.null(test_results[[m]]$posthoc)) {
    if (test_results[[m]]$test == "ANOVA") {
      # Extract p-values from Tukey HSD
      tukey_df <- as.data.frame(test_results[[m]]$posthoc$task_type)
      all_p_values <- c(all_p_values, tukey_df$`p adj`)
    } else {
      # Extract p-values from Dunn's test
      dunn_matrix <- as.matrix(test_results[[m]]$posthoc$p.value)
      p_vals <- dunn_matrix[lower.tri(dunn_matrix)]
      all_p_values <- c(all_p_values, p_vals)
    }
  }
}

# Apply Benjamini-Hochberg correction
if (length(all_p_values) > 0) {
  bh_corrected <- p.adjust(all_p_values, method = "BH")
  
  cat("Original p-values:", sprintf("%.4f", all_p_values), "\n")
  cat("BH-corrected p-values:", sprintf("%.4f", bh_corrected), "\n")
  cat("Number of significant comparisons (alpha = 0.05):", 
      sum(bh_corrected < 0.05), "\n")
}

# VISUALIZATION - PAIRWISE COMPARISONS
# Create pairwise comparison plots for significant models
create_pairwise_plot <- function(data, model_name) {
  model_data <- data %>% filter(model == model_name)
  
  # Calculate mean and confidence intervals
  summary_data <- model_data %>%
    group_by(task_type) %>%
    summarise(
      mean_energy = mean(gpu_energy_j),
      se = sd(gpu_energy_j) / sqrt(n()),
      ci_lower = mean_energy - 1.96 * se,
      ci_upper = mean_energy + 1.96 * se,
      .groups = 'drop'
    )
  
  p <- ggplot(summary_data, aes(x = task_type, y = mean_energy)) +
    geom_point(size = 3, color = "darkblue") +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                  width = 0.2, color = "darkblue") +
    labs(title = paste("Mean Energy Consumption with 95% CI -", model_name),
         x = "Task Type",
         y = "Mean GPU Energy (J) ± 95% CI") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}

# Generate and save pairwise plots
for (m in models) {
  pairwise_plot <- create_pairwise_plot(energy_data, m)
  ggsave(paste0("pairwise_", gsub(" ", "_", m), ".png"), 
         pairwise_plot, width = 10, height = 6, dpi = 300)
}

# RESULTS SUMMARY AND EXPORT
# Create a comprehensive results table
create_results_table <- function() {
  results_df <- data.frame()
  
  for (m in models) {
    model_data <- energy_data %>% filter(model == m)
    
    # Statistical test information
    test_type <- test_results[[m]]$test
    if (test_type == "ANOVA") {
      p_value <- test_results[[m]]$result[[1]]$`Pr(>F)`[1]
    } else {
      p_value <- test_results[[m]]$result$p.value
    }
    
    # Effect size
    if (test_type == "ANOVA") {
      effect_size <- effect_sizes[[m]]$eta_squared
    } else {
      effect_size <- effect_sizes[[m]]$epsilon_squared
    }
    
    # Assumptions
    assumptions <- assumption_results[[m]]
    
    row <- data.frame(
      Model = m,
      n_observations = nrow(model_data),
      test_used = test_type,
      p_value = p_value,
      effect_size = effect_size,
      normality_assumption = assumptions$all_normal,
      homogeneity_assumption = assumptions$variances_homogeneous,
      significant = p_value < 0.05
    )
    
    results_df <- rbind(results_df, row)
  }
  
  return(results_df)
}

# Generate and display results table
final_results <- create_results_table()
cat("\n\n FINAL RESULTS SUMMARY \n")
print(kable(final_results, digits = 4))

# SUMMARY
for (m in models) {
  model_results <- final_results %>% filter(Model == m)
  cat(paste("\n-", m, ":"))
  cat(paste("\n  Test used:", model_results$test_used))
  cat(paste("\n  P-value:", sprintf("%.4f", model_results$p_value)))
  cat(paste("\n  Significant:", ifelse(model_results$significant, "YES", "NO")))
  cat(paste("\n  Effect size:", sprintf("%.4f", model_results$effect_size)))
  
  # Interpretation of effect size
  if (model_results$effect_size < 0.01) {
    cat(" (negligible)")
  } else if (model_results$effect_size < 0.06) {
    cat(" (small)")
  } else if (model_results$effect_size < 0.14) {
    cat(" (medium)")
  } else {
    cat(" (large)")
  }
}

