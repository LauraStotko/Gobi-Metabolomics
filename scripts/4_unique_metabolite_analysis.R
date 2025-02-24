# Load necessary libraries
source("scripts/0_load_libraries.R")

# Function to load the input data
load_input_data <- function() {
  anova_results_combined <- fread("results/anova_results_summary.csv")
  all_metabolites <- fread("results/all_metabolites_significance.csv")
  met_data_filtered <- fread("data/processed/reshaped_met_data.csv")
  
  return(list(anova_results_combined = anova_results_combined, 
              all_metabolites = all_metabolites, 
              met_data_filtered = met_data_filtered))
}

# Function to process metabolite data and compute log2 fold change and p-value
generate_log2foldchange_pvalue_data <- function(anova_results_combined, all_metabolites, met_data) {
  all_metabolites_fig <- all_metabolites
  
  # Add a new column to count TRUE values in significant columns (for each challenge)
  all_metabolites_fig$true_count <- rowSums(all_metabolites_fig[, c("significant_fasting", "significant_pat", "significant_oltt")] == TRUE)

  # Filter rows where only one of the significant columns is TRUE
  all_metabolites_fig <- all_metabolites_fig[true_count == 1]
  
  # Remove the redundant challenge columns and significance columns as they are not needed
  anova_results_combined[, c("challenge.x", "challenge.y", "challenge", "significant.x", "significant.y", "significant") := NULL]

  # Check if the columns exist before removing
  cols_to_remove <- c(
    "significant_fasting_pat", "significant_fasting_oltt", "significant_pat_oltt" 
  ) 
  # Remove columns if they exist
  existing_cols <- cols_to_remove[cols_to_remove %in% names(all_metabolites_fig)]
  if (length(existing_cols) > 0) {
    all_metabolites_fig[, (existing_cols) := NULL]
  }

  # Merge the p-values into the all_metabolites dataframe
  all_metabolites_fig <- merge(all_metabolites_fig, anova_results_combined, by = c("metabolite", "platform_name", "super_pathway", "sub_pathway"), all.x = TRUE)
  
  # Merge all_metabolites with met_data based on metabolite and platform
  all_metabolites_fig <- merge(all_metabolites_fig, met_data[, .(metabolite, platform_name, subject, challenge_time, challenge, response)], 
                                by = c("metabolite", "platform_name"), all.x = TRUE)
  
  # Calculate the mean for each metabolite, platform, challenge, and time
  all_metabolites_fig[, mean_response := mean(response, na.rm = TRUE), by = .(metabolite, platform_name, challenge, challenge_time)]
  
  # Remove 'subject' and 'response' columns
  all_metabolites_fig <- all_metabolites_fig[, !c("subject", "response"), with = FALSE]
  
  # Remove duplicates based on all columns
  all_metabolites_fig <- unique(all_metabolites_fig)
  
  # Calculate log2_foldchange based on the difference in mean_response
  all_metabolites_fig[, log2_foldchange := mean_response - mean_response[challenge_time == 0], by = .(metabolite, platform_name, challenge)]
  
  # Replace NA in mean_response with 0 if time is 0
  all_metabolites_fig[challenge_time == 0 & is.na(mean_response), mean_response := 0]
  
  # Create a new column 'p_value' based on the 'challenge' column
  all_metabolites_fig[, p_value := ifelse(challenge == "Fasting", p_value_fasting,
                                           ifelse(challenge == "PAT", p_value_pat,
                                                  ifelse(challenge == "OLTT", p_value_oltt, NA)))] 
  
  # Now remove the original p_value columns (p_value_fasting, p_value_pat, p_value_oltt)
  all_metabolites_fig[, c("p_value_fasting", "p_value_pat", "p_value_oltt") := NULL]
  
  # Create a new column 'neg_log10_p_value' that takes -log10 of the 'p_value' column
  all_metabolites_fig[, neg_log10_p_value := -log10(p_value)]
  
  # Create a new column 'abs_log2_foldchange' that stores the absolute value of 'log2_foldchange'
  all_metabolites_fig[, abs_log2_foldchange := abs(log2_foldchange)]
  
  # Check if the columns exist before removing
  cols_to_remove <- c("significant_any_challenge", "significant_fasting_pat_oltt")

  # Remove columns if they exist
  existing_cols <- cols_to_remove[cols_to_remove %in% names(all_metabolites_fig)]
  if (length(existing_cols) > 0) {
    all_metabolites_fig[, (existing_cols) := NULL]
  }

  # Remove duplicates based on all columns
  all_metabolites_fig <- unique(all_metabolites_fig)

  # Add 'challenge' column based on the significant flags for each challenge
  fasting_data <- all_metabolites_fig[significant_fasting == TRUE, 
                                      c("metabolite", "platform_name", "super_pathway", "challenge")]
  fasting_data$challenge <- "Fasting"

  pat_data <- all_metabolites_fig[significant_pat == TRUE, 
                                  c("metabolite", "platform_name", "super_pathway", "challenge")]
  pat_data$challenge <- "PAT"

  oltt_data <- all_metabolites_fig[significant_oltt == TRUE, 
                                   c("metabolite", "platform_name", "super_pathway", "challenge")]
  oltt_data$challenge <- "OLTT"

  # Combine all the data and remove duplicates
  unique_data <- rbind(fasting_data, pat_data, oltt_data)

  # Remove duplicates
  unique_data <- unique(unique_data)

  # Save filtered unique data as a separate file
  fwrite(unique_data, "results/unique_metabolites_per_challenge.csv")

  # Select only the necessary columns from all_metabolites_fig
  columns_to_merge <- c("metabolite", "platform_name", "sub_pathway", "challenge", "challenge_time", "neg_log10_p_value", "abs_log2_foldchange")

  # Create a subset of all_metabolites_fig with only the needed columns
  all_metabolites_subset <- all_metabolites_fig[, columns_to_merge, with = FALSE]

  # Merge the unique_data with the subset of all_metabolites_fig by "metabolite", "platform_name", and "challenge"
  merged_data <- merge(unique_data, all_metabolites_subset, by = c("metabolite", "platform_name", "challenge"), all.x = TRUE)

  # Return the processed data frame
  return(merged_data)
}


# Function to find top metabolites based on log2 fold change and -log10 p-value
find_top_metabolites <- function(unique_data) {
  top_metabolites_by_log2fc <- unique_data %>%
    group_by(super_pathway, sub_pathway, challenge) %>%
    dplyr::arrange(desc(abs_log2_foldchange)) %>%
    slice_head(n = 1) %>%
    ungroup()
  
  top_metabolites_by_pvalue <- unique_data %>%
    group_by(super_pathway, sub_pathway, challenge) %>%
    dplyr::arrange(desc(neg_log10_p_value)) %>%
    slice_head(n = 1) %>%
    ungroup()
  
  # Sort and select columns
  top_metabolites_by_log2fc <- top_metabolites_by_log2fc %>%
    dplyr::arrange(desc(abs_log2_foldchange)) %>%
    dplyr::select(super_pathway, sub_pathway, challenge, metabolite, platform_name, abs_log2_foldchange)
  
  top_metabolites_by_pvalue <- top_metabolites_by_pvalue %>%
    dplyr::arrange(desc(neg_log10_p_value)) %>%
    dplyr::select(super_pathway, sub_pathway, challenge, metabolite, platform_name, neg_log10_p_value)
  
  # Save results
  fwrite(top_metabolites_by_log2fc, "results/sup_table_top_metabolites_sorted_by_log2_foldchange.csv")
  fwrite(top_metabolites_by_pvalue, "results/sup_table_top_metabolites_sorted_by_p_value.csv")
  
  return(list(log2_foldchange_sorted = top_metabolites_by_log2fc, p_value_sorted = top_metabolites_by_pvalue))
}

# Function to filter and prepare data for selected metabolites
prepare_selected_metabolites_data <- function(met_data_unique_analysis, selected_metabolites) {
  met_data_unique_analysis[, mean_response := mean(response, na.rm = TRUE), 
                           by = .(metabolite, platform_name, challenge, challenge_time)]
  
  met_data_unique_analysis <- met_data_unique_analysis[, !c("subject", "response"), with = FALSE]
  met_data_unique_analysis <- unique(met_data_unique_analysis)
  
  # Calculate log2_foldchange
  met_data_unique_analysis[, log2_foldchange := mean_response - mean_response[challenge_time == 0],
                           by = .(metabolite, platform_name, challenge)]
  
  filtered_data <- met_data_unique_analysis %>%
    filter(metabolite %in% selected_metabolites)
  
  filtered_data <- filtered_data %>%
    filter(!is.na(time)) %>%
    mutate(log2_foldchange = abs(log2_foldchange))
  
  filtered_data$challenge <- factor(filtered_data$challenge, levels = c("Fasting", "PAT", "OLTT"))
  
  return(filtered_data)
}

# Function to generate plots for selected metabolites
generate_metabolite_plot <- function(filtered_data, selected_metabolites) {
  ggplot(filtered_data, aes(x = time, y = log2_foldchange, color = challenge, group = metabolite)) +
    geom_line(alpha = 0.7) +  # Lines for metabolites
    geom_point(size = 3, alpha = 0.7) +  # Points for metabolites
    labs(
      x = "Time Point", 
      y = "Absolute log2 Fold Change", 
      title = selected_metabolites  # Only the metabolite name as title
    ) +
    scale_color_manual(
      values = c("Fasting" = "red", "PAT" = "blue", "OLTT" = "#BA8E23"),  # Custom colors for each challenge
      name = "Challenge"  # Custom title for the legend
    ) +
    theme_minimal() +
    theme(
      legend.position = "top", 
      axis.text.x = element_text(hjust = 1), # Rotate x-axis labels for better readability
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      axis.line = element_line(color = "black", size = 1), # Ensure x and y axes are shown
      axis.ticks = element_line(color = "black", size = 1), # Make ticks visible
      legend.title = element_text(size = 14, face = "bold"),  # Make legend title larger
      legend.text = element_text(size = 12),  # Make legend text larger
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),  # Center the title and make it bold
      strip.text = element_text(size = 12, face = "bold"),
      strip.background = element_blank()  # Remove the background boxes around facet labels
    ) +
    facet_wrap(~challenge, scales = "free_x", nrow = 1) +  # Separate the x-axis by challenge without titles
    scale_x_continuous(
      breaks = unique(filtered_data$time),  # Automatically identify unique time points as ticks
      labels = unique(filtered_data$time)  # Automatically use those unique time points as labels
    )  # Ensure automatic ticks are shown on the x-axis

  # Save the plot
  ggsave("results/plots/fig_unique_challenge_o-acetylhomoserine.png", width = 15, height = 6, units = "in", dpi = 300, bg = "white")
}


main <- function() {
  # Load the input data
  input_data <- load_input_data()

  # Generate the log2 fold change and p-value data
  unique_data <- generate_log2foldchange_pvalue_data(input_data$anova_results_combined, 
                                                     input_data$all_metabolites, 
                                                     input_data$met_data_filtered)

  # Find the top metabolites
  top_metabolites <- find_top_metabolites(unique_data)

  # Prepare the selected metabolites data
  filtered_data <- prepare_selected_metabolites_data(input_data$met_data_filtered, c("o-acetylhomoserine"))

  # Generate the metabolite plot for 'fumarate'
  generate_metabolite_plot(filtered_data, "o-acetylhomoserine")
}

# Run the main function
main()