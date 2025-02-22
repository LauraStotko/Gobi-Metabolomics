# Load necessary libraries
source("scripts/0_load_libraries.R")

# Function to load the input data
load_input_data <- function() {
  # Adjust the file paths for where your data files are located
  anova_results_combined <- fread("results/anova_results_summary.csv")
  all_metabolites <- fread("results/all_metabolites_significance.csv")
  met_data_filtered <- fread("data/processed/reshaped_met_data.csv")
  
  return(list(anova_results_combined = anova_results_combined, 
              all_metabolites = all_metabolites, 
              met_data_filtered = met_data_filtered))
}

# Function to process metabolite data and compute log2 fold change and p-value
generate_log2foldchange_pvalue_data <- function(anova_results_combined, all_metabolites, met_data) {
  
  # Remove the redundant challenge columns and significance columns as they are not needed
  anova_results_combined[, c("challenge.x", "challenge.y", "challenge", "significant.x", "significant.y", "significant") := NULL]

  all_metabolites_fig <- all_metabolites

  # Check if the columns exist before removing
  cols_to_remove <- c("significant_fasting", "significant_pat", "significant_oltt", 
                      "significant_fasting_pat", "significant_fasting_oltt", "significant_pat_oltt")

  # Remove columns if they exist
  existing_cols <- cols_to_remove[cols_to_remove %in% names(all_metabolites_fig)]
  if (length(existing_cols) > 0) {
    all_metabolites_fig[, (existing_cols) := NULL]
  }

  # Merge the p-values into the all_metabolites dataframe
  all_metabolites_fig <- merge(all_metabolites_fig, anova_results_combined, by = c("metabolite", "platform_name", "super_pathway", "sub_pathway"), all.x = TRUE)
  
  # Merge all_metabolites with met_data based on metabolite and platform
  all_metabolites_fig <- merge(all_metabolites_fig, met_data[, .(metabolite, platform_name, subject, challenge_time, 
                                                                challenge, response)], 
                                by = c("metabolite", "platform_name"), all.x = TRUE)
  
  # Calculate the mean for each metabolite, platform, challenge, and time
  all_metabolites_fig[, mean_response := mean(response, na.rm = TRUE), 
                         by = .(metabolite, platform_name, challenge, challenge_time)]
  
  # Remove 'subject' and 'response' columns
  all_metabolites_fig <- all_metabolites_fig[, !c("subject", "response"), with = FALSE]
  
  # Remove duplicates based on all columns
  all_metabolites_fig <- unique(all_metabolites_fig)
  
  # Calculate log2_foldchange based on the difference in mean_response
  all_metabolites_fig[, log2_foldchange := 
                        mean_response - mean_response[challenge_time == 0],
                      by = .(metabolite, platform_name, challenge)]
  
  # Replace NA in mean_response with 0 if time is 0
  all_metabolites_fig[challenge_time == 0 & is.na(mean_response), mean_response := 0]
  
  # Create a new column 'p_value' based on the 'challenge' column
  all_metabolites_fig[, p_value := 
      ifelse(challenge == "Fasting", p_value_fasting,
      ifelse(challenge == "PAT", p_value_pat,
      ifelse(challenge == "OLTT", p_value_oltt, NA)))] 
  
  # Now remove the original p_value columns (p_value_fasting, p_value_pat, p_value_oltt)
  all_metabolites_fig[, c("p_value_fasting", "p_value_pat", "p_value_oltt") := NULL]
  
  # Create a new column 'neg_log10_p_value' that takes -log10 of the 'p_value' column
  all_metabolites_fig[, neg_log10_p_value := -log10(p_value)]
  
  # Create a new column 'abs_log2_foldchange' that stores the absolute value of 'log2_foldchange'
  all_metabolites_fig[, abs_log2_foldchange := abs(log2_foldchange)]
  
  # Return the processed data frame
  return(all_metabolites_fig)
}

# Function to process and save Figure 2 data
process_and_save_figure2_data <- function(all_metabolites, anova_results_combined, met_data, output_path = "results/figure_2_table.csv") {
  
  # Generate figure 2 data
  all_metabolites_fig <- generate_log2foldchange_pvalue_data(anova_results_combined, all_metabolites, met_data)

  # Remove the unnecessary column if it exists
  col_to_remove <- "significant_fasting_pat_oltt"
  if (col_to_remove %in% names(all_metabolites_fig)) {
    all_metabolites_fig[, (col_to_remove) := NULL]
  }
  
  # Filter rows where platform is 'Metabolon'
  all_metabolites_fig <- all_metabolites_fig[platform_name %in% c("Metabolon HD4 [nt-ms]")]
  
  # Filter rows based on log2 fold change or p-value thresholds
  all_metabolites_fig <- all_metabolites_fig[neg_log10_p_value > 40 | (significant_any_challenge == TRUE & abs_log2_foldchange > 2)]
  
  # Order the data by platform and absolute log2 fold change
  setorder(all_metabolites_fig, -abs_log2_foldchange)

  # Remove the unnecessary column if it exists
  col_to_remove <- "significant_any_challenge"
  if (col_to_remove %in% names(all_metabolites_fig)) {
    all_metabolites_fig[, (col_to_remove) := NULL]
  }

  # Save the table to a CSV file
  fwrite(all_metabolites_fig, output_path)
  
  message("Processed data saved to: ", output_path)
  
  # Return the processed data table
  return(all_metabolites_fig)
}

# Function to preprocess the data
preprocess_data <- function(met_data_filtered, processed_data_fig2) {
  # Merge data and filter
  met_data_fig2_filtered <- merge(met_data_filtered, processed_data_fig2[, .(metabolite, platform_name, challenge)], 
                                   by = c("metabolite", "platform_name", "challenge"), all.x = FALSE)

  met_data_fig2_filtered <- unique(met_data_fig2_filtered)

  # Create a reference column for the response at time == 0
  met_data_fig2_filtered <- met_data_fig2_filtered %>%
    group_by(metabolite, platform_name, subject, challenge) %>%
    mutate(response_at_0 = response[challenge_time == 0][1]) %>%
    ungroup() %>%
    mutate(log2_foldchange = response - response_at_0, 
           abs_log2_foldchange = abs(log2_foldchange))

  return(met_data_fig2_filtered)
}


# Function to split data by challenge
split_data_by_challenge <- function(met_data_fig2_filtered) {
  met_data_fig2_filtered_split <- split(met_data_fig2_filtered, met_data_fig2_filtered$challenge)
  return(met_data_fig2_filtered_split)
}

# Prepare plot data
prepare_plot_data <- function(met_data_combined) {
  variance_data <- met_data_combined %>%
    group_by(metabolite, platform_name, challenge, challenge_time) %>%
    summarise(variance_log2fc = var(log2_foldchange, na.rm = TRUE)) %>%
    ungroup()

   # Print max variance for each challenge
  max_variance_by_challenge <- variance_data %>%
    group_by(challenge) %>%
    summarise(max_variance = max(variance_log2fc, na.rm = TRUE))

  print(max_variance_by_challenge)

  # Pivot the variance data into a wide format
  variance_wide <- variance_data %>%
    pivot_wider(names_from = challenge_time, values_from = variance_log2fc)
  
  # Reshape the data to long format for ggplot
  variance_long <- pivot_longer(variance_wide, 
                                cols = -c(metabolite, platform_name, challenge), 
                                names_to = "time_point", 
                                values_to = "variance_log2fc")

  variance_long$time_point <- factor(variance_long$time_point, 
                                      levels = c("0", "15", "30", "45", "60", "90", "120", "180", 
                                                "240", "300", "360", "420", "480", "600", "720", 
                                                "840", "960", "1920"))

  return(variance_long)
}

create_heatmap <- function(variance_long) {

  challenge_labels <- c(
  "Fasting" = "Fasting",
  "PAT" = "Physical Activity",
  "OLTT" = "Oral Lipid Tolerance Test" 
  )
  print(range(variance_long$variance_log2fc, na.rm = TRUE))  # Check range
  
  # Filter out rows where time_point is "0"
  variance_long <- variance_long %>%
    filter(time_point != "0")
  
  # Define color palette for the general heatmap
  color_palette <- c("white", "blue")  # Adjust this color palette to fit your preference

  # Example plot creation with correct ordering of 'challenge'
  p <- ggplot(variance_long, aes(x = time_point, y = metabolite, fill = variance_log2fc)) +  
    geom_tile(aes(fill = variance_log2fc), color = "gray") +  
    facet_wrap(~ challenge, scales = "free_x", ncol = 3, labeller = labeller(challenge = challenge_labels)) +  
    scale_fill_gradientn(
      colors = color_palette
      #limits = c(0, 15)  # Set the limits of the color scale
    ) +  
    labs(x = "Challenge time [min]", y = "Metabolite", fill = "Inter-individual variation") +  
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.ticks = element_line(color = "black", size = 1),
      axis.line = element_line(color = "black", size = 1),
      strip.background = element_blank(),
      strip.text = element_text(size = 12),
      panel.spacing = unit(2, "lines"),
      axis.title.y = element_text(size = 12, face = "bold"),
      axis.title.x = element_text(size = 12, face = "bold"),
      axis.text.y = element_text(size = 8),
      panel.background = element_blank(),
      plot.background = element_blank(),
      panel.grid = element_blank(),
      legend.position = "top",
      legend.title = element_text(size = 12, face = "bold"),
            legend.text = element_text(size = 10)
    ) +
    guides(fill = guide_colorbar(
      title = "Inter-individual variation",
      title.position = "top",
      title.hjust = 0.5,
      ticks = FALSE
    ))

  return(p)
}

# Function to plot the histogram with custom challenge colors
plot_histogram_by_challenge <- function(variance_long) {
  # Define challenge colors
  challenge_colors <- c("Fasting" = "red",  
                        "PAT" = "blue",  
                        "OLTT" = "#BA8E23")  # Customize as needed

  # Custom labels for challenge
  challenge_labels <- c("Fasting" = "Fasting",
                        "PAT" = "Physical Activity",
                        "OLTT" = "Oral Lipid Tolerance Test")

  ggplot(variance_long, aes(x = variance_log2fc, fill = challenge)) +
    geom_histogram(binwidth = 1, color = "white", alpha = 0.7, position = "dodge") +  # Remove black border from bars
    scale_fill_manual(values = challenge_colors, labels = challenge_labels) +  # Apply custom colors and labels
    labs(title = "Distribution of Variance by Challenge", 
         x = "Variance log2 Fold Change", 
         y = "Frequency", 
         fill = "Challenge") +
    facet_wrap(~ challenge, scales = "free_x", labeller = labeller(challenge = challenge_labels)) +  # Facet by challenge with custom labels
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 14),  # Increase size of x-axis numbers
      axis.text.y = element_text(size = 14),  # Increase size of y-axis numbers
      axis.title.x = element_text(size = 16, face = "bold"),  # Make x-axis title larger and bold
      axis.title.y = element_text(size = 16, face = "bold"),  # Make y-axis title larger and bold
      axis.line = element_line(color = "black", size = 0.5),  # Add black axis lines
      axis.ticks = element_line(color = "black", size = 0.5),  # Add axis ticks
      strip.text = element_text(size = 14),  # Increase size of facet labels
      legend.position = "top", 
      legend.title = element_text(size = 12, face = "bold"),  # Make legend title bigger
      legend.text = element_text(size = 12)  # Increase size of legend text
    ) +
    scale_x_continuous(limits = c(0, NA), expand = c(0, 0)) +  # x-axis starts from 0
    scale_y_continuous(limits = c(0, 150), expand = c(0, 0))   # y-axis starts from 0
}


# Function to save the plot to a file
save_plot <- function(plot, filename) {
  ggsave(filename, plot = plot, bg = "white", width = 13, height = 15, units = "in", dpi = 300) #height 6
}

run_pipeline <- function(met_data_filtered, processed_data_fig2) {
  met_data_fig2_filtered <- preprocess_data(met_data_filtered, processed_data_fig2)
  met_data_fig2_filtered_split <- split_data_by_challenge(met_data_fig2_filtered)
  variance_long <- bind_rows(lapply(met_data_fig2_filtered_split, prepare_plot_data))
  save_plot(create_heatmap(variance_long), "results/plots/sup_fig_1_heatmap.png")
  #save_plot(plot_histogram_by_challenge(variance_long), "results/plots/sup_fig_1_variance_histogram_all.png")
}

main <- function() {
  # Load the input data
  input_data <- load_input_data()

  # Process and save Figure 2 data
  processed_data_fig2 <- process_and_save_figure2_data(input_data$all_metabolites, 
                                                       input_data$anova_results_combined, 
                                                       input_data$met_data_filtered, 
                                                       output_path = "results/inter_individual_analysis_table.csv")
  
  # Run the pipeline with the processed data
  run_pipeline(input_data$met_data_filtered, processed_data_fig2)
}

# Run the main function
main()