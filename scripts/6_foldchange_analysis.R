# Load necessary libraries
source("scripts/0_load_libraries.R")

# Define time point shapes with distinct shapes for each time point
time_point_shapes <- c(
  "30" = 16,  
  "60" = 17,  
  "120" = 15, 
  "240" = 5, 
  "1920" = 18
)

super_pathway_colors <- c(
  "Amino Acids" = "#d62728",
  "Carbohydrates" = "#7f7f7f",
  "Cofactors and Vitamins" = "#9467bd",
  "Energy" = "#8c564b",
  "Lipids" = "#ba8e23",
  "Nucleotides" = "#e377c2",
  "Peptides" = "#2ca02c",
  "Xenobiotics" = "#1f77b4"
)

# Define challenge colors
challenge_colors <- c("Fasting" = "red",  
                      "PAT" = "blue",  
                      "OLTT" = "#BA8E23") 

# Custom labels for challenge
challenge_labels <- c("Fasting" = "Fasting",
                      "PAT" = "Physical Activity",
                      "OLTT" = "Oral lipid tolerance test") 

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
process_and_save_figure2_data <- function(all_metabolites, anova_results_combined, met_data, output_path = "results/figure_2_table_logfc1.csv") {
  
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
  all_metabolites_fig <- all_metabolites_fig[neg_log10_p_value > 40 | (significant_any_challenge == TRUE & abs_log2_foldchange > 1)]
  
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

# Function to generate Figure 2a (volcano plot)
generate_figure2a <- function(all_metabolites_fig, output_path = "results/plots/volcano_plot_figure.png") {
  # Create volcano plot
  volcano_plot <- ggplot(all_metabolites_fig, aes(x = log2_foldchange, y = neg_log10_p_value, 
                     color = challenge, shape = as.factor(challenge_time))) +
    
    # Add transparent rectangle
    annotate("rect", xmin = -1, xmax = 1, ymin = 0, ymax = Inf, alpha = 0.2, fill = "blue") +
    
    # Add dashed vertical lines at x = -1 and x = 1
    geom_vline(xintercept = -1, linetype = "dashed", color = "black", size = 0.5) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "black", size = 0.5) +
    
    # Add dashed horizontal line for significance threshold
    geom_hline(yintercept = -log10(0.05/634), linetype = "dashed", color = "black", size = 0.5) +
    
    # Add points (ensure this is after the rectangle and lines to be on top)
    geom_point(size = 3, alpha = 1) +  # Set alpha to 1 for fully opaque points
    
    # Color and shape scales
    scale_color_manual(values = challenge_colors, labels = challenge_labels) +  
    scale_shape_manual(values = time_point_shapes) +
    
    # Labels
    labs(x = "log2 fold change", 
         y = "-log10(p-value)",
         color = "Challenge", 
         shape = "Challenge time [min]") +
    
    # Adjust x and y axes limits
    scale_y_continuous(expand = c(0, 0) , limits = c(0, 90)) +
    
    # Theme settings
    theme_minimal() +
    theme(
      legend.position = "top",  # Legend stays at the top
      legend.justification = "left",
      legend.box = "horizontal",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 10),
      
      axis.line = element_line(color = "black", size = 1),
      axis.ticks = element_line(color = "black", size = 1),  # Ensure ticks are visible
      axis.title = element_text(size = 14, face = "bold"),  # Increase axis title size
      axis.text = element_text(size = 14)  # Increase size of the numbers next to ticks
    ) +
    
    # Organize legends properly
    guides(
      color = guide_legend(ncol = 1, order = 1, title.position = "top"),  # Challenge legend
      shape = guide_legend(ncol = 2, order = 2, title.position = "top")   # Challenge time legend
    )

  # Save the volcano plot
  ggsave(output_path, plot = volcano_plot, bg = "white", width = 10, height = 8, units = "in")
  
  # Return the volcano plot object in case the user wants to further customize or display it
  return(volcano_plot)
}

library(ggplot2)
library(colorspace)

# Function to generate distinct shades for sub-pathways within a given super_pathway
generate_subpathway_colors <- function(data_subset, color_choice) {
  # Define a mapping of colors to palettes
  color_palettes <- list(
    red = "Reds",
    grey = "Light Grays",
    purple = "Purp",
    brown = "Peach",
    gold = "Heat",
    rose = "PuRd",
    green = "Greens",
    blue = "Teal"
  )
  
  # Check if the provided color_choice exists in the mapping
  if (!(color_choice %in% names(color_palettes))) {
    stop("Invalid color choice. Please select from: red, grey, purple, brown, gold, rose, green, or blue.")
  }
  
  # Extract unique sub_pathways and order them by their position
  sub_pathways <- unique(data_subset$sub_pathway)
  
  # Ensure that sub-pathways are ordered by their occurrence in the list
  sub_pathways_ordered <- sort(sub_pathways, decreasing = FALSE)
  
  # Determine the number of colors needed
  num_colors <- length(sub_pathways_ordered)
  
  # Select the appropriate palette
  palette <- color_palettes[[color_choice]]
  
  # Generate color shades using the selected palette, starting light and getting darker
  sub_pathway_colors <- sequential_hcl(num_colors, palette = palette, rev = TRUE)  # Reversing to ensure darker shades for later items
  
  # Assign colors to each sub_pathway based on their order
  named_colors <- setNames(sub_pathway_colors, sub_pathways_ordered)
  
  return(named_colors)
}

# Create plot for sub-pathway categorized data
create_sub_pathway_plot <- function(data_subset, color_choice, title) {
  subpathway_colors <- generate_subpathway_colors(data_subset, color_choice)

  p <- ggplot(data_subset, aes(x = log2_foldchange, y = metabolite)) +
    # Classification bar (sub_pathway)
    geom_tile(aes(x = -2.4, fill = sub_pathway), width = 0.15, height = 1) +  
    scale_fill_manual(values = subpathway_colors, name = "Sub Pathway") +
    
    # Shaded region
    annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, alpha = 0.3, fill = "gray80") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 0.3) +
    geom_point(aes(color = challenge, shape = as.factor(challenge_time)), size = 2, alpha = 0.8) +
  
    # Proper legend integration
    scale_color_manual(
      values = challenge_colors, 
      labels = c("Fasting" = "Fasting",
                 "PAT" = "Physical Activity",
                 "OLTT" = "Oral lipid tolerance test")
    ) +
    scale_shape_manual(
      values = time_point_shapes,
      guide = guide_legend(title = "Challenge time [min]", order = 2)
    ) +
  
    # Labels
    labs(title = title,
         x = "log2 fold change",
         y = "Metabolites",
         color = "Challenge", 
         shape = "Challenge time [min]") +
  
    theme_bw() +
    theme(
      legend.position = "top",  
      legend.justification = "left",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 10),
      
      axis.text.x = element_text(size = 11),
      axis.text.y = element_text(size = 9),
      axis.title.x = element_text(size = 10, face = "bold"),
      axis.title.y = element_text(size = 10, face = "bold"),
      
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      
      panel.border = element_blank(),
      panel.spacing = unit(0.01, "null"), 
      axis.line = element_line(color = "black"),
      
      strip.text.y.left = element_blank(),
      strip.placement = "outside"
    ) +
  
    # Force legends into specific order and display them in one line
    guides(
      shape = guide_legend(ncol = 2, order = 1, title.position = "top"),
      color = guide_legend(ncol = 1, order = 1, title.position = "top"),    
      fill = guide_legend(ncol = 2, order = 2, title.position = "top", title = "Sub Pathway")
    ) +
  
    # Adjust layout for legends to be placed on one row
    theme(
      legend.box = "horizontal",  # Legends in a horizontal box
      legend.box.spacing = unit(0.5, "lines"),  # Reduced spacing between the legend items
      legend.position = "top",  # Legends on the top
      legend.justification = "right",  # Center the legends
      legend.spacing.x = unit(0.5, "mm"),  # Reduce space between legends
      legend.spacing.y = unit(0.5, "mm")  # Reduce vertical space between the challenge and challenge time legends
    ) +
    
    # Facet grid for sub-pathway
    facet_grid(rows = vars(sub_pathway), scales = "free_y", space = "free_y", switch = "y") +
    
    # Remove space between the y-axis and -3 by setting expand = c(0,0)
    scale_x_continuous(limits = c(-2.5, 2.9), expand = c(0, 0))  # This removes the padding before -3

  # Create directory if it doesn't exist
  if (!dir.exists("results")) {
    dir.create("results")
  }

  # Save the plot
  ggsave(paste0("results/plots/fig_2_B_", title, "_subpathway_plot.png"), 
         plot = p, width = 12, height = 15, dpi = 300, bg = "white")
  
  return(p)
}

# Function to generate figure with separate plots
generate_figure2b <- function(all_metabolites_fig, 
                              output_path_lipids = "results/plots/forest_plot_lipids.png", 
                              output_path_amino_acids = "results/plots/forest_plot_amino_acids.png") {
  
  # Define the custom order for super_pathway
  pathway_order <- c("Carbohydrates", "Nucleotides", "Xenobiotics", 
                     "Amino Acids", "Lipids", "Cofactors and Vitamins", "Energy", "Peptides")
  
  # Ensure the super_pathway is a factor with the specified order
  all_metabolites_fig$super_pathway <- factor(all_metabolites_fig$super_pathway, levels = pathway_order)
  
  # Sort the dataframe by the ordered super_pathway and alphabetically by metabolite within each super_pathway
  all_metabolites_fig <- all_metabolites_fig[order(all_metabolites_fig$super_pathway, all_metabolites_fig$metabolite), ]
  
  # Combine super_pathway and metabolite to ensure metabolites are grouped within the correct super_pathway order
  all_metabolites_fig$metabolite <- factor(all_metabolites_fig$metabolite, 
                                           levels = unique(all_metabolites_fig$metabolite[order(all_metabolites_fig$super_pathway, all_metabolites_fig$metabolite)]))
  
  # **Group datasets**
  lipids_data <- all_metabolites_fig[all_metabolites_fig$super_pathway == "Lipids", ]
  amino_acids_data <- all_metabolites_fig[all_metabolites_fig$super_pathway == "Amino Acids", ]
  
  # **Create plots**
  lipid_plot <- create_sub_pathway_plot(lipids_data, "gold", "Lipids")
  amino_acid_plot <- create_sub_pathway_plot(amino_acids_data, "red", "Amino Acids")

  return(list(lipid_plot, amino_acid_plot))
}


generate_figure2b_other <- function(all_metabolites_fig, output_path = "results/plots/forest_plot.png") {

  # Filter out metabolites from "Amino Acids" and "Lipids"
  all_metabolites_fig <- all_metabolites_fig %>%
    filter(!super_pathway %in% c("Amino Acids", "Lipids"))  # Exclude "Amino Acids" and "Lipids"
  
  # Define the custom order for super_pathway
  pathway_order <- c("Peptides", "Carbohydrates", "Nucleotides", "Xenobiotics", 
                     "Cofactors and Vitamins", "Energy")

  # Ensure the super_pathway is a factor with the specified order
  all_metabolites_fig$super_pathway <- factor(all_metabolites_fig$super_pathway, levels = pathway_order)
  
  # Sort the dataframe by the ordered super_pathway and alphabetically by metabolite within each super_pathway
  all_metabolites_fig <- all_metabolites_fig[order(all_metabolites_fig$super_pathway, all_metabolites_fig$metabolite), ]
  
  # Combine super_pathway and metabolite to ensure metabolites are grouped within the correct super_pathway order
  all_metabolites_fig$metabolite <- factor(all_metabolites_fig$metabolite, 
                                            levels = unique(all_metabolites_fig$metabolite[order(all_metabolites_fig$super_pathway, all_metabolites_fig$metabolite)]))
  
  # Create forest plot
  forest_plot <- ggplot(all_metabolites_fig, aes(x = log2_foldchange, y = metabolite)) +
    geom_tile(aes(x = -2.4, fill = super_pathway), width = 0.15, height = 1) +  # Super pathway bar
    scale_fill_manual(values = super_pathway_colors, name = "Super Pathway") +
    
    # Shaded region
    annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, alpha = 0.3, fill = "gray80") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 0.3) +
    geom_point(aes(color = challenge, shape = as.factor(challenge_time)), size = 2, alpha = 0.8) +
  
    # Proper legend integration
    scale_color_manual(
      values = challenge_colors, 
      labels = challenge_labels, 
      guide = guide_legend(title = "Challenge", order = 1)
    ) +
    scale_shape_manual(
      values = time_point_shapes,
      guide = guide_legend(title = "Challenge time [min]", order = 2)
    ) +
  
    # Labels
    labs(x = "log2 fold change",
         y = "Metabolites",
         color = "Challenge", 
         shape = "Challenge time [min]") +
  
    theme_bw() +
    theme(
      legend.position = "top",  # Legend stays at the top
      legend.justification = "left",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 10),
      
      axis.text.x = element_text(size = 11),
      axis.text.y = element_text(size = 9),
      axis.title.x = element_text(size = 10, face = "bold"),
      axis.title.y = element_text(size = 10, face = "bold"),
      
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "bold"),
      
      panel.border = element_blank(),
      panel.spacing = unit(0.01, "null"), 
      axis.line = element_line(color = "black"),
      
      strip.text.y.left = element_blank(),
      strip.placement = "outside"
    ) +
  
    # Force legends into two separate lists and move 'super_pathway' to the last position
    guides(
      color = guide_legend(ncol = 1, order = 1, title.position = "top"),  # Challenge (list on left)
      shape = guide_legend(ncol = 2, order = 2, title.position = "top"),   # Challenge time (2 columns)
      fill = guide_legend(order = 3, nrow = 3, title.position = "top", title = "Super Pathway")  # Super pathway last
    ) +
  
    # Facet grid for super_pathway
    facet_grid(rows = vars(super_pathway), scales = "free_y", space = "free_y", switch = "y") +

    # Remove space between the y-axis and -3 by setting expand = c(0,0)
    scale_x_continuous(limits = c(-2.5, 4.1), expand = c(0, 0))  # This removes the padding before -3
  
  # Save the plot
  ggsave(output_path, plot = forest_plot, bg = "white", width = 10, height = 12, units = "in")
  
  # Return the forest plot object for further customization or use
  return(forest_plot)
}

main <- function() {
  # Load input data
  input_data <- load_input_data()
  
  # Process and save Figure 2 data
  processed_data_fig2 <- process_and_save_figure2_data(input_data$all_metabolites, input_data$anova_results_combined, 
                                                       input_data$met_data_filtered, output_path = "results/logfc_table.csv")

  # Generate Figure 2a (volcano plot)
  generate_figure2a(processed_data_fig2, "results/plots/fig_2_A_volcano_plot_logfold_pvalue.png")
  
  # Generate Figure 2b (forest plot for lipids and amino acids)
  generate_figure2b(processed_data_fig2, 
                     "results/plots/fig_2_B_forest_plot_logfold_lipids.png", 
                     "results/plots/fig_2_B_forest_plot_logfold_amino_acids.png")
  
  # Generate Figure 2b for other pathways (if needed)
  generate_figure2b_other(processed_data_fig2, "results/plots/fig_2_B_forest_plot_other.png")
}

# Run the main function
main()