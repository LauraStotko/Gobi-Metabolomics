# Load necessary libraries
source("scripts/0_load_libraries.R")

# Function to load the metabolite data
load_data <- function(met_file) {
  met_data <- fread(met_file, sep = ",", fill = TRUE)
  return(met_data)
}

# Function to preprocess the data (convert columns and define threshold)
preprocess_data <- function(met_data) {
  met_data[, time := as.factor(time)]
  met_data[, subject := as.factor(subject)]
  p_threshold <- 0.05 / nrow(unique(met_data[, .(metabolite, platform_name)]))
  return(list(met_data = met_data, p_threshold = p_threshold))
}

# Function to subset data by challenge
subset_data_by_challenge <- function(met_data) {
  metabolite_data_fasting <- met_data[challenge == "Fasting"]
  metabolite_data_pat <- met_data[challenge == "PAT"]
  metabolite_data_oltt <- met_data[challenge == "OLTT"]
  return(list(fasting = metabolite_data_fasting, pat = metabolite_data_pat, oltt = metabolite_data_oltt))
}

# Function to run ANOVA-like test for a given subset
run_anova_for_subset <- function(subset_data, challenge_name, p_threshold) {
  results <- list()
  unique_metabolites <- unique(subset_data[, .(metabolite, platform_name, super_pathway, sub_pathway)])
  
  for (i in seq_len(nrow(unique_metabolites))) {
    met <- unique_metabolites$metabolite[i]
    plat <- unique_metabolites$platform_name[i]
    super_path <- unique_metabolites$super_pathway[i]
    sub_path <- unique_metabolites$sub_pathway[i]
    
    # Subset data for this metabolite and platform
    subset <- subset_data[metabolite == met & platform_name == plat]
    if (nrow(subset) > 2) {
      # Run the ld.f1 test
      test_result <- ld.f1(y = subset$response, time = subset$time, subject = subset$subject, description = FALSE)
      
      # Extract p-value for time effect
      if (!is.null(test_result$ANOVA.test)) {
        p_value <- test_result$ANOVA.test$`p-value`
      } else {
        p_value <- NA  # If the test result is NULL, set p_value as NA
      }
      
      # Store results
      results[[paste(met, plat, sep = "_")]] <- data.table(
        challenge = challenge_name,
        metabolite = met,
        platform_name = plat,
        super_pathway = super_path,
        sub_pathway = sub_path,
        p_value = p_value
      )
    }
  }
  
  anova_results <- rbindlist(results, fill = TRUE)
  
  # Check if any metabolite-platform combination has an actual p_value
  anova_results <- anova_results[!is.na(p_value)]
  
  # Identify significant results
  anova_results[, significant := p_value < p_threshold]
  return(anova_results)
}

# Function to update significance status for matching metabolite-platform pairs
update_significance <- function(all_metabolites, sig_data) {
  if (nrow(sig_data) > 0) {  
    all_metabolites[sig_data, on = .(metabolite, platform_name), significant_any_challenge := TRUE]
  }
}

# Function to update individual significance columns for each challenge
update_significance_column <- function(all_metabolites, sig_data, column_name) {
  if (nrow(sig_data) > 0) {
    all_metabolites[sig_data, on = .(metabolite, platform_name), (column_name) := TRUE]
  }
}

# Function to merge ANOVA results for all challenges
merge_anova_results <- function(anova_results_fasting, anova_results_pat, anova_results_oltt) {
  anova_results_combined <- merge(anova_results_fasting, anova_results_pat, by = c("metabolite", "platform_name", "super_pathway", "sub_pathway"), all = TRUE)
  anova_results_combined <- merge(anova_results_combined, anova_results_oltt, by = c("metabolite", "platform_name", "super_pathway", "sub_pathway"), all = TRUE)
  setnames(anova_results_combined, 
           c("p_value.x", "p_value.y", "p_value"), 
           c("p_value_fasting", "p_value_pat", "p_value_oltt"))
  
  # Check for unique results per metabolite-platform combination
  anova_results_combined <- unique(anova_results_combined)
  
  return(anova_results_combined)
}

# Main function to orchestrate the analysis
main <- function(met_file) {
  # Load and preprocess data
  met_data <- load_data(met_file)
  preprocessed_data <- preprocess_data(met_data)
  
  # Subset data by challenge
  subset_data <- subset_data_by_challenge(preprocessed_data$met_data)
  
  # Run ANOVA-like test for each challenge
  anova_results_fasting <- run_anova_for_subset(subset_data$fasting, "Fasting", preprocessed_data$p_threshold)
  anova_results_pat <- run_anova_for_subset(subset_data$pat, "PAT", preprocessed_data$p_threshold)
  anova_results_oltt <- run_anova_for_subset(subset_data$oltt, "OLTT", preprocessed_data$p_threshold)
  
  # Get all unique metabolites from the updated dataset
  all_metabolites <- unique(met_data[, .(metabolite, platform_name, super_pathway, sub_pathway)])

  # Sort metabolites first by super_pathway, then sub_pathway, then metabolite name
  all_metabolites <- all_metabolites[order(super_pathway, sub_pathway, tolower(metabolite))]

  # Initialize the columns as FALSE for all metabolites
  all_metabolites[, `:=`(
    significant_any_challenge = FALSE,
    significant_fasting = FALSE,
    significant_pat = FALSE,
    significant_oltt = FALSE,
    significant_fasting_pat = FALSE,
    significant_fasting_oltt = FALSE,
    significant_pat_oltt = FALSE,
    significant_fasting_pat_oltt = FALSE
  )]

  # Extract **only** significant metabolites (ensuring metabolite-platform pairs match)
  significant_fasting <- anova_results_fasting[significant == TRUE, .(metabolite, platform_name)]
  significant_pat <- anova_results_pat[significant == TRUE, .(metabolite, platform_name)]
  significant_oltt <- anova_results_oltt[significant == TRUE, .(metabolite, platform_name)]

  # Update significance for each challenge
  update_significance_column(all_metabolites, significant_fasting, "significant_fasting")
  update_significance_column(all_metabolites, significant_pat, "significant_pat")
  update_significance_column(all_metabolites, significant_oltt, "significant_oltt")

  # Update the 'significant_any_challenge' column for any significant result
  update_significance(all_metabolites, significant_fasting)
  update_significance(all_metabolites, significant_pat)
  update_significance(all_metabolites, significant_oltt)

  # Calculate overlaps for significance
  all_metabolites[, `:=`(
    significant_fasting_pat = significant_fasting & significant_pat & !significant_oltt,
    significant_fasting_oltt = significant_fasting & significant_oltt & !significant_pat,
    significant_pat_oltt = significant_pat & significant_oltt & !significant_fasting,
    significant_fasting_pat_oltt = significant_fasting & significant_pat & significant_oltt
  )]
  
  # Merge results for all challenges
  anova_results_combined <- merge_anova_results(anova_results_fasting, anova_results_pat, anova_results_oltt)

  return(list(all_metabolites = all_metabolites, anova_results_combined = anova_results_combined))
}

# Example usage
result <- main("data/processed/reshaped_met_data.csv")
fwrite(result$all_metabolites, "results/all_metabolites_significance.csv")
fwrite(result$anova_results_combined, "results/anova_results_summary.csv")