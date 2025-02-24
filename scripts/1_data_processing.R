# Load necessary libraries
source("scripts/0_load_libraries.R")

# Load data without needing info_file
load_data <- function(met_file) {
  met_data <- fread(met_file, sep = ",", fill = TRUE)
  return(met_data)
}

# Add challenge information
categorize_challenges <- function(met_data) {
  met_data <- met_data %>%
    mutate(challenge = case_when(
      time >= 1 & time <= 10 ~ "Fasting",
      time >= 33 & time <= 39 ~ "PAT",
      time >= 40 & time <= 50 ~ "OLTT",  
      TRUE ~ "Other"  
    ))
  return(met_data)
}

# Add challenge time based on challenge and time
add_challenge_time <- function(met_data) {
  met_data[, challenge_time := case_when(
    challenge == "Fasting" & time == 1 ~ 0,
    challenge == "Fasting" & time == 2 ~ 120,
    challenge == "Fasting" & time == 3 ~ 240,
    challenge == "Fasting" & time == 4 ~ 360,
    challenge == "Fasting" & time == 5 ~ 480,
    challenge == "Fasting" & time == 6 ~ 600,
    challenge == "Fasting" & time == 7 ~ 720,
    challenge == "Fasting" & time == 8 ~ 840,
    challenge == "Fasting" & time == 9 ~ 960,
    challenge == "Fasting" & time == 10 ~ 1920,
    challenge == "PAT" & time == 33 ~ 0,
    challenge == "PAT" & time == 34 ~ 15,
    challenge == "PAT" & time == 35 ~ 30,
    challenge == "PAT" & time == 36 ~ 45,
    challenge == "PAT" & time == 37 ~ 60,
    challenge == "PAT" & time == 38 ~ 90,
    challenge == "PAT" & time == 39 ~ 120,
    challenge == "OLTT" & time == 40 ~ 0,
    challenge == "OLTT" & time == 41 ~ 30,
    challenge == "OLTT" & time == 42 ~ 60,
    challenge == "OLTT" & time == 43 ~ 90,
    challenge == "OLTT" & time == 44 ~ 120,
    challenge == "OLTT" & time == 45 ~ 180,
    challenge == "OLTT" & time == 46 ~ 240,
    challenge == "OLTT" & time == 47 ~ 300,
    challenge == "OLTT" & time == 48 ~ 360,
    challenge == "OLTT" & time == 49 ~ 420,
    challenge == "OLTT" & time == 50 ~ 480,
    TRUE ~ NA_real_  # For all other cases, set NA
  )]
  return(met_data)
}

# Remove metabolites with > 30% NAs and save them into Suppelementary Table 1
remove_high_na_metabolites <- function(met_data, threshold = 0.3, output_file = "results/sup_table_1_removed_metabolites.csv") {

  metabolite_columns <- setdiff(colnames(met_data), c("time", "subject", "challenge", "challenge_time"))
  
  # Calculate the percentage of missing values for each metabolite
  na_percentage <- colMeans(is.na(met_data[, ..metabolite_columns]))

  # Create a data frame with metabolite names and their missingness percentage
  missingness_df <- data.frame(
    metabolite = names(na_percentage),
    missing_percentage = na_percentage * 100  # Convert to percentage
  )

  # Filter for metabolites that exceed the threshold for missingness
  high_na_metabolites_df <- missingness_df[missingness_df$missing_percentage > (threshold * 100), ]

  # Save the high missingness metabolites to a CSV file
  write.csv(high_na_metabolites_df, output_file, row.names = FALSE)

  # Find metabolites with more than `threshold` missing values
  high_na_metabolites <- names(na_percentage[na_percentage > threshold])

  # Remove these metabolites from met_data
  filtered_met_data <- met_data[, !high_na_metabolites, with = FALSE]

  return(filtered_met_data)
}

# Create a dataset with only relevant time intervals
filter_relevant_time_intervals <- function(met_data) {
  met_data <- met_data %>%
    filter(challenge != "Other")
  return(met_data)
}

# Function to filter metabolites based on platform
filter_metabolites <- function(met_data, pattern) {
  metabolite_columns <- setdiff(colnames(met_data), c("time", "subject", "challenge", "challenge_time"))
  selected_cols <- c("time", "subject", "challenge", "challenge_time", metabolite_columns[grepl(pattern, metabolite_columns)]) 
  met_data[, ..selected_cols]
}

# Function to filter metabolites NOT belonging to Metabolon or Biocrates (i.e., Inhouse)
filter_inhouse_metabolites <- function(met_data, platforms) {
  metabolite_columns <- setdiff(colnames(met_data), c("time", "subject", "challenge", "challenge_time"))
  excluded_cols <- unique(unlist(lapply(platforms, function(p) metabolite_columns[grepl(p, metabolite_columns)])))
  selected_cols <- c("time", "subject", "challenge", "challenge_time", setdiff(metabolite_columns, excluded_cols))
  met_data[, ..selected_cols]
}

# Function to convert categorical variables to factors
convert_to_factors <- function(data) {
  data %>%
    mutate(
      #challenge = as.factor(challenge),
      time = as.factor(time),
      subject = as.factor(subject)
    ) %>%
    mutate(across(where(is.character), as.factor))
}

# Function for missForest imputation with adaptive parallelization
perform_missForest <- function(data_subset, ntree_val = 10) {
  num_vars <- ncol(data_subset)  # Get the number of variables
  
  # Adjust cores to be at most the number of variables
  num_cores <- min(detectCores() - 1, num_vars)
  
  # If parallelization is still invalid, set it to 'no'
  parallel_option <- if (num_cores > 1) "variables" else "no"
  
  cl <- makeCluster(num_cores, type = "FORK") 
  registerDoParallel(cl)
  
  set.seed(42)  # Ensures reproducibility
  imputed_data <- missForest(data_subset, ntree = ntree_val, parallelize = parallel_option, verbose = TRUE)
  
  stopCluster(cl)  # Stop cluster
  
  return(imputed_data$ximp)  # Extract imputed dataset
}

# Wrapper function to process and impute metabolite datasets
impute_data <- function(metabolite_datasets) {
  # Convert categorical variables to factors
  metabolite_datasets <- lapply(metabolite_datasets, convert_to_factors)
  
  # Perform imputation with automatic parallelization adjustment
  imputed_data <- lapply(metabolite_datasets, perform_missForest, ntree_val = 400) 
  
  return(imputed_data)
}

# Function to calculate Z-scores
calculate_z_scores <- function(combined_data) {
  columns_to_zscore <- setdiff(names(combined_data), c("time", "subject", "challenge", "challenge_time"))
  combined_data[, (columns_to_zscore) := lapply(.SD, function(x) {
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  }), .SDcols = columns_to_zscore]
  return(combined_data)
}


# Main function to execute the pipeline
run_analysis_pipeline <- function(met_file) {
  load_libraries()
  # Load data
  data <- load_data(met_file)
  met_data <- data$met_data
  
  # Process data
  met_data <- remove_high_na_metabolites(met_data)
  met_data <- categorize_challenges(met_data)
  met_data <- add_challenge_time(met_data)

  # Remove Irrelevant Time Points
  met_data <- filter_relevant_time_intervals(met_data)

  # Filter datasets based on platform
  platforms <- list(
    metabolon = "\\[P, nt-ms\\]",
    biocrates = "\\[P, t-ms\\]"
  )
  met_data_metabolon <- filter_metabolites(met_data, platforms$metabolon)
  met_data_biocrates <- filter_metabolites(met_data, platforms$biocrates)
  met_data_inhouse <- filter_inhouse_metabolites(met_data, platforms)
  
  
  # List of metabolite datasets
  metabolite_datasets <- list(
    metabolon = met_data_metabolon,
    biocrates = met_data_biocrates,
    inhouse = met_data_inhouse
  )
  
  # Apply pipeline to each dataset
  imputed_metabolite_data <- impute_data(metabolite_datasets)
  
  # Merge datasets based on time, subject, and challenge
  combined_data <- Reduce(function(x, y) {
    merge(x, y, by = c("time", "subject", "challenge", "challenge_time"), all = TRUE)
  }, list(imputed_metabolite_data$metabolon, imputed_metabolite_data$biocrates, imputed_metabolite_data$inhouse))
  
  # Calculate Z-scores
  imputed_z_score_data <- calculate_z_scores(combined_data)

  return(imputed_z_score_data)
}

main <- function() {
  # Run the analysis pipeline with input files
  imputed_z_score_data <- run_analysis_pipeline("input/raw/humet_data_raw_none_subjects15_tp57.csv")
  
  # Save the resulting data to a CSV file
  fwrite(imputed_z_score_data, "data/processed/humet_imputed_400trees_z_score.csv")
}

# Call the main function
main()