# Load necessary libraries
source("scripts/0_load_libraries.R")

# Function to load the data
load_data <- function(z_score_data, info_file) {
  z_score_data <- fread(z_score_data, sep = ",", fill = TRUE)
  info_data <- fread(info_file, sep = ",", fill = TRUE)
  return(list(z_score_data = z_score_data, info_data = info_data))
}

# Function to add platform information and reshape each dataset into long format
reshape_long <- function(data) {
  # Identify metabolite columns (exclude time, subject, challenge)
  metabolite_columns <- setdiff(names(data), c("time", "subject", "challenge", "challenge_time"))
  
  # Convert all metabolite columns to numeric (preserves NA values)
  data[, (metabolite_columns) := lapply(.SD, as.numeric), .SDcols = metabolite_columns]
  
  # Reshape into long format
  long_data <- melt(data,
                    id.vars = c("time", "subject", "challenge", "challenge_time"),  # Keep these columns unchanged
                    measure.vars = metabolite_columns,  # Only reshape metabolite columns
                    variable.name = "metabolite",
                    value.name = "response",
                    na.rm = FALSE)  # Keep NA values instead of removing them
  
  # Add platform name based on the metabolite column name
  long_data[, platform_name := case_when(
    grepl("\\[P, t-ms\\]", metabolite) ~ "Biocrates p150 [t-ms]",
    grepl("\\[P, nt-ms\\]", metabolite) ~ "Metabolon HD4 [nt-ms]",
    grepl("\\[P, chem.\\]", metabolite) ~ "In-house biochemistry [chem.]",
    TRUE ~ "Unknown"  # Default case for anything that doesn't match
  )]
  
  return(long_data)
}

# Function to clean metabolite names
clean_metabolite_names <- function(met_data) {
  met_data[, metabolite := gsub("\\[.*?\\]", "", metabolite)]  # Remove text inside brackets
  met_data[, metabolite := trimws(metabolite)]  # Trim leading/trailing spaces
  met_data[, metabolite := tolower(metabolite)]  # Convert to lowercase
  return(met_data)
}

# Function to clean info_data
clean_info_data <- function(info_data) {
  # Keep only rows where fluid == "plasma"
  info_data <- info_data[fluid == "plasma"]

  # Ensure correct encoding and remove asterisks
  info_data$metabolite <- gsub("[*]", "", info_data$metabolite)  # Remove all asterisks
  info_data$metabolite <- gsub("\u200B", "", info_data$metabolite)  # Remove zero-width spaces (if present)
  info_data$metabolite <- gsub("[[:space:]]+$", "", info_data$metabolite)  # Trim trailing spaces
  info_data$metabolite <- trimws(info_data$metabolite)  # Remove any remaining spaces
  info_data$metabolite <- tolower(info_data$metabolite)  # Convert to lowercase
  return(info_data)
}

# Function to merge met_data with info_data
merge_met_data_info <- function(met_data, info_data) {
  met_data <- merge(met_data, 
                    info_data[, .(metabolite, platform_name, super_pathway, sub_pathway)], 
                    by = c("metabolite", "platform_name"), 
                    all.x = TRUE)  # Keep all rows in met_data
  return(met_data)
}

run_pipeline <- function(z_score_file, info_file) {
  # Load data
  data <- load_data(z_score_file, info_file)
  z_score_data <- data$z_score_data
  info_data <- data$info_data

  # Reshape the combined dataset into long format
  met_data <- reshape_long(z_score_data)

  # Clean the metabolite names
  met_data <- clean_metabolite_names(met_data)

  # Clean info data
  info_data <- clean_info_data(info_data)

  # Merge met_data with info_data
  met_data <- merge_met_data_info(met_data, info_data)

  fwrite(met_data, "data/processed/reshaped_met_data.csv")
  return(met_data)
}

main <- function() {
  met_data <- run_pipeline("data/processed/humet_imputed_400trees_z_score.csv", "input/humet_info.csv")
  # Save the processed data to a CSV file
  fwrite(met_data, "data/processed/reshaped_met_data.csv")
}

main()