# Rebuild the full dataset from the four source partitions stored in /data.

parts <- list.files(
  "data",
  pattern = "^diabetes_part[1-4]\\.csv$",
  full.names = TRUE
)

if (length(parts) != 4) {
  stop("Expected four diabetes_part*.csv files in the data directory.")
}

parts <- sort(parts)
diabetes <- do.call(rbind, lapply(parts, read.csv))

write.csv(diabetes, "data/diabetes.csv", row.names = FALSE)
cat("Created data/diabetes.csv with", nrow(diabetes), "rows.\n")
