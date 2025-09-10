pacman::p_load(
  tidyverse,  # A set of many useful libraries
  readxl,     # To import the dataset from Excel
  here,       # To avoid problems with file directories
  janitor,    # To clean data in a fast way
  broom,           
  broom.helpers,
  gt,         # Output tables
  gtsummary,  # Output tables for models and survival data
  patchwork, # merge more plots
  statmod  # tweedie models
)

data <- read_csv(
  here("data", "FINAL", "dataframe_cleaned.csv")
  )

data |> 
  distinct() |> 
  mutate(
    class_year = cut(birth_year, 
                     breaks = seq(1900, 2010, by = 10), 
                     dig.lab = 4,
                     include.lowest = TRUE
                     ),
    class_age = cut(age, 
                     breaks = c(seq(0, 70, by = 10), max(age)), 
                     dig.lab = 3,
                     include.lowest = TRUE,
                     ordered_result = T
                     ),
    .before = birth_year
  ) -> data