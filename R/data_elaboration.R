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
  filter(
    donation_type == "SANGUE",
    age <= 70
  ) |>
  distinct() |>
  mutate(
    class_year = cut(birth_year,
                     breaks = seq(1940, 2010, by = 10),
                     dig.lab = 4,
                     include.lowest = TRUE
                     ) |> factor(),
    # cut again to avoid peopole death in 2009
    class_age = cut(age,
                     breaks = c(min(age), seq(25, 65, by = 10), max(age)),
                     dig.lab = 3,
                     include.lowest = TRUE,
                     ordered_result = T
                     ) |> factor(),
    .before = birth_year
  ) -> data

# data model summerized for total_donations
data |>
  filter(
    donation_type == "SANGUE",
    age <= 70
  ) |>
  arrange(unique_number, age) |>
  mutate(
    total_donations = cumsum(number_of_donations),
    .by = unique_number,
    .after = number_of_donations
  ) |>
  slice_max(total_donations, by = unique_number) |>
  filter(total_donations < 100) |>
  mutate(
    gender = factor(gender),
    class_year = cut(birth_year,
                     breaks = seq(1940, 2010, by = 10),
                     dig.lab = 4,
                     include.lowest = TRUE
                     ) |> factor(),
    # cut again to avoid peopole death in 2009
    class_age = cut(age,
                     breaks = c(min(age), seq(25, 65, by = 10), max(age)),
                     dig.lab = 3,
                     include.lowest = TRUE,
                     ordered_result = T
                     ) |> factor(),
  ) -> data_total

# data_model per panel models
data |>
  filter(
    donation_type == "SANGUE",
    age <= 70
  ) |>
  mutate(
    gender = factor(gender),
    # cut again to avoid peopole death in 2009
    class_age = cut(age,
                     breaks = c(min(age), seq(25, 65, by = 10), max(age)),
                     dig.lab = 3,
                     include.lowest = TRUE,
                     ordered_result = T
                     ) |> factor(),
    class_year = cut(birth_year,
                     breaks = seq(1900, 2010, by = 10),
                     dig.lab = 4,
                     include.lowest = TRUE
                     ),
  ) -> data_panel


# check outliers
# data |>
#   filter(
#     donation_type == "SANGUE",
#     number_of_donations < 5,
#     age > 70
#   ) |>
#   tbl_summary(
#     by = donor_class
#   )


# data_total |>
#   filter(
# birth_year > 2000
# ) |>
#   slice_max(total_donations) |>
#   select(birth_year, total_donations)

# data_total |>
#   summarise(
#     across(total_donations, mean),
#     .by = class_year
#   )


# per i modelli var
data |>
  filter(donation_type == 'SANGUE', donor_class == 'P') |>
  pivot_wider(
    names_from = year,
    names_prefix = "y_",
    values_from = number_of_donations,
    id_cols = unique_number,
    values_fill = 0
  ) |>
  # take who has donated in the last two year
  filter(if_any(c(y_2022, y_2021), \(x) x > 0)) -> donations

data |>
  reframe(
    class_year,
    birth_year,
    first_donation_year,
    gender,
    .by = unique_number
  ) |>
  distinct() -> sociodemographic

right_join(
  sociodemographic,
  donations,
  by = "unique_number"
) -> recent_donations
