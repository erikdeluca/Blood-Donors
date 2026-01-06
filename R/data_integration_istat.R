residenti <- read_csv(here("data", "residenti_Trieste_e_Gorizia.csv"))

residenti |>
  filter(
    Sesso != "totale"
  ) |>
  rename(
    age = Età,
    gender = Sesso,
    year = TIME,
    population = Value,
  ) |>
  summarise(
    across(population, sum),
    .by = c(age, gender, year)
  ) |>
  mutate(
    age = str_extract(age, "[:digit:]*") |> as.numeric(),
    gender = case_when(
      gender == "maschi" ~ "M",
      gender == "femmine" ~ "F",
      T ~ NA
    )
  ) -> residenti

lifecontingencies::demoIta |>
  transmute(
    age = X,
    maschi = SIM02,
    femmine = SIF02,
    across(c(maschi, femmine), \(x) x / lag(x, default = 1e5), .names = "{col}_px")
  ) |>
  pivot_longer(ends_with("px"), names_to = "gender", values_to = "px") |>
  # select(-maschi, - femmine) |>
  mutate(
    gender = case_when(
      gender == "maschi_px" ~ "M",
      gender == "femmine_px" ~ "F",
      T ~ NA
    )
  ) -> life_table

residenti |>
  add_row(year = 2009:2018, .before = 1) |>
  complete(age, gender, year) |>
  filter(!if_any(c(age, gender), is.na)) |>
  left_join(life_table, by = c("age", "gender")) |>
  arrange(-year, age) |>
  filter(gender == "M") |>
  pivot_wider(names_from = year, values_from = population, names_prefix = "y_") |>
  mutate(
    y_2018 = lead(y_2019, default = 0) * maschi / lead(maschi, default = 0),
    y_2017 = lead(y_2019, n = 2, default = 0) * maschi / lead(maschi, default = 0),
    y_2016 = lead(y_2019, n = 3, default = 0) * maschi / lead(maschi, default = 0),
  )

filled_residenti <-
  residenti |>
  add_row(year = 2009:2018, .before = 1) |>
  complete(age, gender, year) |>
  filter(!if_any(c(age, gender), is.na)) |>
  left_join(life_table, by = c("age", "gender")) |>
  arrange(-year, age) |>
  pivot_wider(names_from = gender, values_from = c(px, population)) |>
  pivot_wider(names_from = year, values_from = c(population_M, population_F))

years <- 2018:2009
gender <- "M"

filled_residenti <- reduce(years, function(df, year) {
  col <- paste0("population_", c("F", "M"), "_", year)
  col_19 <- paste0("population_", c("F", "M"), "_2019")
  df |>
    mutate(
      !!sym(col[1]) := lead(!!sym(col_19[1]), 2019 - year, 0) * femmine / lead(femmine, 2019 - year, 0),
      !!sym(col[2]) := lead(!!sym(col_19[2]), 2019 - year, 0) * maschi / lead(maschi, 2019 - year, 0)
      )
}, .init = filled_residenti) |>
  mutate(across(starts_with("population"), round))

filled_residenti |>
  select(age, starts_with("population")) |>
  pivot_longer(starts_with("population"), values_to = "population") |>
  separate(name, sep = "_", into = c("pop", "gender", "year")) |>
  select(-pop) |>
  mutate(across(year, as.numeric)) -> residenti
