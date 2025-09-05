install.packages("renv")

renv::init()

install.packages("pacman")

pacman::p_load(
  tidyverse,
  readxl,
  here,
  janitor,
  gt,
  gtsummary,
  patchwork,
  statmod,
  lifecontingencies,
  tweedie,
  doParallel,
  foreach,
  tidymodels,
  ggstatsplot
)

renv::snapshot()