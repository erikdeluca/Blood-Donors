install.packages("renv")

if (file.exists("renv.lock")) {
  renv::restore()
} else {
 renv::init()
}

cran_pkgs <- c(
  "tidyverse",
  "readxl",
  "here",
  "janitor",
  "gt",
  "gtsummary",
  "patchwork",
  "statmod",
  "lifecontingencies",
  "tweedie",
  "doParallel",
  "foreach",
  "tidymodels",
  "ggstatsplot",
  "broom",
  "hexbin",
  "broom.helpers",
  "poissonreg",
  "dotwhisker",
  "glmnet",
  "doParallel",
  "parallel",
  "tune"
)

renv::install(cran_pkgs)

renv::snapshot(prompt = FALSE)