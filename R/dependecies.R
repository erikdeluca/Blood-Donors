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
  "showtext",
  "broom",
  "hexbin",
  "broom.helpers",
  "poissonreg",
  "dotwhisker",
  "glmnet"
)

renv::install(cran_pkgs)

renv::snapshot(prompt = FALSE)