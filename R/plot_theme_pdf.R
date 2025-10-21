pacman::p_load(
  tidyverse,
  here
)

# 1) Palette default per scale discrete
axes_palette <- c("#8c1c13ff", "#86ba90ff", "#54403bff")
state_cols   <- c("#8c1c13ff", "#df9457ff", "#86ba90ff", "#54403bff")

pdf_theme <- envalysis::theme_publish()

theme_set(pdf_theme)

options(
  ggplot2.discrete.colour = axes_palette,
  ggplot2.discrete.fill   = axes_palette
)
