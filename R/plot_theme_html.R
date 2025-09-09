library(ggplot2)
library(here)
library(showtext)

# 1) Default color cycle (discrete colour/fill) and state palette
axes_palette <- c("#8c1c13ff", "#86ba90ff", "#54403bff")
state_cols   <- c("#8c1c13ff", "#df9457ff", "#86ba90ff", "#54403bff")

options(
  ggplot2.discrete.colour = axes_palette,
  ggplot2.discrete.fill   = axes_palette
)

# 2) Register and use the Figtree font
font_add("Figtree", regular = here::here("python", "Figtree-Regular.ttf"))
showtext_auto()

# 3) Set global theme with background color analogous to figure.facecolor
theme_set(
  theme_minimal(base_family = "Figtree") +
    theme(
      plot.background  = element_rect(fill = "#F4ECE2", colour = NA),
      panel.background = element_rect(fill = "#F4ECE2", colour = NA)
    )
)