library(ggplot2)
library(here)
library(showtext)

# 1) Palette default per scale discrete
axes_palette <- c("#8c1c13ff", "#86ba90ff", "#54403bff")
state_cols   <- c("#8c1c13ff", "#df9457ff", "#86ba90ff", "#54403bff")

options(
  ggplot2.discrete.colour = axes_palette,
  ggplot2.discrete.fill   = axes_palette
)

# 2) Registra Figtree (locale) e attiva showtext
font_path <- here::here("python", "Figtree-Regular.ttf")
showtext::font_add(family = "Figtree", regular = font_path)
showtext::showtext_auto(enable = TRUE)
# opzionale: migliora resa per export raster
showtext::showtext_opts(dpi = 300)

# 3) Imposta tema globale con lo stesso background
theme_set(
  theme_minimal(base_family = "Figtree") +
    theme(
      plot.background  = element_rect(fill = "#F4ECE2", colour = NA),
      panel.background = element_rect(fill = "#F4ECE2", colour = NA),
      text = element_text(family = "Figtree")
    )
)

# 4) Assicura che geom_text/label usino Figtree di default
update_geom_defaults("text",  list(family = "Figtree"))
update_geom_defaults("label", list(family = "Figtree"))