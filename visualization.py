from pymol import cmd

cmd.hide("everything", "all")


# TAT protein
cmd.select("TAT", "polymer.protein")
cmd.show("sphere", "polymer.protein")
cmd.color("deepsalmon", "polymer.protein")


# Hide ions
cmd.hide("everything", "resn SOD+POT+CLA")

# Water
cmd.select("WATER", "solvent")
cmd.show("lines", "WATER")
cmd.color("lightblue", "WATER")


# Outer leaflet (LPSA)
cmd.select(
    "outer_leaflet",
    "resn ECLIP+AKDO+AHEP+AGLC+AGAL"
)

cmd.show("spheres", "outer_leaflet")
cmd.color("cyan", "outer_leaflet")

# Inner leaflet
cmd.select(
    "inner_leaflet",
    "resn DMPG+DMPE+TMCL1"
)

cmd.show("spheres", "inner_leaflet")
cmd.color("limon", "inner_leaflet")


# Sphere quality
cmd.set("sphere_quality", 2)

# Sphere transparency

cmd.set("sphere_transparency", 0.30, "outer_leaflet")
cmd.set("sphere_transparency", 0.30, "inner_leaflet")


# Simulation cell
cmd.show("cell", "all")
cmd.set_color("cell_red", [1.0, 0.0, 0.0])
cmd.set("cell_color", "cell_red")
cmd.bg_color("white")


# Camera orientation
cmd.orient("all")
cmd.zoom("all", 5)


cmd.turn("z", 0)

cmd.smooth
cmd.mplay()

