# OVITO visualisation

## MaxVol Gamma (nbh grades)

The .xyz file resulting from *crystal_diffusion/analysis/ovito_visualisation.py* can be loaded directly in OVITO.

See *examples/local/mtp_to_ovito.sh* for a bash script example.

Once loaded, all atoms appear white. You can change the radius by clicking on **Particles** on the right hand side.
The menu **Particle display** can customized the view, in particular, the **Standard radius**. It should defaults at 0.5

To get the gamma value, click on **Particles**. On the right hand side near the top, click on the scrolldown menu
**Add modification** -> **Color coding**. A reasonable start value (around 1) and end value (around 10) should be set.
**Adjust range (all frames)** will match the colorbar to be the same across the frames. Low gamme value should look
blue, and high values are red.