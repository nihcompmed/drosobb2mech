# drosobb2mech
Codes and data for the manuscript "Hybrid deep learning–mechanistic modeling of cellular dynamics from a spatiotemporal single-cell atlas"


We name the files as Step*_****.py or ipynb. In both Method Section and Result Section, we need around 8 steps to go from original data to final Hill mechanistic model:

Step 1: organize the data(rearrange, pre-imputation, and save the data)

Step 2: train the VAE

Step 3: imputation

Step 4: evaluation of the trained VAE and imputed dataset(not needed for some sections)

Step 5: train the neural ODE

Step 6: integrate the trained black-box dynamics to save the trajectory to generate teacher derivative values

Step 7: fit the Hill mechanistic model

Step 8: evaluate based on what we obtained.

