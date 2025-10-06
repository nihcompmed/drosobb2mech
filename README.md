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

Now for each step we show what to do:

Step 1: Open the Step1_organizedata.m with Matlab and run it, it will unzip data and generate needed datasets. (All from the same data, but with different pre-imputation using rand-fill(called fill_rand) and avg-fill and zero-fill. Their names are in the form of 'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv'
where ctgxyz means they have col 1 as cellid, col2 as time, col3to101 as g, col102to104 as xyz spatial of positional bins. first fillrand means pre-imputation; second fillzero means some data slots from original dataset is missing, so we fill by average no matter what pre-imputation we use; 90 percent means we exclude the 10% fixed testing set.

Step 2: Run the swarm file 'Step2_VAEpenaltyJacdir_withmask_250916.swarm' to train VAEs.

Step 3: Run the swarm file 'step3_impute_250916.swarm' to do the imputation with respect to each of trained VAEs. We can then make the Figure 3 of the manuscript.



