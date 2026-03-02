# Foundation Model for Biological Temporal Data Dynamics with Experimental Validation
Codes and data for the manuscript "Foundation Model for Biological Temporal Data Dynamics with Experimental Validation"

Part 1: the Drosophila dataset and its codes.

We name the files as Step*_****.py or ipynb. In both Method Section and Result Section, we need around 8 steps to go from original data to final Hill mechanistic model:

Step 1: organize the data(rearrange, pre-imputation, and save the data)

Step 2: train the VAE

Step 3: imputation

Step 4: evaluation of the trained VAE and imputed dataset(not needed for some sections)

Step 5: train the neural ODE

Step 6: integrate the trained black-box dynamics to save the trajectory to generate teacher derivative values

Step 7: fit the Hill mechanistic model

Step 8: evaluate based on what we obtained.


Some codes need large csv files that cannot be uploaded. Can contact the author for the details if any question.



Now for each step we show what to do:

Step 1: Open the Step1_organizedata.m with Matlab and run it, it will unzip data and generate needed datasets. (All from the same data, but with different pre-imputation using rand-fill(called fill_rand) and avg-fill and zero-fill. Their names are in the form of 'all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv'
where ctgxyz means they have col 1 as cellid, col2 as time, col3to101 as g, col102to104 as xyz spatial of positional bins. first fillrand means pre-imputation; second fillzero means some data slots from original dataset is missing, so we fill by average no matter what pre-imputation we use; 90 percent means we exclude the 10% fixed testing set.

Step 2: Run the swarm file 'Step2_VAEpenaltyJacdir_withmask_250916.swarm' to train VAEs.

Step 3: Run the swarm file 'step3_impute_250916.swarm' to do the imputation with respect to each of trained VAEs.

Step 4: We can then make the Figure 3 of the manuscript, and do other evaluations.

Step 5: run 'Step5_neuralODE250916.swarm' to train the neural ODE corresponding to trained VAE and its corresponding imputation.

Step 6: run 'Step6_integrateBB.py' to get teacher derivatives, this is only needed if using integral form of neural ODE (that is, decoder of trajectory of neural ODE in the latent space). If using pushforward Jac_dec(mu)f(mu), then can skip this step.

Step 7: fit the Hill function. Running 'Step7_fitHill99D_VAEonly_autorun.py'

Step 8: evaluate after fitting Hill function. This is done in each of Result section.





******** How to run codes in Sec_impact_of_teacher:
The python files have guidance of what to run inside each file, so we do not show every command needed to be run here.

Step 0: make sure the data file all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv is in current folder.

Step 0: fit the Hill model not from deep learning but directly from data. Run 'Step0_fitHill_Sec_impact_of_teacher.py' to fit a Hill model.

Step 2: train VAE for this 27D, since all 27 genes are fully observed, no need to pre-impute or do imputation, so no Step3 or 4. Also since we skip training neural ODE for this section only, we can also skip Step 5 and 6 here for a simpler comparison.

To train the VAE, run Step2_trainVAE_Sec_impact_of_teacher.swarm, or run 'Step2_trainVAE_Sec_impact_of_teacher.py' directly with the hyperparameter setting as you like.

Step 7: run 'Step7_fitHill_Sec_impact_of_teacher.py' to fit the Hill model from trained VAE. Need to put the ckpt file of VAE in current folder.

Step 8: Use 'Step8_integrateHill_zerosmallVij_code2_window.py' to evaluate the Hill model with proper input(we input both the Hill model from teacher or without teacher). Before this, we can use the Step8p1 to Step8p5 files to calibrate for both models first. 



******** How to run codes in Sec_universality:

Step 1: prepare the data as before.

Step 2: run 'Step2_trainVAE_universality01.swarm' and 'Step2_trainVAE_universality02.swarm' for different VAEs (these VAE all have same structures, but with different pre-imputation method, different data points chosen from different places(a/m/p, or d/m/v, or random 33%), different latent dimension or activation function).

Step 3: use 'Step3_impute_universality.py' to do imputation after training of VAE for each VAE-training job you run.

Step 4: use 'Step4b_fitHill99D_VAEonly_forUniversalitySection' to fit the Hill function.

Step 5: evaluate the error from fit Hill function with 'Step5_evalHill_errors.py'





Part 2: Air Quality data and code:
For figure 2 of comparing prediction MSE between our trained black-box dynamics and VAR and VARX and SARIMAX, just go to Plot_Figure2.ipynb and run all cells. It involve fitting and forecasting of SARIMAX which could take up to 5 minutes waiting. Then at last it will replot the Figure 2 and show the comparison.

For other codes on air quality,
See 'AQ251201004_dropNMHC_nocontrol.ipynb'

**********  How to plot all figures in the main text:

Figure 3: 
just run in the plotfigures folder:
python PlotFigure3_adaptation.py \
   --indir files_needed_for_fig3plot \
   --epoch-pick 3

Figure 4:
Run this :
python eegmi251213021_universal_latentODE_finalfigs_NPJSBA_Fig4.py   --data-dir EEGMI_numpy   --subject S033   --run 1   --run-ctrl 1   --ckpt-vae networks_for_Fig4/vae_universal30_win9_hd512_ld8_beta3e-04_251210003eegmiVAE_universal.pth   --ckpt-ode networks_for_Fig4/latentODE_universal_ctrl7_h9_endw_gain2_251213017.pt   --out-dir figures_uploadgithub  --start 2000   --num-windows 90   --t-end 9.0   --rollout-stride 1   --max-rollouts 36

Figure 5 and S4 and S5:
just run the jupyter notebook file:
Plot_Figure5_S4_S5_newest.ipynb




