#!/usr/bin/env bash
#SBATCH --job-name=only_addmalts # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=srikar.katta@duke.edu     # Where to send mail
#SBATCH --output=evaluate_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec


# # # make data for CATE estimation experiments ----
# srun -u python3 ./normal_sim_dgp_generate_data.py trunc_normal_linear_diff
# srun -u python3 ./normal_sim_dgp_generate_data.py trunc_normal_complex
# srun -u python3 ./normal_sim_dgp_generate_data.py trunc_normal_variance
# srun -u python3 ./dist_cov_sim_dgp_generate_data.py dist_cov_complex

# wait

# # # estimate CATEs using ADD MALTS ----

# srun -u python3 ./cate_addmalts_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_addmalts_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_addmalts_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_addmalts_experiments.py ./experiments/dist_cov_complex

# # estimate CATEs using baselines ----

# # linear propensity score matching

# srun -u python3 ./cate_lin_psmatch_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_lin_psmatch_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_lin_psmatch_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_lin_psmatch_experiments.py ./experiments/dist_cov_complex

# # random forest propensity score matching

# srun -u python3 ./cate_rf_psmatch_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_rf_psmatch_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_rf_psmatch_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_rf_psmatch_experiments.py ./experiments/dist_cov_complex

# # linear outcome model 


# srun -u python3 ./cate_lin_reg_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_lin_reg_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_lin_reg_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_lin_reg_experiments.py ./experiments/dist_cov_complex

# # doubly robust, linear outcome model with linear propensity score

# srun -u python3 ./cate_dr_linps_normal_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_dr_linps_normal_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_dr_linps_normal_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_dr_linps_normal_experiments.py ./experiments/dist_cov_complex


# # doubly robust, linear outcome model with random forest propensity score

# srun -u python3 ./cate_dr_rfps_normal_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_dr_rfps_normal_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_dr_rfps_normal_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_dr_rfps_normal_experiments.py ./experiments/dist_cov_complex

# # single wasserstein tree_cate

# srun -u python3 ./cate_ftree_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_ftree_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_ftree_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_ftree_experiments.py ./experiments/dist_cov_complex

# # wasserstein forest


# srun -u python3 ./cate_frf_experiments.py ./experiments/trunc_normal_linear_diff
# srun -u python3 ./cate_frf_experiments.py ./experiments/trunc_normal_complex
# srun -u python3 ./cate_frf_experiments.py ./experiments/trunc_normal_variance
# srun -u python3 ./cate_frf_experiments.py ./experiments/dist_cov_complex


# wait

srun -u python3 ./cate_addmalts_change_k_experiments.py ./experiments/trunc_normal_linear_diff
srun -u python3 ./cate_addmalts_change_k_experiments.py ./experiments/trunc_normal_complex
srun -u python3 ./cate_addmalts_change_k_experiments.py ./experiments/trunc_normal_variance
srun -u python3 ./cate_addmalts_change_k_experiments.py ./experiments/dist_cov_complex


wait

# # get relative error for each experiment ----

srun -u python3 ./cate_evaluation.py ./experiments/trunc_normal_linear_diff mire
srun -u python3 ./cate_evaluation.py ./experiments/trunc_normal_complex mire
srun -u python3 ./cate_evaluation.py ./experiments/trunc_normal_variance mire
srun -u python3 ./cate_evaluation.py ./experiments/dist_cov_complex mire
srun -u python3 ./cate_evaluation.py ./experiments_old/friedman_sim_dgp mise

# # evaluate overlap ----
srun -u python3 ./positivity_experiments.py

# wait

# # # make all figures for the main paper ----


# # additional experiments on another DGP