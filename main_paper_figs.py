import pandas as pd
from cate_evaluation import mise_cate
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def cate_plot():
    metric = 'mise'

    # mise_df = pd.read_csv('./experiments/dist_cov_friedmans_beta/mire.csv')

    metric_df_list = []
    for folder in os.listdir('./experiments'):
        if 'trunc_normal' or 'dist_cov' in folder:
            metric_df = pd.read_csv('./experiments/' + folder + '/' + metric + '.csv').assign(sim = folder)
            metric_df_list.append(metric_df)
    metric_df = pd.concat(metric_df_list)

    include_methods = [
        'Lin PSM',
        'RF PSM',
        'FT',
        'FRF',
        'LR', 
        'LR + Lin PS',
        'LR + RF PS',
        'ADD MALTS',
        # 'ADD MALTS (k = 5)',
        # 'ADD MALTS (k = 2)'
        ]

    color_palette = {  
            'LR' : "#1F77B4FF",
            'FT' : "#FF7F0EFF",
            'FRF' : "#2CA02CFF",
            'LR + Lin PS' : "#D62728FF",
            'LR + RF PS' : "#9467BDFF",
            'Lin PSM' : "#E377C2FF",
            'RF PSM' : '#BCBD22FF',
            # 'FRF + RF PS' : "#fdcce5",
            'ADD MALTS' : "#17BECFFF",
            'ADD MALTS (k = 5)' : "#1F74C0AB",
            'ADD MALTS (k = 2)' : "#FF77C0AB"
        }

    metric_df = metric_df[include_methods + ['df', 'sim']].melt(id_vars = ['df', 'sim'], var_name='Method')
    labels = []
    for i in metric_df.sim.values:
        if i == 'trunc_normal_variance3':
            labels.append('Variance')
        elif i == 'trunc_normal_linear_diff3':
            labels.append('Linear')
        elif i == 'trunc_normal_complex3':
            labels.append('Complex')
        elif i == 'dist_cov_complex':
            labels.append('Dist Cov')
        else:
            labels.append('')
    metric_df['label'] = labels

    sns.set_theme(style = 'whitegrid')
    sns.set(rc = {'figure.figsize':(15,5)})
    sns.set(font_scale = 1.5)

    g = sns.boxplot(metric_df.loc[metric_df['sim'].isin(['trunc_normal_variance3', 'trunc_normal_linear_diff3', 'trunc_normal_complex3', 'dist_cov_complex'])],
                    hue = 'Method',
                    y = 'value',
                    x = 'label', 
                    order = ['Variance', 'Linear', 'Complex', 'Dist Cov'],
                    palette =[color_palette[i] for i in include_methods],
                    showfliers = False)

    if metric == 'mire':
        plt.ylabel('Integrated Relative Error (%)')
    elif metric == 'bias':
        plt.ylabel('Integrated Bias')
    elif metric == 'mise':
        plt.ylabel('Integrated Squared Error')

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    plt.xlabel(None)

    g.legend(ncol = 4, title = 'Method', fontsize = 20)

    plt.tight_layout()

    plt.savefig(f'./figures/cate_experiment.png', dpi = 300, transparent = True, bbox_inches = 'tight')

def positivity_plot():
    positivity_df = pd.read_csv('./experiments/positivity.csv')
    prune_lasso = positivity_df.query(f'df == 0').prune_lasso.values
    prune_rf = positivity_df.query(f'df == 0').prune_rf.values
    prune_addmalts = positivity_df.query(f'df == 0').prune_addmalts.values
    X_est = positivity_df.query(f'df == 0')[['X0', 'X1']].to_numpy()
    xv, yv = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), sparse = True)

    def make_pi(xv, yv):
        return 1/(1 + np.exp(-(0.5 * xv + 0.5 * yv))) * ((xv >= -0.5) + (yv >= -0.5))# + (yv <= -0.5) + (yv >= 0.5))
    pi_grid = 1/(1 + np.exp(-(0.5 * xv + 0.5 * yv)))
    pi_grid_violation = make_pi(xv, yv) 

    plt.rcParams.update({'font.size': 25})

    fig, ax = plt.subplots(1, 3, sharey = True, sharex = True, figsize = (13, 4))
    ax[0].contourf(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), pi_grid_violation)
    ax[0].scatter(x = X_est[:, 0], y = X_est[:, 1], color = ['red' if i else 'lightblue' for i in prune_lasso]) #color = ['lightblue' if i else 'red' for i in (prune_est == prune_lasso)])
    ax[0].title.set_text('Linear PS')

    ax[1].contourf(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), pi_grid_violation)
    ax[1].scatter(x = X_est[:, 0], y = X_est[:, 1], color = ['red' if i else 'lightblue' for i in prune_rf]) #, color = ['lightblue' if i else 'red' for i in (prune_est == prune_rf)])
    ax[1].title.set_text('RF PS')

    cont = ax[2].contourf(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), pi_grid_violation)
    ax[2].scatter(x = X_est[:, 0], y = X_est[:, 1], color = ['red' if i else 'lightblue' for i in prune_addmalts]) #, color = ['lightblue' if i else 'red' for i in (prune_est == prune)])
    ax[2].title.set_text('ADD MALTS')



    axins = inset_axes(ax[2],
            width="5%", # width = 10% of parent_bbox width
            height="100%", # height : 50%
            loc=6,
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax[2].transAxes,
            borderpad=0,
        )

    plt.colorbar(cont, cax = axins)

    plt.savefig('./figures/positivity_experiment.png', dpi = 300, transparent = True)

def main():
    cate_plot()
    positivity_plot()

    print('done plotting synthetic experiment, main figures')