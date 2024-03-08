import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from functools import partial
import sys

def mise_cate(dataset_directory, dataset_iteration, file_name, metric = 'mire'):
    '''
    dataset_iteration : goes into seed
    '''
    
    folder = dataset_directory + '/dataset_' + str(2020 + 1000 * dataset_iteration)
    
    CATE_true_df = pd.read_csv(folder + '/ITE_est.csv').to_numpy()
    # try:
    CATE_est_df = pd.read_csv(folder + file_name, index_col='Unnamed: 0').to_numpy()
    
    # mise = ((np.abs(CATE_true_df - wass_tree_CATE_df)**2)).sum(axis = 1)/()
    # mire = ((np.abs(CATE_true_df - CATE_est_df)**2)).mean(axis = 1)/((CATE_true_df**2).sum(axis = 1))
    
    if metric == 'mire':
        mire = (np.abs((CATE_true_df - CATE_est_df)/CATE_est_df)).mean(axis = 1) * 100
        return mire
    elif metric == 'bias': 
        bias = ((CATE_true_df - CATE_est_df)).mean(axis = 1)
        return bias
    elif metric == 'mise':
        mise = ((CATE_true_df - CATE_est_df)**2).mean(axis = 1)
        return mise
    # except:
    #     return np.repeat(a = np.nan, repeats = n_est_units) 

def parallel_plot(dataset_iteration, dataset_directory, metric = 'mire'):
    print(dataset_directory, dataset_iteration)
    mise_list = []
    plot_df = pd.DataFrame({'ADD MALTS' : mise_cate(dataset_directory, dataset_iteration, '/addmalts_CATE.csv', metric),
                            'df' : dataset_iteration})
    
    plot_df['Lin PSM'] = mise_cate(dataset_directory, dataset_iteration, '/lin_ps_CATE.csv', metric)
    plot_df['RF PSM'] = mise_cate(dataset_directory, dataset_iteration, '/rf_ps_CATE.csv', metric)
    plot_df['LR'] = mise_cate(dataset_directory, dataset_iteration, '/lin_reg_CATE.csv', metric)
    plot_df['LR + Lin PS'] = mise_cate(dataset_directory, dataset_iteration, '/dr_linps_CATE.csv', metric)
    plot_df['LR + RF PS'] = mise_cate(dataset_directory, dataset_iteration, '/dr_rfps_CATE.csv', metric)
    plot_df['FT'] = mise_cate(dataset_directory, dataset_iteration, '/wass_tree_CATE.csv', metric)
    try:
        plot_df['FRF'] = mise_cate(dataset_directory, dataset_iteration, '/wrf_CATE.csv', metric)
        plot_df['ADD MALTS (k = 5)'] = mise_cate(dataset_directory, dataset_iteration, '/addmalts_k_5_CATE.csv')
        plot_df['ADD MALTS (k = 2)'] = mise_cate(dataset_directory, dataset_iteration, '/addmalts_k_2_CATE.csv')
    except:
        pass
    plot_df.to_csv(dataset_directory + '/dataset_' + str(2020 + 1000 * dataset_iteration) + '/' + metric + '.csv', index = False)
    
    return plot_df


def plot_mise(dataset_directory, mise_df, metric):
    mise_df = mise_df.rename(columns = {'MALTSPro' : 'ADD MALTS'})
    # uncomment any methods you want included in the final plot
    include_methods = [
        'LR', 
        'Lin PSM',
        'RF PSM',
        'LR + Lin PS',
        'LR + RF PS',
        'FT',
        'FRF',
        'ADD MALTS',
        # 'ADD MALTS (k = 5)',
        # 'ADD MALTS (k = 2)'
        ]
    mise_df = mise_df[include_methods].melt()
    # colors: "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"
    color_palette = {  
        'LR' : "#1F77B4FF",
        'FT' : "#FF7F0EFF",
        'FRF' : "#2CA02CFF",
        'LR + Lin PS' : "#D62728FF",
        'LR + RF PS' : "#9467BDFF",
        'Lin PSM' : "#E377C2FF",
        'RF PSM' : '#BCBD22FF',
        'Ablation' : '#8BD3C7',
        'ADD MALTS' : "#17BECFFF",
        'ADD MALTS (k = 2)' : "#1F77B4FF",
        'ADD MALTS (k = 5)' : "#FF7F0EFF",
    }
    sns.set_theme(style = 'whitegrid')
    sns.set(rc = {'figure.figsize':(10,5)})
    sns.set(font_scale = 1.5)
    sns.boxplot(data = mise_df, y = 'variable', x = 'value', showfliers = False, palette=[color_palette[i] for i in include_methods])
    if metric == 'mire':
        plt.xlabel('Integrated Relative Error (%)')
    elif metric == 'bias':
        plt.xlabel('Integrated Bias')
    elif metric == 'mise':
        plt.xlabel('Integrated Squared Error')
    
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    plt.ylabel(None)
    # plt.xticks(rotation = 15)
    if 'ADD MALTS (k = 5)' in include_methods:
        plt.savefig(dataset_directory + '/addmalts_k_comparison' + metric + '.png', dpi = 300, transparent = True, bbox_inches = 'tight')
    else:
        plt.savefig(dataset_directory + '/cate_estimation_' + metric + '.png', dpi = 300, transparent = True, bbox_inches = 'tight')



def main(argv):
    dataset_directory = sys.argv[1]
    metric = sys.argv[2]
    try:
        plot_df = pd.read_csv(dataset_directory + '/' + metric + '.csv')
        plot_mise(dataset_directory, plot_df, metric)
    except:
        dataset_iterations_to_conduct = range(0, 100)
        with Pool(processes = 25) as pool:
            dfs = pool.map(partial(parallel_plot, dataset_directory = dataset_directory, metric = metric),
                dataset_iterations_to_conduct)
        plot_df = pd.concat(dfs, axis = 0)
        plot_df.to_csv(dataset_directory + '/' + metric + '.csv', index = False)
        plot_mise(dataset_directory, plot_df, metric)

if __name__ == '__main__':
    main(sys.argv)