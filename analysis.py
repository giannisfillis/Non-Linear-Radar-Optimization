import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums, pearsonr, spearmanr, kendalltau
import warnings
import re
import io

# apenergopoise ta warnings 
warnings.filterwarnings("ignore")


files = {
    'NewtonDG': 'Newton.txt',
    'BFGS-W': 'BFGS.txt',
    'NM': 'NelderMead.txt',
    'GA': 'GA.txt',
    'PSO': 'PSO.txt'
}

dataframes = {}
print("--- Starting Analysis ---")

for algo_name, file_name in files.items():
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        df = pd.read_csv(io.StringIO(content), sep=r'\s+')
        df = df[pd.to_numeric(df['Exper'], errors='coerce').notnull()]
        
        # metatropi se arithimitika
        for col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
             
        dataframes[algo_name] = df
        print(f"loaded: {algo_name}")
        
    except Exception as e:
        print(f"Error in file {file_name}: {e}")

# enose gia ta graphs
all_data = []
for algo_name, df in dataframes.items():
    temp_df = df.copy()
    temp_df['Algorithm'] = algo_name
    all_data.append(temp_df)

full_df = pd.concat(all_data) if all_data else pd.DataFrame()

if not full_df.empty:
    
    # stats 
    stats_list = []
    for algo_name, df in dataframes.items():
        for metric in ['MSEtrain', 'MSEtest', 'LastHit']:
            stats_list.append({
                'Algorithm': algo_name,
                'Metric': metric,
                'Mean': df[metric].mean(),
                'Median': df[metric].median(),
                'Std.Dev': df[metric].std(),
                'Min': df[metric].min(),
                'Max': df[metric].max()
            })

    pd.DataFrame(stats_list).to_csv('descriptive_statistics.csv', index=False)
    print("Στατιστικά: descriptive_statistics.csv")

    # graphs 
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(x='Algorithm', y='MSEtrain', data=full_df, ax=axes[0], showfliers=False)
    axes[0].set_title('MSE Train')
    axes[0].set_yscale('log')

    sns.boxplot(x='Algorithm', y='MSEtest', data=full_df, ax=axes[1], showfliers=False)
    axes[1].set_title('MSE Test (Generalization)')
    axes[1].set_yscale('log')

    sns.boxplot(x='Algorithm', y='LastHit', data=full_df, ax=axes[2])
    axes[2].set_title('Last Hit (Cost)')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=300)
    print("Γραφήματα: boxplots.png")

    # wilcoxon
    algo_names = list(dataframes.keys())
    n_algos = len(algo_names)
    comparison_matrix = pd.DataFrame(index=algo_names, columns=algo_names)
    
    results_summary = {name: {'Wins': 0, 'Losses': 0, 'Ties': 0} for name in algo_names}
    alpha = 0.05

    for i in range(n_algos):
        for j in range(n_algos):
            name_A = algo_names[i]
            name_B = algo_names[j]
            
            if i == j:
                comparison_matrix.iloc[i, j] = "="
                results_summary[name_A]['Ties'] += 1
                continue
            
            df_A = dataframes[name_A]
            df_B = dataframes[name_B]
            
            # check mse test
            try:
                stat, p_val_mse = ranksums(df_A['MSEtest'], df_B['MSEtest'])
            except: p_val_mse = 1.0

            result = ""
            if p_val_mse < alpha:
                # nikitis aytos me to mikrotero median 
                if df_A['MSEtest'].median() < df_B['MSEtest'].median():
                    result = "+"
                else:
                    result = "-"
            else:
                # isopalia mse check lasthit
                try:
                    stat_hit, p_val_hit = ranksums(df_A['LastHit'], df_B['LastHit'])
                except: p_val_hit = 1.0
                
                if p_val_hit < alpha:
                    if df_A['LastHit'].median() < df_B['LastHit'].median():
                        result = "+"
                    else:
                        result = "-"
                else:
                    result = "=" 
            
            comparison_matrix.iloc[i, j] = result
            
            if result == "+": results_summary[name_A]['Wins'] += 1
            elif result == "-": results_summary[name_A]['Losses'] += 1
            elif result == "=": results_summary[name_A]['Ties'] += 1

    final_matrix = comparison_matrix.copy()
    total_col = []
    for name in algo_names:
        res = results_summary[name]
        total_col.append(f"{res['Wins']}/{res['Losses']}/{res['Ties']}")
    final_matrix['Total (W/L/T)'] = total_col
    
    final_matrix.to_csv('comparison_matrix.csv')
    print("Comp matrix: comparison_matrix.csv")

    # sysxetiseis
    corr_list = []
    for algo_name, df in dataframes.items():
        if len(df) > 1:
            p_corr, _ = pearsonr(df['MSEtrain'], df['MSEtest'])
            s_corr, _ = spearmanr(df['MSEtrain'], df['MSEtest'])
            k_corr, _ = kendalltau(df['MSEtrain'], df['MSEtest'])
            
            corr_list.append({
                'Algorithm': algo_name,
                'Pearson': round(p_corr, 4),
                'Spearman': round(s_corr, 4),
                'Kendall': round(k_corr, 4)
            })
    
    pd.DataFrame(corr_list).to_csv('correlations.csv', index=False)
    print("Corr file: correlations.csv")
    print("\nCompleted!.")

else:
    print("No Data.")