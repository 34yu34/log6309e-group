import json
import numpy as np
import scikit_posthocs as sp
from itertools import combinations
from scipy import stats


if __name__ == "__main__":

    results_file = 'results.json'
    with open(results_file, 'r') as json_file:
        data = json.load(json_file)

    results = data['random']
    models = []
    scores = []
    for key, value in results.items():
        models.append(key)
        scores.append(value['f1'])
        
    group_means = [np.mean(group) for group in scores]
    group_dict = {
    models[0]: group_means[0],
    models[1]: group_means[1],
    models[2]: group_means[2],
    models[3]: group_means[3]
    }   
    
    # Combine all data into a single list
    all_data = [value for group_data in scores for value in group_data]
    # Sort the data by values
    sorted_data = sorted(zip(all_data, [models[i] for i in range(4)]))
    
    # Calculate the sum of squared ranks for each group
    group_ranks = {}
    current_group = None
    rank_sum = 0
    for rank, (value, group) in enumerate(sorted_data, start=1):
        if group != current_group:
            if current_group is not None:
                group_ranks[current_group] = rank_sum
            rank_sum = 0
            current_group = group
        rank_sum += rank
    
    # Add the last group's rank
    if current_group is not None:
        group_ranks[current_group] = rank_sum
    
    # Calculate the Scott-Knott rank statistic
    n = len(all_data)
    N = len(data)
    Q = 0
    for group, rank_sum in group_ranks.items():
        Q += (rank_sum - (n+1)/2)**2

    # Calculate the critical value for Scott-Knott test
    critical_value = stats.chi2.ppf(0.95, df=N-1)
    
    # Perform the Scott-Knott test
    if Q > critical_value:
        print("There is a significant difference between groups.")
    else:
        print("There is no significant difference between groups.")
    
    
    # Sort the groups by their Scott-Knott rank statistic (higher is better)
    sorted_groups = sorted(group_ranks.items(), key=lambda x: x[1], reverse=True)

    # Rank the groups based on their Scott-Knott rank statistic
    ranked_groups = {group[0]: rank + 1 for rank, group in enumerate(sorted_groups)}

    # Print the ranking
    for group, rank in ranked_groups.items():
        print(f"Rank {rank}: {group}")
        
    from scipy.stats import kruskal

    # Perform Kruskal-Wallis test
    kruskal_statistic, p_value = kruskal(*scores)

    # Set your significance level (alpha)
    alpha = 0.05

    # Compare the p-value to the significance level
    if p_value < alpha:
        print("There is a statistically significant difference between groups.")
    else:
        print("There is no statistically significant difference between groups.")
        
    from scipy.stats import f_oneway

    
    # Perform one-way ANOVA
    # breakpoint()
    f_statistic, p_value = f_oneway(*scores)

    # Set your significance level (alpha)
    alpha = 0.05

    # Compare the p-value to the significance level
    if p_value < alpha:
        print("There is a statistically significant difference between groups.")
    else:
        print("There is no statistically significant difference between groups.")
    
    

