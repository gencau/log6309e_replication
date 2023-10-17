import numpy as np
from scipy.stats import ttest_ind

def scott_knott(data, alpha=0.05):
    """
    Scott-Knott test implementation.
    
    Parameters:
    - data: Dictionary where keys are group names and values are lists of observations.
    - alpha: Significance level for the t-test.
    
    Returns:
    - Clusters of statistically distinct groups.
    """
    # Sort groups by their means
    means = {group: np.mean(values) for group, values in data.items()}
    sorted_groups = sorted(data.keys(), key=lambda x: means[x])
    
    # Initial cluster
    clusters = [sorted_groups]
    
    while True:
        new_clusters = []
        split_occurred = False
        
        for cluster in clusters:
            if len(cluster) <= 1:
                new_clusters.append(cluster)
                continue
            
            max_t_stat = float('-inf')
            split_index = None
            
            # Try splitting the cluster at every possible position
            for i in range(1, len(cluster)):
                left_group = [item for subgroup in cluster[:i] for item in data[subgroup]]
                right_group = [item for subgroup in cluster[i:] for item in data[subgroup]]
                
                t_stat, p_val = ttest_ind(left_group, right_group)
                
                if p_val < alpha and t_stat > max_t_stat:
                    max_t_stat = t_stat
                    split_index = i
            
            # If a valid split is found, split the cluster
            if split_index is not None:
                new_clusters.append(cluster[:split_index])
                new_clusters.append(cluster[split_index:])
                split_occurred = True
            else:
                new_clusters.append(cluster)
        
        # If no splits occurred in this iteration, break
        if not split_occurred:
            break
        
        clusters = new_clusters
    
    return clusters

# Sample data
data = {
    'Group1': [10, 11, 10.5, 10.7, 10.8],
    'Group2': [20, 21, 20.5, 20.7, 20.8],
    'Group3': [10.2, 10.3, 10.1, 10.4, 10.6],
    'Group4': [30, 31, 30.5, 30.7, 30.8]
}

# Run Scott-Knott test
clusters = scott_knott(data)
print(clusters)
