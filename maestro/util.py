import random
from torch.utils.data import Subset
from collections import Counter



def is_ipython():
    try:
        get_ipython()
        return True
    except NameError:
        return False

if is_ipython():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def find_best_combination(sizes, counts, target):

    dp = [{} for _ in range(target + 1)]
    dp[0][tuple(0 for _ in sizes)] = True  # Base case

    for i in range(target):
        for comb in dp[i]:
            for idx, size in enumerate(sizes):
                count = comb[idx]
                if count < counts[idx]:
                    new_size = i + size
                    if new_size <= target:
                        new_comb = list(comb)
                        new_comb[idx] += 1
                        dp[new_size][tuple(new_comb)] = True

    # Find the combination with total size closest to target
    for diff in range(target + 1):
        for offset in [diff, -diff]:
            idx = target + offset
            if 0 <= idx <= target and dp[idx]:
                best_combination = random.choice(list(dp[idx].keys()))
                return best_combination

    # If no combination is found (should not happen)
    return None

def stratify_split(dataset, *, stratify_key=None, stratify_labels = None, p_train = 0.8, shuffle=True, group_by_key=None, group_by_labels=None):

    if stratify_key is not None and stratify_labels is not None:
        raise ValueError("Only one of stratify_key and stratify_labels should be provided.")

    if stratify_key is None and stratify_labels is None:
        raise ValueError("One of stratify_key and stratify_labels should be provided.")

    if group_by_key is not None and group_by_labels is not None:
        raise ValueError("Only one of group_by_key and group_by_labels should be provided.")        

    if isinstance(stratify_key, str):
        key_copy = stratify_key
        stratify_key = lambda x: x[key_copy]

    if isinstance(group_by_key, str):
        key_copy = group_by_key
        group_by_key = lambda x: x[key_copy]


    if stratify_labels is None:
        stratify_labels = [stratify_key(item) for item in dataset]

    if group_by_labels is None and group_by_key is not None:
        group_by_labels = [group_by_key(item) for item in dataset]

    do_grouping = group_by_labels is not None

    # Do grouping first
    if do_grouping:
        index_groups = {}
        for i, label in enumerate(group_by_labels):
            if label not in index_groups:
                index_groups[label] = []
            index_groups[label].append(i)

        # Validate and convert stratify_labels
        group_stratify_labels = {}
        for label, group_indices in index_groups.items():
            current_group_strat_labels = set([stratify_labels[i] for i in group_indices])
            if len(current_group_strat_labels) > 1:
                raise ValueError(f"Group {label} has multiple stratification labels: {current_group_strat_labels}")
            group_stratify_labels[label] = current_group_strat_labels.pop()

        # Each value is a list of lists, a collection of groups
        stratified_groups = {}
        for label, strat_label in group_stratify_labels.items():
            if strat_label not in stratified_groups:
                stratified_groups[strat_label] = []
            stratified_groups[strat_label].append(index_groups[label])
        
        # Shuffle groups if needed
        if shuffle:
            for group_collection in stratified_groups.values():
                random.shuffle(group_collection)
                # Now actually shuflle each group
                for group in group_collection:
                    random.shuffle(group)
        
        train_indices = []
        val_indices = []

        # Now sample from each stratification category
        for group_collection in stratified_groups.values():
            group_size_counts = Counter([len(group) for group in group_collection])
            total_items = sum(len(group) for group in group_collection)
            n_train = int(p_train * total_items)
            n_val = total_items - n_train
            
            # Use DP to find the best combination of groups
            group_sizes = list(group_size_counts.keys())
            group_counts = list(group_size_counts.values())
            best_combination = find_best_combination(group_sizes, group_counts, min(n_train, n_val))

            if best_combination is None:
                raise ValueError("Could not find a valid combination of groups")

            #for the best combo
            small_indices = []
            for size, count in zip(group_sizes, best_combination):
                for _ in range(count):
                    #randomly pick a group with this size
                    to_pop_index = random.choice([i for i, group in enumerate(group_collection) if len(group) == size])
                    group = group_collection.pop(to_pop_index)
                    small_indices.extend(group)
            
            #now pick the remainder into big_indices
            big_indices = []
            for group in group_collection:
                big_indices.extend(group)
            
            if(n_train < n_val):
                train_indices.extend(small_indices)
                val_indices.extend(big_indices)
            else:
                train_indices.extend(big_indices)
                val_indices.extend(small_indices)   
    else:
        index_groups = {}
        for i, label in enumerate(stratify_labels):
            if label not in index_groups:
                index_groups[label] = []
            index_groups[label].append(i)

        train_indices = []
        val_indices = []

        for label, indices in index_groups.items():
            if shuffle:
                random.shuffle(indices)
            split_idx = int(len(indices) * p_train)
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set

