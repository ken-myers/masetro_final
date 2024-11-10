import random
from torch.utils.data import Subset

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


def stratify_split(dataset, *, stratify_key=None, stratify_labels = None, p_train = 0.8, shuffle=True):

    if stratify_key is not None and stratify_labels is not None:
        raise ValueError("Only one of stratify_key and stratify_labels should be provided.")

    if stratify_key is None and stratify_labels is None:
        raise ValueError("One of stratify_key and stratify_labels should be provided.")

    if isinstance(stratify_key, str):
        key_copy = stratify_key
        stratify_key = lambda x: x[key_copy]

    index_groups = {}

    if stratify_labels is None:
        stratify_labels = [stratify_key(item) for item in dataset]

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

