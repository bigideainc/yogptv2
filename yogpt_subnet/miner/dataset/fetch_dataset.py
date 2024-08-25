from datasets import load_dataset

def fetch_dataset(dataset_id, config=None, split=None):
    """
    Fetches a dataset from Hugging Face, with optional configuration and split.
    
    Args:
    dataset_id (str): Identifier for the dataset to be fetched.
    config (str, optional): Configuration or subset of the dataset if applicable.
    split (str, optional): Specific split of the dataset to fetch, e.g., 'train', 'test', 'validation'.

    Returns:
    Dataset or DatasetDict: Loaded dataset object.
    """
    if config and split:
        # Load specific configuration and specific split
        return load_dataset(dataset_id, config, split=split)
    elif config:
        # Load specific configuration, all splits
        return load_dataset(dataset_id, config)
    elif split:
        # Load default configuration but specific split
        return load_dataset(dataset_id, split=split)
    else:
        # Load default configuration and all splits
        return load_dataset(dataset_id)