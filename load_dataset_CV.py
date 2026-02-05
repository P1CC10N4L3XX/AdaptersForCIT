import os

from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import config
from utils import *

def make_cases_dataset_CV():
    dataset_cases = get_local_case_dataset()
    domain_folds = {}

    for domain_name in dataset_cases.keys():
        dataset_cases_for_domain = np.array(dataset_cases[domain_name])
        norm_types = [e['norm_type'] for e in dataset_cases_for_domain]

        skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        domain_folds[domain_name] = []

        for train_index, test_index in skf.split(dataset_cases_for_domain, norm_types):
            train = dataset_cases_for_domain[train_index].tolist()
            test = dataset_cases_for_domain[test_index].tolist()
            domain_folds[domain_name].append((train,test))
    return domain_folds

def save_k_fold_dataset(domain_folds, output_path=config.K_FOLD_DATASET_PATH):
    os.makedirs(output_path, exist_ok = True)
    num_folds = len(next(iter(domain_folds.values())))
    for fold_id in range(num_folds):
        fold_dir = os.path.join(output_path, f'fold_{fold_id}')
        os.makedirs(fold_dir, exist_ok = True)

        for domain_name, folds in domain_folds.items():
            train_fold, test_fold = folds[fold_id]
            domain_dir = os.path.join(fold_dir, domain_name)
            os.makedirs(domain_dir, exist_ok = True)

            train_dataset = Dataset.from_list(train_fold)
            test_dataset = Dataset.from_list(test_fold)

            ds_dict = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })

            ds_dict.save_to_disk(domain_dir)
    print("✅ Dataset saved successfully for k-fold cross-validation.")

def load_k_fold_dataset(input_path=config.K_FOLD_DATASET_PATH):
    k_fold_data = {}
    for fold_name in sorted(os.listdir(input_path)):
        fold_path = os.path.join(input_path, fold_name)
        if not os.path.isdir(fold_path):
            continue
        fold_id = int(fold_name.replace("fold_",""))
        k_fold_data[fold_id] = {}
        for domain_name in os.listdir(fold_path):
            domain_path = os.path.join(fold_path, domain_name)
            if not os.path.isdir(domain_path):
                continue
            ds_dict = load_from_disk(domain_path)
            k_fold_data[fold_id][domain_name]=ds_dict
    return k_fold_data


if __name__ == "__main__":
    domain_folds = make_cases_dataset_CV()
    save_k_fold_dataset(domain_folds)
    print(load_k_fold_dataset()[0]['GDPR']['train'])