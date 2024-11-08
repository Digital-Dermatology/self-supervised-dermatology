import argparse
import copy
import gc
import itertools
import pickle
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.memory import (
    end_memory_monitor,
    init_memory_monitor,
    start_memory_monitor,
)
from ssl_library.src.datasets.downstream_tasks.ddi_dataset import DDILabel
from ssl_library.src.datasets.downstream_tasks.fitzpatrick17_dataset import (
    FitzpatrickLabel,
)
from ssl_library.src.datasets.downstream_tasks.passion_dataset import PASSIONLabel
from ssl_library.src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ssl_library.src.datasets.helper import DatasetName, get_dataset
from ssl_library.src.pkg import Embedder
from ssl_library.src.utils.utils import fix_random_seeds

data_quality_issues_dict = {
    DatasetName.MED_NODE: "assets/data_quality_issues_lists/MED-NODE_data_quality_issues.pickle",
    DatasetName.DDI: "assets/data_quality_issues_lists/DDI_data_quality_issues.pickle",
    DatasetName.DERM7PT: "assets/data_quality_issues_lists/Derm7pt_data_quality_issues.pickle",
    DatasetName.PAD_UFES_20: "assets/data_quality_issues_lists/PAD-UFES-20_data_quality_issues.pickle",
    DatasetName.SD_128: "assets/data_quality_issues_lists/SD-128_data_quality_issues.pickle",
    # HAM10000v2
    DatasetName.HAM10000: "assets/data_quality_issues_lists/HAM10000_ISIC_Nr_data_quality_issues.pickle",
    DatasetName.IMAGENET_1K: "assets/data_quality_issues_lists/ImageNet-1k_data_quality_issues.pickle",
}


def sample_x_points_of_class(
    df: pd.DataFrame,
    n_samples: int,
    label_col: str,
    sample_w_replacement: bool = False,
) -> Union[pd.DataFrame, None]:
    df_samples = pd.DataFrame()
    for lbl in df[label_col].unique():
        df_lbl = df.loc[df[label_col] == lbl]
        updated_n_samples = n_samples
        # stop the loop if the number of samples are lower then required
        # --> we'll then skip this
        if df_lbl.shape[0] < n_samples and not sample_w_replacement:
            return None
        df_sample = df_lbl.sample(n=updated_n_samples, replace=sample_w_replacement)
        df_samples = pd.concat([df_samples, df_sample], ignore_index=True)
    # shuffle the returned dataset
    return df_samples.sample(frac=1)


def evaluate_few_shot(
    train_ds,
    test_ds,
    lbl_col,
    path,
    df: pd.DataFrame,
    add_cols: dict,
    l_samples=[1, 3, 10, 30, 100, 300, 1000, None],
    n_repeats: int = 50,
    linear_evaluation: bool = False,
    num_workers: int = 20,
) -> pd.DataFrame:
    # monitor the memory
    process = init_memory_monitor()
    # create the test loader
    test_dataset = DataLoader(
        test_ds,
        batch_size=128,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    # store the f1 scores for the models
    f1_scores = {}
    ticks = set()
    for model_name in [
        "Monet",
        "ImageNet",
        "ImageNet_ViT_Tiny",
        "ColorMe",
        "SimCLR",
        "SimCLR_ImageNet",
        "BYOL",
        "DINO",
        "iBOT",
    ]:
        # monitor the memory
        mem_start = start_memory_monitor(process)
        # load the embedder
        print(f'{"-"*10}{model_name}{"-"*10}')
        if model_name != "GoogleDermFound":
            model = Embedder.load_pretrained(model_name.lower()).to(DEVICE).eval()
        else:
            model = None
        # embed test data
        test_labels, test_emb_space = embed_dataset(test_dataset, model)
        # define metrics for current model
        f1_scores[model_name] = {}
        f1_scores[model_name]["min_f1"] = []
        f1_scores[model_name]["max_f1"] = []
        f1_scores[model_name]["mean_f1"] = []
        f1_scores[model_name]["std_f1"] = []
        for n_samples in l_samples:
            # list of max. score when varying the neighbors
            # 1 entry (max. score) for every repeat
            l_f1s = []
            if n_samples is None:
                reps = 1
            else:
                reps = n_repeats
            for r in range(reps):
                print(f"Repeat: {r+1}/{reps}")
                t_ds = copy.copy(train_ds)
                if n_samples is not None:
                    meta_data = sample_x_points_of_class(
                        t_ds.meta_data, n_samples, lbl_col
                    )
                    if meta_data is None:
                        print(
                            f"Skipping {n_samples} since not all classes have the required amount of samples"
                        )
                        continue
                    t_ds.meta_data = meta_data
                    if type(t_ds) is EncryptedImageDataset:
                        set_meta_data_torch_loader(t_ds)
                if n_samples is not None:
                    ticks.add(n_samples)
                if linear_evaluation:
                    best_f1_score = train_lin(
                        t_ds,
                        model,
                        test_labels,
                        test_emb_space,
                        num_workers=num_workers,
                    )
                else:
                    best_f1_score = train_knn(
                        t_ds,
                        model,
                        test_labels,
                        test_emb_space,
                        lbl_col=lbl_col,
                        num_workers=num_workers,
                    )
                l_f1s.append(best_f1_score)
                del t_ds
                gc.collect()
            if len(l_f1s) > 0:
                f1_scores[model_name]["min_f1"].append(np.min(l_f1s))
                f1_scores[model_name]["max_f1"].append(np.max(l_f1s))
                f1_scores[model_name]["mean_f1"].append(np.mean(l_f1s))
                f1_scores[model_name]["std_f1"].append(np.std(l_f1s))
                print(
                    f"(SUMMARY) {str(n_samples)}: "
                    f"min F1 = {f1_scores[model_name]['min_f1'][-1]}, "
                    f"max F1 = {f1_scores[model_name]['max_f1'][-1]}, "
                    f"mean F1 = {f1_scores[model_name]['mean_f1'][-1]}, "
                    f"std F1 = {f1_scores[model_name]['std_f1'][-1]} "
                )

                df_row = {
                    **add_cols,
                    "model_name": model_name,
                    "f1s": l_f1s,
                    **f1_scores[model_name],
                }
                df = df.append(df_row, ignore_index=True)
        end_memory_monitor(process, mem_start)
    ticks = sorted(list(ticks))
    with open(
        f'{path.split(".")[0]}{"_lin" if linear_evaluation else "_knn"}.pkl', "wb"
    ) as f:
        save_dict = f1_scores
        save_dict["ticks"] = ticks
        pickle.dump(save_dict, f)
    return df


def set_meta_data_torch_loader(t_ds):
    t_ds.samples = t_ds.meta_data.values.tolist()
    t_ds.samples = [tuple(x) for x in t_ds.samples]
    t_ds.imgs = t_ds.samples
    t_ds.targets = t_ds.meta_data.label.tolist()


def train_knn(
    t_ds,
    model,
    test_labels,
    test_emb_space,
    lbl_col: str,
    num_workers: int = 20,
):
    train_dataset = DataLoader(
        t_ds,
        batch_size=128,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    if any(t_ds.meta_data[lbl_col].value_counts() > 1):
        split_ratio = args.train_test_split
        if round(len(t_ds) * split_ratio) < len(t_ds.meta_data[lbl_col].unique()):
            split_ratio = len(t_ds.meta_data[lbl_col].unique()) / len(t_ds)
        train_range, val_range = train_test_split(
            list(range(len(t_ds))),
            test_size=split_ratio,
            random_state=random_seed,
            stratify=t_ds.meta_data[lbl_col],
        )
    else:
        # for the case where there is only one sample train = val
        # which does not matter because `k` can only be 1
        train_range = list(range(len(t_ds)))
        val_range = list(range(len(t_ds)))
    labels, emb_space = embed_dataset(train_dataset, model)
    poss_f1_val, poss_f1_test = [], []
    for n_neigh in [1, 5, 10, 20, 30, 50, 100, 200]:
        try:
            neigh = KNeighborsClassifier(
                n_neighbors=n_neigh,
                metric="cosine",
            )
            neigh.fit(emb_space[train_range].squeeze(), labels[train_range])
            # NOTE: maybe we also want to report something different for different tasks
            # e.g. balanced Acc for multi-class, F1 for binary
            val_f1 = (
                f1_score(
                    labels[val_range],
                    neigh.predict(emb_space[val_range]),
                    average="macro" if len(np.unique(labels)) > 2 else "binary",
                )
                * 100
            )
            test_f1 = (
                f1_score(
                    test_labels,
                    neigh.predict(test_emb_space),
                    average="macro" if len(np.unique(labels)) > 2 else "binary",
                )
                * 100
            )
        except Exception:
            continue
        poss_f1_val.append(val_f1)
        poss_f1_test.append(test_f1)
    del train_dataset

    idx_best_val_f1 = np.argmax(poss_f1_val)
    return poss_f1_test[idx_best_val_f1]


def train_lin(
    t_ds,
    model,
    test_labels,
    test_emb_space,
    num_workers: int = 20,
):
    train_dataset = DataLoader(
        t_ds,
        batch_size=128,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    labels, emb_space = embed_dataset(train_dataset, model)
    lin = LogisticRegression(
        max_iter=10_000,
    )
    lin.fit(emb_space.squeeze(), labels)
    f1 = (
        f1_score(
            test_labels,
            lin.predict(test_emb_space),
            average="macro" if len(np.unique(labels)) > 2 else "binary",
        )
        * 100
    )
    del train_dataset
    return f1


def embed_dataset(data_loader, model, return_all_tokens: bool = False):
    l_emb_space = []
    l_labels = []
    if model is None:
        data_loader.dataset.return_embedding = True
    for batch_tup in tqdm(data_loader):
        if len(batch_tup) == 3:
            batch, _, label = batch_tup
        elif len(batch_tup) == 2:
            batch, label = batch_tup
        else:
            raise ValueError("Unknown batch tuple.")

        if model is None:
            # here the loader already returns the pre-computed embedding
            l_emb_space.append(batch)
            l_labels.append(label)
            continue

        batch = batch.to(DEVICE)
        with torch.no_grad():
            if not return_all_tokens:
                emb = model(batch)
            else:
                emb = model(batch, return_all_tokens=True)
                emb = emb[:, 1:, :]
            l_emb_space.append(emb)
            l_labels.append(label)
    l_emb_space = torch.concat(l_emb_space).cpu().squeeze()
    l_labels = torch.concat(l_labels).cpu().squeeze()
    if model is None:
        # make sure to reset this!
        data_loader.dataset.return_embedding = False
    return l_labels, l_emb_space


def evaluate_dataset_w_name(
    dataset_name: DatasetName,
    args,
    transform,
    random_seed: int,
    df: pd.DataFrame,
    add_cols: dict,
    **kwargs,
) -> pd.DataFrame:
    if dataset_name is DatasetName.PASSION:
        dataset_path = args.data_path
        if kwargs.get("dataset_path"):
            dataset_path = kwargs.pop("dataset_path")
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            transform=transform,
            return_loader=False,
            **kwargs,
        )
        train_ds = copy.deepcopy(dataset)
        test_ds = copy.deepcopy(dataset)

        train_ds.meta_data = train_ds.meta_data[train_ds.meta_data["Split"] == "TRAIN"]
        test_ds.meta_data = test_ds.meta_data[test_ds.meta_data["Split"] == "TEST"]

        lbl_col = dataset.LBL_COL
    elif dataset_name is DatasetName.PAD_UFES_20:
        dataset_path = args.data_path
        if kwargs.get("dataset_path"):
            dataset_path = kwargs.pop("dataset_path")
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            data_quality_issues_list=data_quality_issues_dict.get(dataset_name),
            transform=transform,
            return_loader=False,
            **kwargs,
        )
        train_ds = copy.deepcopy(dataset)
        test_ds = copy.deepcopy(dataset)
        # splitting based on patient IDs
        patients_train, patients_test = train_test_split(
            dataset.meta_data["patient_id"].unique(),
            test_size=args.train_test_split,
            random_state=random_seed,
        )
        train_ds.meta_data = train_ds.meta_data[
            train_ds.meta_data["patient_id"].isin(patients_train)
        ]
        test_ds.meta_data = test_ds.meta_data[
            test_ds.meta_data["patient_id"].isin(patients_test)
        ]
        lbl_col = dataset.LBL_COL
    elif dataset_name is DatasetName.HAM10000:
        dataset_path = args.data_path
        if kwargs.get("dataset_path"):
            dataset_path = kwargs.pop("dataset_path")
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            data_quality_issues_list=data_quality_issues_dict.get(dataset_name),
            transform=transform,
            return_loader=False,
            **kwargs,
        )
        train_ds = copy.deepcopy(dataset)
        test_ds = copy.deepcopy(dataset)

        train_ds.meta_data = train_ds.meta_data[
            train_ds.meta_data["dataset_origin"] == "Train"
        ]
        test_ds.meta_data = test_ds.meta_data[
            test_ds.meta_data["dataset_origin"] == "Test"
        ]

        lbl_col = dataset.LBL_COL
    else:
        dataset_path = args.data_path
        if kwargs.get("dataset_path"):
            dataset_path = kwargs.pop("dataset_path")
        dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
            data_quality_issues_list=data_quality_issues_dict.get(dataset_name),
            transform=transform,
            return_loader=False,
            **kwargs,
        )
        lbl_col = dataset.LBL_COL

        train_ds = copy.deepcopy(dataset)
        test_ds = copy.deepcopy(dataset)
        train_ds.meta_data, test_ds.meta_data = train_test_split(
            dataset.meta_data,
            test_size=args.train_test_split,
            random_state=random_seed,
            stratify=dataset.meta_data[lbl_col],
        )
    return evaluate_few_shot(
        train_ds=train_ds,
        test_ds=test_ds,
        lbl_col=lbl_col,
        path=f"{add_cols['dataset_name']}_{add_cols['overall_seed']}.jpg",
        df=df,
        add_cols=add_cols,
        l_samples=(
            [None]
            if args.knn_performance or args.linear_evaluation
            else [1, 3, 10, 30, 100, 300, 1000]
        ),
        n_repeats=args.n_repeats,
        linear_evaluation=args.linear_evaluation,
        num_workers=args.num_workers,
    )


my_parser = argparse.ArgumentParser(
    description="Generate kNN figures for classification figures."
)
my_parser.add_argument(
    "--data_path",
    type=str,
    default="data/",
    help="Path to the datasets.",
)
my_parser.add_argument(
    "--train_test_split",
    type=float,
    default=0.15,
    help="Train test split fraction.",
)
my_parser.add_argument(
    "--n_repeats",
    type=int,
    default=50,
    help="Train test split fraction.",
)
my_parser.add_argument(
    "--num_workers",
    type=int,
    default=20,
    help="Number of workers for data loaders.",
)
my_parser.add_argument(
    "--knn_performance",
    action="store_true",
    help="If only the kNN performance on the whole data should be reported.",
)
my_parser.add_argument(
    "--linear_evaluation",
    action="store_true",
    help="If instead of fitting a kNN a linear clf should be used.",
)
my_parser.add_argument(
    "--datasets",
    nargs="+",
    default=[
        "med_node",
        "ph2",
        "ddi-malignant",
        # NOTE: removed because it contains single class samples
        # "ddi-disease",
        "derm7pt",
        "pad_ufes_20",
        "SD_128",
        "passion-conditions",
        "passion-impetigo",
        "ham10000",
        "fitzpatrick17k-high",
        "fitzpatrick17k-mid",
        "fitzpatrick17k-low",
    ],
    help="Name of the datasets to evaluate on.",
)
args = my_parser.parse_args()


if __name__ == "__main__":
    l_SEED = [1, 9, 21, 42, 555]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.DataFrame()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    base_eval_func = partial(
        evaluate_dataset_w_name,
        args=args,
        transform=transform,
    )
    dataset_dict = {
        "med_node": partial(
            base_eval_func,
            dataset_name=DatasetName.MED_NODE,
        ),
        "ph2": partial(
            base_eval_func,
            dataset_name=DatasetName.PH2,
        ),
        "ddi-malignant": partial(
            base_eval_func,
            dataset_name=DatasetName.DDI,
            label_col=DDILabel.MALIGNANT,
        ),
        "ddi-disease": partial(
            base_eval_func,
            dataset_name=DatasetName.DDI,
            label_col=DDILabel.DISEASE,
        ),
        "derm7pt": partial(
            base_eval_func,
            dataset_name=DatasetName.DERM7PT,
        ),
        "SD_128": partial(
            base_eval_func,
            dataset_name=DatasetName.SD_128,
        ),
        "fitzpatrick17k": partial(
            base_eval_func,
            dataset_name=DatasetName.FITZPATRICK17K,
        ),
        "fitzpatrick17k-high": partial(
            base_eval_func,
            dataset_name=DatasetName.FITZPATRICK17K,
            label_col=FitzpatrickLabel.HIGH,
        ),
        "fitzpatrick17k-mid": partial(
            base_eval_func,
            dataset_name=DatasetName.FITZPATRICK17K,
            label_col=FitzpatrickLabel.MID,
        ),
        "fitzpatrick17k-low": partial(
            base_eval_func,
            dataset_name=DatasetName.FITZPATRICK17K,
            label_col=FitzpatrickLabel.LOW,
        ),
        "pad_ufes_20": partial(
            base_eval_func,
            dataset_name=DatasetName.PAD_UFES_20,
            # pre_computed_embeddings_path="assets/google_foundation_embeddings/GDF_pad_ufes_20.pickle",
        ),
        "ham10000": partial(
            base_eval_func,
            dataset_name=DatasetName.HAM10000,
        ),
        "passion-conditions": partial(
            base_eval_func,
            dataset_name=DatasetName.PASSION,
            dataset_path="data/PASSION/PASSION_collection_2020_2023",
            split_file="PASSION_split.csv",
            label_col=PASSIONLabel.CONDITIONS,
            # pre_computed_embeddings_path="assets/google_foundation_embeddings/GDF_passion.pickle",
        ),
        "passion-impetigo": partial(
            base_eval_func,
            dataset_name=DatasetName.PASSION,
            dataset_path="data/PASSION/PASSION_collection_2020_2023",
            split_file="PASSION_split.csv",
            label_col=PASSIONLabel.IMPETIGO,
            # pre_computed_embeddings_path="assets/google_foundation_embeddings/GDF_passion.pickle",
        ),
    }

    for dataset_name, random_seed in itertools.product(args.datasets, l_SEED):
        dataset_func = dataset_dict.get(dataset_name, None)
        if dataset_func is None:
            raise ValueError("Unknown dataset.")
        print(f"{'*'*20} Evaluation of {dataset_name} {'*'*20}")
        add_cols = {"dataset_name": dataset_name, "overall_seed": random_seed}
        fix_random_seeds(random_seed)
        df = dataset_func(df=df, add_cols=add_cols, random_seed=random_seed)
    out_str = ""
    if args.knn_performance:
        out_str += "knn"
    elif args.linear_evaluation:
        out_str += "lin"
    out_str += f"_Seed_{'_'.join([str(x) for x in l_SEED])}_performance.csv"
    df.to_csv(out_str, index=False)
