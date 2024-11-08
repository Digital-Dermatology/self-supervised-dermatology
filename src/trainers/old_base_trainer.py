import copy
import math
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

from ssl_library.src.datasets.downstream_tasks.fitzpatrick17_dataset import (
    Fitzpatrick17kDataset,
)
from ssl_library.src.datasets.downstream_tasks.ham10000_dataset import HAM10000Dataset
from ssl_library.src.datasets.downstream_tasks.isic_2018_task_1_dataset import (
    ISICTask1Dataset,
)
from ssl_library.src.datasets.downstream_tasks.pad_ufes_20_dataset import (
    PADUFES20Dataset,
)
from ssl_library.src.datasets.downstream_tasks.ppp_dataset import PPPDataset
from ssl_library.src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ssl_library.src.datasets.utils import get_train_validation_data_loaders
from ssl_library.src.losses.utils import (
    get_segmentation_loss,
    mixup_criterion,
    mixup_data,
)
from ssl_library.src.models.fine_tuning.classifiers import LinearClassifier
from ssl_library.src.models.utils import ModelType
from ssl_library.src.optimizers.utils import get_lin_scaled_optimizer
from ssl_library.src.pkg.embedder import Embedder
from ssl_library.src.pkg.segmenter import Segmenter
from ssl_library.src.utils.logging import (
    log_segmentation_pred,
    visualize_nearest_neighbors,
    visualize_self_attention,
)
from ssl_library.src.utils.metrics import calculate_embedding_entropy, get_seg_metrics
from ssl_library.src.utils.utils import (
    EarlyStopping,
    fix_random_seeds,
    has_batchnorms,
    is_dist_avail_and_initialized,
    is_main_process,
    set_requires_grad,
)


class Trainer(ABC, object):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        config: dict,
        config_path: Union[str, Path],
        arch_name: str,
        debug=False,
        evaluation: bool = False,
        project_name="SSL",
    ):
        self.config = config
        self.config_path = config_path
        self.arch_name = arch_name
        self.device = self._get_device()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.start_epoch = 1
        self.debug = debug
        self.multi_gpu = False
        self.dist_training = is_dist_avail_and_initialized()

        if not evaluation:
            # logging to W&B
            wandb.init(
                config=self.config,
                project=project_name,
                group=arch_name if not evaluation else "downstream",
            )

            # distributed training configuration
            if self.dist_training:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                run_name = f"{arch_name}-{wandb.run.name}-rank-{self.local_rank}"
            else:
                run_name = f"{arch_name}-{wandb.run.name}"

            # update the name of the run
            wandb.run.name = run_name
            wandb.run.save()
            self.run_dir = Path(wandb.run.dir)

            # set all the required attributes of the model
            self.set_model_attributes()
            # logging
            print(
                f"Data loaded: there are "
                f"{len(self.train_dataset)*self.config['batch_size']} train images."
            )
            print(
                f"Data loaded: there are "
                f"{len(self.val_dataset)*self.config['batch_size']} val images."
            )
        # debug pytorch code
        torch.autograd.set_detect_anomaly(True)
        # optimize various tensor operations automatically
        torch.backends.cudnn.benchmark = True

    @abstractmethod
    def fit(self):
        pass

    def _get_embedding(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.multi_gpu:
            model = model.module
        if self.model_type is ModelType.VIT:
            n = self.config["model"]["eval"]["n_last_blocks"]
            if "backbone" in dir(model):
                inter_out = model.backbone.get_intermediate_layers(images, n)
            else:
                inter_out = model.get_intermediate_layers(images, n)
            emb = torch.cat([x[:, 0] for x in inter_out], dim=-1)
            if self.config["model"]["eval"]["avgpool_patchtokens"]:
                emb = torch.cat(
                    (
                        emb.unsqueeze(-1),
                        torch.mean(inter_out[-1][:, 1:], dim=1).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                emb = emb.reshape(emb.shape[0], -1)
        else:
            emb = model.backbone(images)
        emb = emb.squeeze()
        return emb

    @property
    def get_ckp_path(self) -> Path:
        return self.run_dir / self.config["fine_tune_from"] / "checkpoints"

    def set_model_attributes(self):
        # get model type
        self.model_type = ModelType[self.config["model"]["model_type"]]
        if self.model_type is None:
            raise ValueError("Wrong model type")
        if self.model_type is ModelType.VIT:
            self.embed_dim = (
                self.config["model"]["emb_dim"]
                * self.config["model"]["eval"]["n_last_blocks"]
            )
        else:
            self.embed_dim = self.config["model"]["emb_dim"]
            if "base_model" in self.config["model"]:
                embed_dict = {
                    "resnet18": 512,
                    "resnet50": 2048,
                }
                self.embed_dim = embed_dict.get(
                    self.config["model"]["base_model"], self.embed_dim
                )

    def distribute_model(
        self,
        model: torch.nn.Module,
        broadcast_buffers: bool = True,
    ) -> torch.nn.Module:
        if torch.cuda.device_count() > 1:
            print(
                "Multiple GPUs detected, model will run on "
                f"{torch.cuda.device_count()} GPUs!"
            )
            if self.dist_training:
                if has_batchnorms(model=model):
                    print("Batch norms detected, will sync them.")
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                print("Distributed training, distributing the model.")
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=broadcast_buffers,
                )
            else:
                model = torch.nn.DataParallel(model)
            self.multi_gpu = True
        else:
            print("Single GPU detected, model will run on single instance.")
        return model

    def load_downstream_tasks(
        self, dataset: str, ssl_model: Union[str, None] = None, fine_tune: bool = False
    ):
        # configs
        if fine_tune:
            l_transforms = [
                transforms.RandomResizedCrop(self.run_config["img_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(self.run_config["rand_rotation"]),
                transforms.ColorJitter(
                    brightness=self.run_config["color_jitter"],
                    contrast=self.run_config["color_jitter"],
                    hue=self.run_config["color_jitter"],
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
            if self.run_config["use_rand_erasing"]:
                l_transforms.append(transforms.RandomErasing())
        else:
            l_transforms = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.run_config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]

        train_transform = transforms.Compose(l_transforms)
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.run_config["img_size"],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.run_config["img_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        d_config = self.config["downstream"]

        if dataset == "pad_ufes_20":
            # PADUFES20Dataset (diagnosis, classification)
            self.load_pad_ufes_20(
                d_config,
                train_transform,
                val_transform,
                ssl_model=ssl_model,
            )
        if dataset == "ham10000":
            # HAM10000Dataset (diagnosis, classification)
            self.load_ham10000(
                d_config,
                train_transform,
                val_transform,
                ssl_model=ssl_model,
            )
        if dataset == "fitzpatrick17k":
            # Fitzpatrick17kDataset (diagnosis, classification)
            self.load_fitzpatrick17k(
                d_config,
                train_transform,
                val_transform,
                ssl_model=ssl_model,
            )
        if dataset == "body_loc":
            # Body localization
            self.load_body_loc(
                d_config,
                train_transform,
                val_transform,
                ssl_model=ssl_model,
            )
        if dataset == "isic_task1":
            # ISIC Task 1 (segmentation)
            self.load_isic_task1(d_config, ssl_model)
        if dataset == "ppp":
            # PPP (segmentation)
            self.load_ppp(d_config, ssl_model)

    def load_pad_ufes_20(
        self,
        d_config: dict,
        train_transform: transforms.Compose,
        val_transform: transforms.Compose,
        ssl_model: Union[str, None] = None,
    ):
        if d_config["pu_path"] is not None:
            pu_path = Path(d_config["pu_path"])
            self.pu_dataset = PADUFES20Dataset(
                csv_file=pu_path / "metadata.csv",
                root_dir=pu_path / "images",
                transform=train_transform,
                val_transform=val_transform,
            )
            (
                self.pu_train,
                self.pu_val,
                self.pu_test,
            ) = get_train_validation_data_loaders(
                self.pu_dataset,
                self.run_config["batch_size"],
                **d_config["splitter"],
            )
            # classifier
            if ssl_model is not None:
                model, info, _ = Embedder.load_pretrained(
                    ssl_model,
                    return_info=True,
                    n_head_layers=self.run_config["n_head_layers"],
                )
                print(f"Loaded pretrained SSL model: {info}")
                self.pu_clf = torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("backbone", model),
                            ("flatten", torch.nn.Flatten()),
                            (
                                "fc",
                                LinearClassifier(
                                    info.out_dim,
                                    self.pu_dataset.n_classes,
                                    dropout_rate=self.run_config["dropout"],
                                    use_dropout_in_head=self.run_config[
                                        "use_dropout_in_head"
                                    ],
                                    large_head=self.run_config["large_head"],
                                    use_bn=self.run_config["use_bn"],
                                    log_softmax=self.run_config["loss_fn"] == "nll",
                                ),
                            ),
                        ]
                    )
                )
            else:
                self.pu_clf = LinearClassifier(
                    self.embed_dim,
                    self.pu_dataset.n_classes,
                    dropout_rate=self.run_config["dropout"],
                    use_dropout_in_head=self.run_config["use_dropout_in_head"],
                    large_head=self.run_config["large_head"],
                    use_bn=self.run_config["use_bn"],
                    log_softmax=self.run_config["loss_fn"] == "nll",
                )
            self.pu_clf = self.pu_clf.to(self.device)
            self.pu_clf = self.distribute_model(self.pu_clf)

    def load_ham10000(
        self,
        d_config: dict,
        transform: transforms.Compose,
        val_transform: transforms.Compose,
        ssl_model: str = None,
    ):
        if d_config["ham_path"] is not None:
            ham_path = Path(d_config["ham_path"])
            self.ham_dataset = HAM10000Dataset(
                csv_file=ham_path / "HAM10000_metadata.csv",
                dataset_dir=ham_path,
                transform=transform,
                val_transform=val_transform,
            )
            (
                self.ham_train,
                self.ham_val,
                self.ham_test,
            ) = get_train_validation_data_loaders(
                self.ham_dataset,
                self.run_config["batch_size"],
                **d_config["splitter"],
            )
            # classifier
            if ssl_model is not None:
                model, info, _ = Embedder.load_pretrained(
                    ssl_model,
                    return_info=True,
                    n_head_layers=self.run_config["n_head_layers"],
                )
                print(f"Loaded pretrained SSL model: {info}")
                self.ham_clf = torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("backbone", model),
                            ("flatten", torch.nn.Flatten()),
                            (
                                "fc",
                                LinearClassifier(
                                    info.out_dim,
                                    self.ham_dataset.n_classes,
                                    dropout_rate=self.run_config["dropout"],
                                    use_dropout_in_head=self.run_config[
                                        "use_dropout_in_head"
                                    ],
                                    large_head=self.run_config["large_head"],
                                    use_bn=self.run_config["use_bn"],
                                    log_softmax=self.run_config["loss_fn"] == "nll",
                                ),
                            ),
                        ]
                    )
                )
            else:
                self.ham_clf = LinearClassifier(
                    self.embed_dim,
                    self.ham_dataset.n_classes,
                    dropout_rate=self.run_config["dropout"],
                    use_dropout_in_head=self.run_config["use_dropout_in_head"],
                    large_head=self.run_config["large_head"],
                    use_bn=self.run_config["use_bn"],
                    log_softmax=self.run_config["loss_fn"] == "nll",
                )
            self.ham_clf = self.ham_clf.to(self.device)
            self.ham_clf = self.distribute_model(self.ham_clf)

    def load_fitzpatrick17k(
        self,
        d_config: dict,
        transform: transforms.Compose,
        val_transform: transforms.Compose,
        ssl_model: str = None,
    ):
        if d_config["fitz_path"] is not None:
            fitz_path = Path(d_config["fitz_path"])
            self.fitz_dataset = Fitzpatrick17kDataset(
                csv_file=fitz_path / "fitzpatrick17k.csv",
                dataset_dir=fitz_path,
                transform=transform,
                val_transform=val_transform,
            )
            (
                self.fitz_train,
                self.fitz_val,
                self.fitz_test,
            ) = get_train_validation_data_loaders(
                self.fitz_dataset,
                self.run_config["batch_size"],
                **d_config["splitter"],
            )
            # classifier
            if ssl_model is not None:
                model, info, _ = Embedder.load_pretrained(
                    ssl_model,
                    return_info=True,
                    n_head_layers=self.run_config["n_head_layers"],
                )
                print(f"Loaded pretrained SSL model: {info}")
                self.fitz_clf = torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("backbone", model),
                            ("flatten", torch.nn.Flatten()),
                            (
                                "fc",
                                LinearClassifier(
                                    info.out_dim,
                                    self.fitz_dataset.n_classes,
                                    dropout_rate=self.run_config["dropout"],
                                    use_dropout_in_head=self.run_config[
                                        "use_dropout_in_head"
                                    ],
                                    large_head=self.run_config["large_head"],
                                    use_bn=self.run_config["use_bn"],
                                    log_softmax=self.run_config["loss_fn"] == "nll",
                                ),
                            ),
                        ]
                    )
                )
            else:
                self.fitz_clf = LinearClassifier(
                    self.embed_dim,
                    self.fitz_dataset.n_classes,
                    dropout_rate=self.run_config["dropout"],
                    use_dropout_in_head=self.run_config["use_dropout_in_head"],
                    large_head=self.run_config["large_head"],
                    use_bn=self.run_config["use_bn"],
                    log_softmax=self.run_config["loss_fn"] == "nll",
                )
            self.fitz_clf = self.fitz_clf.to(self.device)
            self.fitz_clf = self.distribute_model(self.fitz_clf)

    def load_body_loc(
        self,
        d_config: dict,
        transform: transforms.Compose,
        val_transform: transforms.Compose,
        ssl_model: str = None,
    ):
        if d_config["body_loc_path"] is not None:
            base_path = Path(d_config["body_loc_path"])
            train_path = base_path / "strong_labels_train_no_leak"
            val_path = base_path / "strong_labels_val_no_leak"
            test_path = base_path / "strong_labels_test"
            # Train set
            self.body_loc_dataset = EncryptedImageDataset(
                train_path,
                enc_keys=[str(base_path / "key.key")],
                transform=transform,
                val_transform=val_transform,
            )
            self.body_loc_train = DataLoader(
                self.body_loc_dataset,
                batch_size=self.run_config["batch_size"],
                num_workers=self.config["downstream"]["splitter"]["num_workers"],
                drop_last=True,
                shuffle=True,
            )
            # Val set
            self.body_loc_val = EncryptedImageDataset(
                val_path,
                enc_keys=[str(base_path / "key.key")],
                transform=transform,
                val_transform=val_transform,
            )
            self.body_loc_val = DataLoader(
                self.body_loc_val,
                batch_size=self.run_config["batch_size"],
                num_workers=self.config["downstream"]["splitter"]["num_workers"],
                drop_last=False,
                shuffle=False,
            )
            # Test set
            self.body_loc_test = EncryptedImageDataset(
                test_path,
                enc_keys=[str(base_path / "key.key")],
                transform=val_transform,
            )
            self.body_loc_test = DataLoader(
                self.body_loc_test,
                batch_size=self.run_config["batch_size"],
                num_workers=self.config["downstream"]["splitter"]["num_workers"],
                drop_last=False,
                shuffle=False,
            )
            # classifier
            if ssl_model is not None:
                model, info, _ = Embedder.load_pretrained(
                    ssl_model,
                    return_info=True,
                    n_head_layers=self.run_config["n_head_layers"],
                )
                print(f"Loaded pretrained SSL model: {info}")
                self.body_loc_clf = torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("backbone", model),
                            ("flatten", torch.nn.Flatten()),
                            (
                                "fc",
                                LinearClassifier(
                                    info.out_dim,
                                    len(self.body_loc_dataset.classes),
                                    dropout_rate=self.run_config["dropout"],
                                    use_dropout_in_head=self.run_config[
                                        "use_dropout_in_head"
                                    ],
                                    large_head=self.run_config["large_head"],
                                    use_bn=self.run_config["use_bn"],
                                    log_softmax=self.run_config["loss_fn"] == "nll",
                                ),
                            ),
                        ]
                    )
                )
            else:
                self.body_loc_clf = LinearClassifier(
                    self.embed_dim,
                    len(self.body_loc_dataset.classes),
                    dropout_rate=self.run_config["dropout"],
                    use_dropout_in_head=self.run_config["use_dropout_in_head"],
                    large_head=self.run_config["large_head"],
                    use_bn=self.run_config["use_bn"],
                    log_softmax=self.run_config["loss_fn"] == "nll",
                )
            self.body_loc_clf = self.body_loc_clf.to(self.device)
            self.body_loc_clf = self.distribute_model(self.body_loc_clf)

    def load_isic_task1(
        self,
        d_config: dict,
        ssl_model: Union[str, None] = None,
    ):
        if d_config["isic_path"] is None:
            raise ValueError("ISIC dataset path must be specified.")

        # dataset
        isic_path = Path(d_config["isic_path"])
        self.isic_train = ISICTask1Dataset(dataset_dir=isic_path, dataset_type="train")
        self.isic_val = ISICTask1Dataset(dataset_dir=isic_path, dataset_type="val")
        self.isic_train = DataLoader(
            self.isic_train,
            batch_size=self.run_config["batch_size"],
            num_workers=d_config["splitter"]["num_workers"],
            drop_last=True,
            shuffle=True,
        )
        self.isic_val = DataLoader(
            self.isic_val,
            batch_size=self.run_config["batch_size"],
            num_workers=d_config["splitter"]["num_workers"],
            drop_last=False,
            shuffle=False,
        )
        self.isic_dataset = [(self.isic_train, self.isic_val)]

        # decoder
        if ssl_model is not None:
            self.isic_dec, info = Segmenter.load_pretrained(
                ssl_model, n_classes=1, return_info=True
            )
            print(f"Loaded pretrained SSL model: {info}")
        self.isic_dec = self.isic_dec.to(self.device)
        self.isic_dec = self.distribute_model(self.isic_dec)

    def load_ppp(self, d_config: dict, ssl_model: str = None):
        if d_config["ppp_path"] is not None:
            dataset_path = Path(d_config["ppp_path"])
            train_path = dataset_path / "train_no_leak"
            val_path = dataset_path / "val_no_leak"
            test_path = dataset_path / "test"
            key_path = str(dataset_path / "key.key")

            self.ppp_train = PPPDataset(
                train_path,
                enc_key_path=key_path,
                dataset_type="train",
            )
            self.ppp_val = PPPDataset(
                val_path,
                enc_key_path=key_path,
                dataset_type="val",
            )
            self.ppp_test = PPPDataset(
                test_path,
                enc_key_path=key_path,
                dataset_type="val",
            )

            self.ppp_train = DataLoader(
                self.ppp_train,
                batch_size=self.run_config["batch_size"],
                num_workers=d_config["splitter"]["num_workers"],
                drop_last=True,
                shuffle=True,
            )
            self.ppp_val = DataLoader(
                self.ppp_val,
                batch_size=self.run_config["batch_size"],
                num_workers=d_config["splitter"]["num_workers"],
                drop_last=False,
                shuffle=False,
            )
            self.ppp_test = DataLoader(
                self.ppp_test,
                batch_size=self.run_config["batch_size"],
                num_workers=d_config["splitter"]["num_workers"],
                drop_last=False,
                shuffle=False,
            )

            if ssl_model is not None:
                self.ppp_dec, info = Segmenter.load_pretrained(
                    ssl_model,
                    n_classes=self.ppp_train.dataset.n_classes,
                    return_info=True,
                )
                print(f"Loaded pretrained SSL model: {info}")
            self.ppp_dec = self.ppp_dec.to(self.device)
            self.ppp_dec = self.distribute_model(self.ppp_dec)

    def eval_classification_task(
        self,
        train_loader,
        val_loader,
        classifier: torch.nn.Module,
        task_name: str,
        test_loader=None,
        fine_tune: bool = False,
    ):
        if fine_tune:
            # make sure the classifier can get trained
            set_requires_grad(classifier, True)
        else:
            # freeze the backbone and let only the classifier be trained
            set_requires_grad(classifier, True)
            set_requires_grad(classifier.backbone, False)

        try:
            in_size = (1, 3, self.run_config["img_size"], self.run_config["img_size"])
            summary(classifier, input_size=in_size)
        except RuntimeError:
            print(f"Summary can not be displayed for a Huggingface model.")
            print(
                f"Number of parameters backbone: {classifier.backbone.model.num_parameters():,}"
            )
        wandb.watch(classifier, log="all")
        # loss function, optimizer, scores
        if self.run_config["loss_fn"] == "ce":
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=self.run_config["label_smoothing"],
                weight=train_loader.dataset.get_class_weights(),
            )
        elif self.run_config["loss_fn"] == "nll":
            criterion = torch.nn.NLLLoss(
                weight=train_loader.dataset.get_class_weights(),
            )
        else:
            raise ValueError(
                f'Unrecognized loss function: {self.run_config["loss_fn"]}'
            )
        criterion = criterion.to(self.device)

        optimizer = get_lin_scaled_optimizer(
            model=classifier,
            optimizer_name=self.run_config["optim"],
            lr=self.run_config["lr"],
            bs=self.run_config["batch_size"],
        )

        # early stopping
        early_stopping = EarlyStopping(
            patience=self.run_config["early_stopping_patience"],
        )

        # define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.run_config["downstream_train_epochs"],
            eta_min=0,
        )

        # define metrics
        loss_metric_train = torchmetrics.MeanMetric().to(self.device)
        accuracy_train = torchmetrics.Accuracy(
            num_classes=classifier.fc.num_labels,
            top_k=1,
        ).to(self.device)
        f1_score_train = torchmetrics.F1Score(
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(self.device)
        auroc_train = torchmetrics.AUROC(
            num_classes=classifier.fc.num_labels,
        ).to(self.device)

        loss_metric_val = torchmetrics.MeanMetric().to(self.device)
        accuracy_val = torchmetrics.Accuracy(
            num_classes=classifier.fc.num_labels,
            top_k=1,
        ).to(self.device)
        f1_score_val = torchmetrics.F1Score(
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(self.device)
        auroc_val = torchmetrics.AUROC(
            num_classes=classifier.fc.num_labels,
        ).to(self.device)

        # metrics
        l_loss_train, l_f1_train, l_acc_train, l_auroc_train = [], [], [], []
        l_loss_val, l_f1_val, l_acc_val, l_auroc_val = [], [], [], []
        step = 0
        best_val_loss = np.inf
        best_model_wts = copy.deepcopy(classifier.state_dict())

        # start training
        for epoch in range(self.run_config["downstream_train_epochs"]):
            if fine_tune:
                if epoch >= self.run_config["warmup_epochs"]:
                    # make sure the classifier and backbone get trained
                    set_requires_grad(classifier, True)
                else:
                    # freeze the backbone and let only the classifier be trained
                    set_requires_grad(classifier, True)
                    set_requires_grad(classifier.backbone, False)

            # training
            classifier.train()
            train_loader.dataset.training = True
            for img, target in train_loader:
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if self.run_config["use_mixup"]:
                    # MixUp if wanted
                    inputs, targets_a, targets_b, lam = mixup_data(
                        img, target, device=self.device
                    )
                    inputs, targets_a, targets_b = map(
                        torch.autograd.Variable,
                        (inputs, targets_a, targets_b),
                    )
                    # run the forwardpass
                    pred = classifier(img)
                    # calculate loss and scores
                    loss = mixup_criterion(
                        criterion=criterion,
                        pred=pred,
                        y_a=targets_a,
                        y_b=targets_b,
                        lam=lam,
                    )
                else:
                    pred = classifier(img)
                    loss = criterion(pred, target)

                # backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step()

                # log to W&B
                log_dict = {
                    f"Downstream/{task_name}/train_loss": loss.item(),
                    f"Downstream/{task_name}/train_f1": f1_score_train(pred, target),
                    f"Downstream/{task_name}/train_accuracy": accuracy_train(
                        pred, target
                    ),
                    f"Downstream/{task_name}/train_auroc": auroc_train(pred, target),
                    f"Downstream/{task_name}/epoch": epoch,
                    f"Downstream/{task_name}/step": step,
                }
                wandb.log(log_dict)
                step += 1

                # add to overall metrics
                loss_metric_train.update(loss.detach())
                accuracy_train.update(pred, target)
                f1_score_train.update(pred, target)
                auroc_train.update(pred, target)
            l_loss_train.append(loss_metric_train.compute())
            l_f1_train.append(f1_score_train.compute())
            l_acc_train.append(accuracy_train.compute())
            l_auroc_train.append(auroc_train.compute())

            # Validation
            classifier.eval()
            val_loader.dataset.training = False
            for img, target in val_loader:
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # retreive the embedding
                with torch.no_grad():
                    pred = classifier(img)

                # calculate loss and scores
                loss = criterion(pred, target)
                # add to overall metrics
                loss_metric_val.update(loss.detach())
                accuracy_val.update(pred, target)
                f1_score_val.update(pred, target)
                auroc_val.update(pred, target)
            l_loss_val.append(loss_metric_val.compute())
            l_f1_val.append(f1_score_val.compute())
            l_acc_val.append(accuracy_val.compute())
            l_auroc_val.append(auroc_val.compute())

            # check if we have new best model
            if l_loss_val[-1] < best_val_loss:
                best_val_loss = l_loss_val[-1]
                best_model_wts = copy.deepcopy(classifier.state_dict())

            # check early stopping
            early_stopping(l_loss_val[-1])
            if early_stopping.early_stop:
                print("EarlyStopping, validation did not decrease.")
                break
            print(
                f"Epoch: {epoch}, "
                f"Train Loss: {l_loss_train[-1]:.4f}, "
                f"Train Acc: {l_acc_train[-1]:.4f}, "
                f"Train F1: {l_f1_train[-1]:.4f}, "
                f"Train AUROC: {l_auroc_train[-1]:.4f}, "
                f"Valid Loss: {l_loss_val[-1]:.4f}, "
                f"Valid Acc: {l_acc_val[-1]:.4f}, "
                f"Valid F1: {l_f1_val[-1]:.4f}, "
                f"Valid AUROC: {l_auroc_val[-1]:.4f}"
            )
            # log to W&B
            log_dict = {
                f"Downstream/{task_name}/valid_loss": l_loss_val[-1],
                f"Downstream/{task_name}/valid_f1": l_f1_val[-1],
                f"Downstream/{task_name}/valid_accuracy": l_acc_val[-1],
                f"Downstream/{task_name}/valid_auroc": l_auroc_val[-1],
                f"Downstream/{task_name}/epoch": epoch,
                f"Downstream/{task_name}/step": step,
            }
            wandb.log(log_dict)
        # get the best epoch in terms of F1 score
        best_epoch = torch.Tensor(l_f1_val).argmax()
        # log to W&B
        log_dict = {
            f"Downstream/{task_name}/best_val_epoch": best_epoch,
            f"Downstream/{task_name}/best_train_loss": l_loss_train[best_epoch],
            f"Downstream/{task_name}/best_train_acc_top_1": l_acc_train[best_epoch],
            f"Downstream/{task_name}/best_train_f1_score": l_f1_train[best_epoch],
            f"Downstream/{task_name}/best_val_loss": l_loss_val[best_epoch],
            f"Downstream/{task_name}/best_val_acc_top_1": l_acc_val[best_epoch],
            f"Downstream/{task_name}/best_val_f1_score": l_f1_val[best_epoch],
            f"Downstream/{task_name}/epoch": epoch,
            f"Downstream/{task_name}/step": step,
        }
        wandb.log(log_dict)

        # Save the best model to wandb and load it
        classifier.load_state_dict(best_model_wts)
        best_path = str(Path(wandb.run.dir) / "model_best.pth")
        torch.save(classifier.state_dict(), best_path)

        # Test
        if test_loader is not None:
            accuracy_test = torchmetrics.Accuracy(
                num_classes=classifier.fc.num_labels, top_k=1
            ).to(self.device)
            f1_score_test = torchmetrics.F1Score(
                num_classes=classifier.fc.num_labels, average="macro"
            ).to(self.device)
            precision_test = torchmetrics.Precision(
                num_classes=classifier.fc.num_labels, average="macro"
            ).to(self.device)
            recall_test = torchmetrics.Recall(
                num_classes=classifier.fc.num_labels, average="macro"
            ).to(self.device)
            auroc_test = torchmetrics.AUROC(
                num_classes=classifier.fc.num_labels,
            ).to(self.device)

            classifier.eval()
            for img, target in test_loader:
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # retreive the embedding
                with torch.no_grad():
                    pred = classifier(img)

                # add to overall metrics
                accuracy_test.update(pred, target)
                f1_score_test.update(pred, target)
                precision_test.update(pred, target)
                recall_test.update(pred, target)
                auroc_test.update(pred, target)
            # log to W&B
            log_dict = {
                f"Downstream/{task_name}/test_f1": f1_score_test.compute(),
                f"Downstream/{task_name}/test_accuracy": accuracy_test.compute(),
                f"Downstream/{task_name}/test_precision": precision_test.compute(),
                f"Downstream/{task_name}/test_recall": recall_test.compute(),
                f"Downstream/{task_name}/test_auroc": auroc_test.compute(),
                f"Downstream/{task_name}/epoch": epoch,
                f"Downstream/{task_name}/step": step,
            }
            wandb.log(log_dict)

    def eval_segmentation_task(
        self,
        train_loader,
        val_loader,
        decoder: torch.nn.Module,
        task_name: str,
        test_loader=None,
        fine_tune: bool = False,
    ):
        if fine_tune:
            # make sure the decoder can get trained
            set_requires_grad(decoder, True)
        else:
            # freeze the backbone and let only the decoder be trained
            set_requires_grad(decoder, True)
            set_requires_grad(decoder.encoder, False)

        try:
            in_size = (1, 3, self.run_config["img_size"], self.run_config["img_size"])
            summary(decoder, input_size=in_size)
        except RuntimeError:
            print(f"Summary can not be displayed for a Huggingface model.")
            print(
                f"Number of parameters backbone: {decoder.encoder.model.num_parameters():,}"
            )
        wandb.watch(decoder, log="all")
        # loss function, optimizer, scores
        n_classes = train_loader.dataset.n_classes
        mode = "multiclass" if n_classes > 2 else "binary"
        criterion = get_segmentation_loss(
            loss_fn_name=self.run_config["loss_fn"],
            mode=mode,
        )
        criterion = criterion.to(self.device)
        optimizer = get_lin_scaled_optimizer(
            model=decoder,
            optimizer_name=self.run_config["optim"],
            lr=self.run_config["lr"],
            bs=self.run_config["batch_size"],
        )
        # define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.run_config["downstream_train_epochs"],
            eta_min=0,
        )

        # early stopping
        early_stopping = EarlyStopping(
            patience=self.run_config["early_stopping_patience"],
        )

        # define metrics
        loss_metric_train = torchmetrics.MeanMetric()
        loss_metric_train = loss_metric_train.to(self.device)
        iou_metric_train = torchmetrics.MeanMetric()
        iou_metric_train = iou_metric_train.to(self.device)
        f1_metric_train = torchmetrics.MeanMetric()
        f1_metric_train = f1_metric_train.to(self.device)

        loss_metric_val = torchmetrics.MeanMetric()
        loss_metric_val = loss_metric_val.to(self.device)
        iou_metric_val = torchmetrics.MeanMetric()
        iou_metric_val = iou_metric_val.to(self.device)
        f1_metric_val = torchmetrics.MeanMetric()
        f1_metric_val = f1_metric_val.to(self.device)

        # accumulated metrics
        l_loss_train = []
        l_iou_train = []
        l_f1_train = []
        l_loss_val = []
        l_iou_val = []
        l_f1_val = []
        step = 0
        best_val_loss = np.inf
        best_model_wts = copy.deepcopy(decoder.state_dict())

        # start training
        for epoch in range(self.run_config["downstream_train_epochs"]):
            if fine_tune:
                if epoch >= self.run_config["warmup_epochs"]:
                    # make sure the decoder can get trained
                    set_requires_grad(decoder, True)
                else:
                    # freeze the backbone and let only the decoder be trained
                    set_requires_grad(decoder, True)
                    set_requires_grad(decoder.encoder, False)
            # training
            decoder.train()
            train_loader.dataset.training = True
            for img, target in train_loader:
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # run the forwardpass
                mask = decoder(img)

                # calculate loss and scores
                loss = criterion(mask, target)
                iou_score, f1_score = get_seg_metrics(
                    mask=mask,
                    target=target,
                    n_classes=n_classes,
                    mode=mode,
                )

                # backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step()

                # log to W&B
                log_dict = {
                    f"Downstream/{task_name}/train_loss": loss.item(),
                    f"Downstream/{task_name}/train_iou": iou_score,
                    f"Downstream/{task_name}/train_f1": f1_score,
                    f"Downstream/{task_name}/epoch": epoch,
                    f"Downstream/{task_name}/step": step,
                }
                wandb.log(log_dict)
                step += 1

                # add to overall metrics
                loss_metric_train.update(loss.detach())
                iou_metric_train.update(iou_score)
                f1_metric_train.update(f1_score)
            l_loss_train.append(loss_metric_train.compute())
            l_iou_train.append(iou_metric_train.compute())
            l_f1_train.append(f1_metric_train.compute())

            # Validation
            decoder.eval()
            val_loader.dataset.training = False
            for i, (img, target) in enumerate(val_loader):
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # retreive the embedding
                with torch.no_grad():
                    mask = decoder(img)

                # log prediction
                if i == 0 and epoch % 50 == 0:
                    log_segmentation_pred(img, mask, target, mode=mode)

                # calculate loss and scores
                loss = criterion(mask, target)
                iou_score, f1_score = get_seg_metrics(
                    mask=mask,
                    target=target,
                    n_classes=n_classes,
                    mode=mode,
                )
                # add to overall metrics
                loss_metric_val.update(loss.detach())
                iou_metric_val.update(iou_score)
                f1_metric_val.update(f1_score)
            l_loss_val.append(loss_metric_val.compute())
            l_iou_val.append(iou_metric_val.compute())
            l_f1_val.append(f1_metric_val.compute())

            # check if we have new best model
            if l_loss_val[-1] < best_val_loss:
                best_val_loss = l_loss_val[-1]
                best_model_wts = copy.deepcopy(decoder.state_dict())

            # check early stopping
            early_stopping(l_loss_val[-1])
            if early_stopping.early_stop:
                print("EarlyStopping, validation did not decrease.")
                break
            print(
                f"Epoch: {epoch}, "
                f"Train Loss: {l_loss_train[-1]}, "
                f"Train IOU: {l_iou_train[-1]}, "
                f"Train F1: {l_f1_train[-1]}, "
                f"Valid Loss: {l_loss_val[-1]}, "
                f"Valid IOU: {l_iou_val[-1]}, "
                f"Valid F1: {l_f1_val[-1]}"
            )
            # log to W&B
            log_dict = {
                f"Downstream/{task_name}/valid_loss": l_loss_val[-1],
                f"Downstream/{task_name}/valid_iou": l_iou_val[-1],
                f"Downstream/{task_name}/valid_f1": l_f1_val[-1],
                f"Downstream/{task_name}/epoch": epoch,
                f"Downstream/{task_name}/step": step,
            }
            wandb.log(log_dict)

        # get the best epoch in terms of IOU score
        best_epoch = torch.Tensor(l_iou_val).argmax()
        # log to W&B
        log_dict = {
            f"Downstream/{task_name}/best_val_epoch": best_epoch,
            f"Downstream/{task_name}/best_train_loss": l_loss_train[best_epoch],
            f"Downstream/{task_name}/best_train_iou_score": l_iou_train[best_epoch],
            f"Downstream/{task_name}/best_train_f1_score": l_f1_train[best_epoch],
            f"Downstream/{task_name}/best_val_loss": l_loss_val[best_epoch],
            f"Downstream/{task_name}/best_val_iou_score": l_iou_val[best_epoch],
            f"Downstream/{task_name}/best_val_f1_score": l_f1_val[best_epoch],
            f"Downstream/{task_name}/epoch": epoch,
            f"Downstream/{task_name}/step": step,
        }
        wandb.log(log_dict)

        # Save the best model to wandb and load it
        decoder.load_state_dict(best_model_wts)
        best_path = str(Path(wandb.run.dir) / "model_best.pth")
        torch.save(decoder.state_dict(), best_path)

        # Test
        if test_loader is not None:
            iou_test = torchmetrics.MeanMetric()
            iou_test = iou_test.to(self.device)
            f1_test = torchmetrics.MeanMetric()
            f1_test = f1_test.to(self.device)

            decoder.eval()
            test_loader.dataset.training = False
            for img, target in test_loader:
                # move batch to device
                img = img.to(self.device)
                target = target.to(self.device)

                # retreive the embedding
                with torch.no_grad():
                    mask = decoder(img)

                # add to overall metrics
                iou_score, f1_score = get_seg_metrics(
                    mask=mask,
                    target=target,
                    n_classes=n_classes,
                    mode=mode,
                )
                iou_test.update(iou_score)
                f1_test.update(f1_score)
            # log to W&B
            log_dict = {
                f"Downstream/{task_name}/test_iou": iou_test.compute(),
                f"Downstream/{task_name}/test_f1": f1_test.compute(),
                f"Downstream/{task_name}/epoch": epoch,
                f"Downstream/{task_name}/step": step,
            }
            wandb.log(log_dict)

    def hyperopt_downstream(self, dataset: str):
        dataset_dict = {
            "fitzpatrick17k": self.eval_fitzpatrick17k,
            "pad_ufes_20": self.eval_pad_ufes_20,
            "ham10000": self.eval_ham10000,
            "body_loc": self.eval_body_loc,
            "isic_task1": self.eval_isic_task1,
            "ppp": self.eval_ppp,
        }
        dataset_func = dataset_dict.get(dataset, None)
        if dataset_func is None:
            raise ValueError("Unknown dataset.")
        # Initialize a new wandb run
        with wandb.init(config=self.config, group="downstream"):
            # update the name of the run
            run_name = f"{self.arch_name}-{wandb.run.name}"
            wandb.run.name = run_name
            wandb.run.save()

            # get the config for the run
            self.run_config = wandb.config

            # seed everything
            fix_random_seeds(self.run_config["seed"])

            # load the downstream tasks and classifiers
            self.load_downstream_tasks(
                dataset,
                ssl_model=self.run_config["SSL_model"].lower(),
                fine_tune=self.run_config["fine_tune"],
            )
            # evaluate the model
            dataset_func(fine_tune=self.run_config["fine_tune"])
            # finish the run
            wandb.finish()

    def eval_fitzpatrick17k(self, fine_tune: bool = False):
        self.eval_classification_task(
            train_loader=self.fitz_train,
            val_loader=self.fitz_val,
            test_loader=self.fitz_test,
            classifier=self.fitz_clf,
            task_name="fitzpatrick17k",
            fine_tune=fine_tune,
        )

    def eval_pad_ufes_20(self, fine_tune: bool = False):
        self.eval_classification_task(
            train_loader=self.pu_train,
            val_loader=self.pu_val,
            test_loader=self.pu_test,
            classifier=self.pu_clf,
            task_name="pad_ufes_20",
            fine_tune=fine_tune,
        )

    def eval_ham10000(self, fine_tune: bool = False):
        self.eval_classification_task(
            train_loader=self.ham_train,
            val_loader=self.ham_val,
            test_loader=self.ham_test,
            classifier=self.ham_clf,
            task_name="ham10000",
            fine_tune=fine_tune,
        )

    def eval_body_loc(self, fine_tune: bool = False):
        self.eval_classification_task(
            train_loader=self.body_loc_train,
            val_loader=self.body_loc_val,
            test_loader=self.body_loc_test,
            classifier=self.body_loc_clf,
            task_name="body_loc",
            fine_tune=fine_tune,
        )

    def eval_isic_task1(self, fine_tune: bool = False):
        self.eval_segmentation_task(
            train_loader=self.isic_train,
            val_loader=self.isic_val,
            decoder=self.isic_dec,
            task_name="isic_task1",
            fine_tune=fine_tune,
        )

    def eval_ppp(self, fine_tune: bool = False):
        self.eval_segmentation_task(
            train_loader=self.ppp_train,
            val_loader=self.ppp_val,
            test_loader=self.ppp_test,
            decoder=self.ppp_dec,
            task_name="ppp",
            fine_tune=fine_tune,
        )

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device

    def _save_config_file(self, model_checkpoints_folder: Path):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy(self.config_path, model_checkpoints_folder / "config.yaml")

    def _log_embeddings(
        self,
        model,
        n_iter: int,
        n_items: Union[int, None] = 3_000,
        log_self_attention: bool = False,
        return_embedding: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        # evaluate model on val set
        if is_main_process():
            model.eval()
            with torch.no_grad():
                imgs = []
                lbls = []
                embeddings = []
                entropy = []
                for i, (img, lbl) in enumerate(self.val_dataset):
                    img = img.to(self.device)
                    # get the embeddings
                    emb = self._get_embedding(model, img)
                    ent_emb = calculate_embedding_entropy(emb)
                    # visualize self attention if requested
                    if i == 0 and log_self_attention:
                        visualize_self_attention(
                            model=model,
                            images=img,
                            n_iter=n_iter,
                            patch_size=self.config["model"]["student"]["patch_size"],
                            multi_gpu=self.multi_gpu,
                            wandb_cat="self-attention",
                            imgs_to_visualize=self.config["imgs_to_visualize"],
                            remove_cls_token=(
                                False
                                if "swin" in self.config["model"]["base_model"]
                                else True
                            ),
                            adapt_patch_size=(
                                True
                                if "swin" in self.config["model"]["base_model"]
                                else False
                            ),
                        )
                    # add info to lists
                    embeddings.append(emb.cpu())
                    imgs.append(img.cpu())
                    lbls.append(lbl.cpu())
                    entropy.append(ent_emb)

            # create (concat) our embedding space
            embeddings = torch.concat(embeddings, dim=0)
            imgs = torch.concat(imgs, dim=0).cpu()
            lbls = torch.concat(lbls, dim=0).cpu()

            # entropy embedding space
            ent_avg = torch.mean(torch.Tensor(entropy)[:, 0])
            ent_min = torch.mean(torch.Tensor(entropy)[:, 1])
            ent_max = torch.mean(torch.Tensor(entropy)[:, 2])
            ent_std = torch.mean(torch.Tensor(entropy)[:, 3])
            ent_med = torch.mean(torch.Tensor(entropy)[:, 4])

            # nearest neighbors
            visualize_nearest_neighbors(
                embeddings=embeddings,
                imgs=imgs,
                n_iter=n_iter,
                imgs_to_visualize=self.config["imgs_to_visualize"],
            )

            # select only N items (otherwise the embedding logging is to slow)
            if n_items is not None:
                embeddings = embeddings[:n_items]
                imgs = imgs[:n_items]
                lbls = lbls[:n_items]

            if return_embedding:
                return embeddings, imgs, lbls

            # log the embeddings to wandb
            imgs = [wandb.Image(x) for x in imgs]
            df_emb = pd.DataFrame(embeddings.tolist())
            emb_cols = [f"dim_{x+1}" for x in range(embeddings[0].size()[0])]
            df_emb.columns = emb_cols
            df_emb["lbls"] = lbls.tolist()
            df_emb["image"] = imgs
            cols = df_emb.columns.tolist()
            df_emb = df_emb[cols[-1:] + cols[:-1]]
            wandb.log(
                {
                    "embeddings": df_emb,
                    "entropy/val_ent_avg": ent_avg,
                    "entropy/val_ent_min": ent_min,
                    "entropy/val_ent_max": ent_max,
                    "entropy/val_ent_std": ent_std,
                    "entropy/val_ent_med": ent_med,
                },
                step=n_iter,
            )

    def update_optim_from_schedulers(
        self,
        optimizer,
        lr_schedule,
        wd_schedule,
        n_iter: int,
    ):
        # update weight decay and LR according to their schedule
        # but only if wanted
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None and self.config["use_lr_scheduler"]:
                param_group["lr"] = lr_schedule[n_iter]
            if i == 0:  # only the first group is regularized
                if wd_schedule is not None and self.config["use_wd_scheduler"]:
                    param_group["weight_decay"] = wd_schedule[n_iter]

    def check_loss_nan(self, loss):
        # check if loss is not infinite
        if not math.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            wandb.alert(title="Loss NaN", text=f"Loss is {loss}, stopping training.")
            sys.exit(1)
