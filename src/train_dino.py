import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import ImageFile
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from ssl_library.src.augmentations.multi_crop import DINODataAugmentation
from ssl_library.src.datasets.generic_image_dataset import GenericImageDataset
from ssl_library.src.trainers.dino_trainer import DINOTrainer
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.utils import cleanup, fix_random_seeds, init_distributed_mode

ImageFile.LOAD_TRUNCATED_IMAGES = True

my_parser = argparse.ArgumentParser(description="Trains DINO on a given dataset.")
my_parser.add_argument(
    "--config_path", type=str, required=True, help="Path to the config yaml for DINO."
)
args = my_parser.parse_args()

if __name__ == "__main__":
    # load config yaml
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    config = yaml.load(open(args.config_path, "r"), Loader=Loader)

    # initialize distribution
    init_distributed_mode()
    # seed everything
    fix_random_seeds(config["seed"])

    # load the train dataset
    train_path = config["dataset"]["train_path"]
    dino_transform = DINODataAugmentation(**config["dataset"]["augmentations"])
    train_dataset = GenericImageDataset(
        dataset_dir=train_path,
        transform=dino_transform,
        image_extensions=["*.png", "*.webp", "*.jpg", "*.jpeg", "*.bmp"],
    )
    if "train_df" in config["dataset"].keys():
        df = pd.read_csv(config["dataset"]["train_df"])
        df = df[df["dataset_type"] != "private_resized"]
        df.reset_index(drop=True, inplace=True)
        _df = train_dataset.meta_data.merge(
            df[["img_path"]], on="img_path", how="inner"
        )
        train_dataset.meta_data = _df
        print(
            f"Setting dataframe of the training dataset to: {config['dataset']['train_df']}"
        )
    print(train_dataset.meta_data)
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataset = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config["batch_size"],
        **config["dataset"]["loader"],
    )

    # load the val dataset
    val_path = config["dataset"]["val_path"]
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    val_dataset = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        **config["dataset"]["val_loader"],
    )

    # initialize the trainer
    trainer = DINOTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        config_path=args.config_path,
        additional_run_info=(
            f"{Path(train_path[0]).stem}"
            if type(train_path) is list
            else f"{Path(train_path).stem}"
        ),
        print_model_summary=True,
    )
    # train
    trainer.fit()
    # cleanup distributed training
    cleanup()
