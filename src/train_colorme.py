import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

from ssl_library.src.dataset_wrappers.colorme_wrapper import ColorMeDatasetWrapper
from ssl_library.src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ssl_library.src.trainers.colorme_trainer import ColorMeTrainer
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.utils import cleanup, fix_random_seeds, init_distributed_mode

my_parser = argparse.ArgumentParser(description="Trains ColorMe on a given dataset.")
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml for ColorMe.",
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
    train_dataset = EncryptedImageDataset(
        train_path, enc_keys=config["decryption"]["keys"], transform=None
    )
    train_dataset = ColorMeDatasetWrapper(train_dataset, **config["dataset"]["wrapper"])
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataset = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config["batch_size"],
        **config["dataset"]["loader"],
    )

    # load the val dataset
    val_path = config["dataset"]["val_path"]
    val_dataset = datasets.ImageFolder(val_path, transform=None)
    val_dataset = ColorMeDatasetWrapper(val_dataset, **config["dataset"]["wrapper"])
    val_dataset = DataLoader(
        val_dataset, batch_size=config["batch_size"], **config["dataset"]["val_loader"]
    )

    # initialize the trainer
    trainer = ColorMeTrainer(train_dataset, val_dataset, config, args.config_path)
    # train
    trainer.fit()

    # cleanup distributed training
    cleanup()
