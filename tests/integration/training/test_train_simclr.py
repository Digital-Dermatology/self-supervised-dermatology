import os

os.environ["WANDB_MODE"] = "offline"

import copy
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from ssl_library.src.augmentations.simclr import SimCLRDataAugmentation
from ssl_library.src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ssl_library.src.trainers.simclr_trainer import SimCLRTrainer
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.utils import (
    cleanup,
    compare_models,
    fix_random_seeds,
    init_distributed_mode,
)


class TestSimCLRTraining(unittest.TestCase):
    def setUp(self):
        # set the paths
        self.config_path = Path("tests/test_utils/configs") / "simclr.yaml"
        # load config yaml
        self.config = yaml.load(open(self.config_path, "r"), Loader=Loader)
        # initialize distribution
        init_distributed_mode()
        # seed everything
        fix_random_seeds(self.config["seed"])

    def tearDown(self):
        cleanup()

    def test_training(self):
        # load the datasets (train and val are the same for testing)
        val_path = self.config["dataset"]["val_path"]
        transform = SimCLRDataAugmentation(**self.config["dataset"]["augmentations"])
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        train_dataset = EncryptedImageDataset(
            val_path, enc_keys=self.config["decryption"]["keys"], transform=transform
        )
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataset = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.config["batch_size"],
            **self.config["dataset"]["val_loader"]
        )
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        val_dataset = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            **self.config["dataset"]["val_loader"]
        )

        # initialize the trainer and train the model
        trainer = SimCLRTrainer(
            train_dataset, val_dataset, self.config, self.config_path, debug=True
        )
        start_model = copy.deepcopy(trainer.model)
        trainer.fit()

        # compare models (check that the models are not equal)
        end_model = copy.deepcopy(trainer.model)
        n_differs = compare_models(start_model, end_model)
        self.assertTrue(n_differs > 0)

        # check if the output is different between the models
        img_size = self.config["dataset"]["augmentations"]["target_size"]
        img_size = (1, 3, img_size, img_size)
        rand_img = torch.rand(img_size).to(trainer.device)
        out_start = start_model(rand_img)
        out_end = end_model(rand_img)
        self.assertFalse(torch.equal(out_start[0], out_end[0]))
        self.assertFalse(torch.equal(out_start[1], out_end[1]))


if __name__ == "__main__":
    unittest.main()
