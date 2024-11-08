import os

os.environ["WANDB_MODE"] = "offline"

import copy
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

from ssl_library.src.dataset_wrappers.colorme_wrapper import ColorMeDatasetWrapper
from ssl_library.src.datasets.encrypted_image_dataset import EncryptedImageDataset
from ssl_library.src.trainers.colorme_trainer import ColorMeTrainer
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.utils import (
    cleanup,
    compare_models,
    fix_random_seeds,
    init_distributed_mode,
)


class TestColorMeTraining(unittest.TestCase):
    def setUp(self):
        # set the paths
        self.config_path = Path("tests/test_utils/configs") / "colorme.yaml"
        # load config yaml
        self.config = yaml.load(open(self.config_path, "r"), Loader=Loader)
        # initialize distribution
        init_distributed_mode()
        # seed everything
        fix_random_seeds(self.config["seed"])

    def tearDown(self):
        cleanup()

    def test_colorme_training(self):
        # load the datasets (train and val are the same for testing)
        val_path = self.config["dataset"]["val_path"]
        train_dataset = EncryptedImageDataset(
            val_path, enc_keys=self.config["decryption"]["keys"], transform=None
        )
        train_dataset = ColorMeDatasetWrapper(
            train_dataset, **self.config["dataset"]["wrapper"]
        )
        sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataset = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.config["batch_size"],
            **self.config["dataset"]["val_loader"]
        )

        val_dataset = datasets.ImageFolder(val_path, transform=None)
        val_dataset = ColorMeDatasetWrapper(
            val_dataset, **self.config["dataset"]["wrapper"]
        )
        val_dataset = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            **self.config["dataset"]["val_loader"]
        )

        # initialize the trainer and train the model
        trainer = ColorMeTrainer(
            train_dataset, val_dataset, self.config, self.config_path, debug=True
        )
        start_model = copy.deepcopy(trainer.model)
        trainer.fit()

        # compare models (check that the models are not equal)
        end_model = copy.deepcopy(trainer.model)
        n_differs = compare_models(start_model, end_model)
        self.assertTrue(n_differs > 0)

        # check if the output is different between the models
        img_shape = eval(self.config["dataset"]["wrapper"]["target_shape"])
        img_shape = (1, 1, img_shape[0], img_shape[1])
        rand_img = torch.rand(img_shape).to(trainer.device)
        out_start = start_model(rand_img)
        out_end = end_model(rand_img)
        self.assertFalse(torch.equal(out_start[0], out_end[0]))
        self.assertFalse(torch.equal(out_start[1], out_end[1]))

        # check the ranges of the output
        self.assertAlmostEqual(out_start[1].sum().item(), 1.0)
        self.assertAlmostEqual(out_end[1].sum().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
