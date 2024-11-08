import os

os.environ["WANDB_MODE"] = "offline"

import copy
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from ssl_library.src.augmentations.multi_crop import iBOTDataAugmentation
from ssl_library.src.dataset_wrappers.ibot_wrapper import ImageFolderMask
from ssl_library.src.trainers.ibot_trainer import iBOTTrainer
from ssl_library.src.utils.loader import Loader
from ssl_library.src.utils.utils import (
    cleanup,
    compare_models,
    fix_random_seeds,
    init_distributed_mode,
)


class TestIBOTTraining(unittest.TestCase):
    def setUp(self):
        # set the paths
        self.config_path = Path("tests/test_utils/configs") / "ibot.yaml"
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
        transform = iBOTDataAugmentation(**self.config["dataset"]["augmentations"])
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

        train_dataset = ImageFolderMask(
            val_path,
            enc_keys=self.config["decryption"]["keys"],
            transform=transform,
            patch_size=self.config["model"]["student"]["patch_size"],
            **self.config["dataset"]["MIM"]
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
        trainer = iBOTTrainer(
            train_dataset, val_dataset, self.config, self.config_path, debug=True
        )
        start_model = copy.deepcopy(trainer.student).to(trainer.device)
        trainer.fit()

        # compare models (check that the models are not equal)
        end_model = trainer.student.to(trainer.device)
        n_differs = compare_models(start_model, end_model)
        self.assertTrue(n_differs > 0)

        # set the MIM to false so that no masks are required
        start_model.masked_im_modeling = False
        end_model.backbone.masked_im_modeling = False

        # check if the output is different between the models
        img_shape = (1, 3, 224, 224)
        rand_img = torch.rand(img_shape).to(trainer.device)
        out_start = start_model(rand_img)
        out_end = end_model(rand_img)[0]
        self.assertFalse(torch.equal(out_start, out_end))


if __name__ == "__main__":
    unittest.main()
