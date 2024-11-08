import unittest
from pathlib import Path

import numpy as np
import torch

from ssl_library.src.datasets.downstream_tasks.isic_2018_task_1_dataset import (
    ISICTask1Dataset,
)


class TestISICTask1Dataset(unittest.TestCase):
    def setUp(self):
        self.data_path = Path("data/ISIC_Task1/")
        self.data_type = "train"

    def test_iterator(self):
        dataset = ISICTask1Dataset(self.data_path, dataset_type=self.data_type)
        for sample in dataset:
            self.assertEqual(type(sample), tuple)
            self.assertEqual(len(sample), 2)
            self.assertEqual(type(sample[0]), torch.Tensor)
            self.assertEqual(type(sample[1]), torch.Tensor)
            self.assertIsNotNone(sample[0])
            self.assertIsNotNone(sample[1])
            img = np.asarray(sample[0])
            mask = np.asarray(sample[1])
            self.assertEqual(len(img.shape), 3)
            self.assertEqual(img.shape[0], 3)
            self.assertEqual(len(mask.shape), 3)
            break

    def test_iterator_transform(self):
        dataset = ISICTask1Dataset(
            self.data_path,
            dataset_type=self.data_type,
        )
        for image, mask in dataset:
            self.assertIsNotNone(image)
            self.assertIsNotNone(mask)
            self.assertEqual((3, 224, 224), image.shape)
            self.assertEqual((1, 224, 224), mask.shape)
            break


if __name__ == "__main__":
    unittest.main()
