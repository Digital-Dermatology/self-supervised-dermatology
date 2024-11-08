import unittest
from pathlib import Path

from PIL import Image
from torchvision import transforms

from ssl_library.src.datasets.downstream_tasks.pad_ufes_20_dataset import (
    PADUFES20Dataset,
)


class TestPADUFES20Dataset(unittest.TestCase):
    def setUp(self):
        self.data_path = Path("data/PAD-UFES-20/")
        self.csv_file = self.data_path / "metadata.csv"
        self.root_dir = self.data_path / "images"

    def test_iterator(self):
        dataset = PADUFES20Dataset(self.csv_file, self.root_dir)
        for sample in dataset:
            self.assertEqual(type(sample), tuple)
            self.assertEqual(len(sample), 2)
            self.assertEqual(type(sample[0]), Image.Image)
            self.assertEqual(type(sample[1]), int)
            self.assertIsNotNone(sample[0])
            self.assertIsNotNone(sample[1])
            break

    def test_iterator_transform(self):
        transform = transforms.Resize((10, 10))
        dataset = PADUFES20Dataset(self.csv_file, self.root_dir, transform=transform)
        for image, diagnosis in dataset:
            self.assertIsNotNone(image)
            self.assertIsNotNone(diagnosis)
            self.assertEqual((10, 10), image.size)
            break


if __name__ == "__main__":
    unittest.main()
