import unittest
from pathlib import Path

from PIL import Image
from torchvision import transforms

from ssl_library.src.datasets.downstream_tasks.fitzpatrick17_dataset import (
    Fitzpatrick17kDataset,
)


class TestFitzpatrick17Dataset(unittest.TestCase):
    def setUp(self):
        self.data_path = Path("data/fitzpatrick17k/")
        self.csv_file = self.data_path / "fitzpatrick17k.csv"

    def test_iterator(self):
        dataset = Fitzpatrick17kDataset(self.csv_file, self.data_path)
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
        dataset = Fitzpatrick17kDataset(
            self.csv_file,
            self.data_path,
            transform=transform,
        )
        for image, diagnosis in dataset:
            self.assertIsNotNone(image)
            self.assertIsNotNone(diagnosis)
            self.assertEqual((10, 10), image.size)
            break


if __name__ == "__main__":
    unittest.main()
