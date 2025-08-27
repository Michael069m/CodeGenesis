import unittest
from src.dataset import YourDatasetClass  # Replace with the actual class name

class TestDataset(unittest.TestCase):

    def setUp(self):
        # Initialize your dataset class here
        self.dataset = YourDatasetClass('path/to/dataset')  # Update with the actual path

    def test_load_images(self):
        images = self.dataset.load_images()
        self.assertEqual(len(images), 10000)  # Check if 10,000 images are loaded

    def test_load_labels(self):
        labels = self.dataset.load_labels()
        self.assertEqual(len(labels), 10000)  # Check if 10,000 labels are loaded

    def test_image_preprocessing(self):
        processed_image = self.dataset.preprocess_image('path/to/sample/image.jpg')  # Update with a sample image path
        self.assertIsNotNone(processed_image)  # Ensure the image is processed correctly

    def test_label_format(self):
        label = self.dataset.load_labels()[0]  # Load a sample label
        self.assertIn('class', label)  # Check if the label contains the expected keys

if __name__ == '__main__':
    unittest.main()