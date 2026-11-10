"""
Unit tests for FairFake Feature 2.4 (Demographic Dataset & Data Augmentation)
"""

import os
import tempfile
import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from backend.deepfake_detector.src.fair_dataset import (
    AttributeAnnotatedDataset,
    fair_train_transform,
    fair_eval_transform,
    fair_collate_fn,
    create_demographic_sampler,
    TARGET_ATTRIBUTES
)


class TestFairDataset(unittest.TestCase):
    def test_synthetic_dataset_shapes(self):
        ds = AttributeAnnotatedDataset(synth_samples=16, transform=fair_eval_transform)
        self.assertEqual(len(ds), 16)

        item = ds[0]
        self.assertEqual(item["image"].shape, (3, 224, 224))
        self.assertIsInstance(item["label"], torch.Tensor)
        self.assertIn("attributes", item)

        for attr in TARGET_ATTRIBUTES:
            self.assertIn(attr, item["attributes"])
            self.assertIsInstance(item["attributes"][attr], torch.Tensor)

    def test_collate_function(self):
        ds = AttributeAnnotatedDataset(synth_samples=8, transform=fair_train_transform)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=fair_collate_fn)

        batch = next(iter(loader))
        self.assertEqual(batch["images"].shape, (4, 3, 224, 224))
        self.assertEqual(batch["labels"].shape, (4,))
        self.assertIn("attributes", batch)

        for attr in TARGET_ATTRIBUTES:
            self.assertEqual(batch["attributes"][attr].shape, (4,))

    def test_demographic_sampler(self):
        ds = AttributeAnnotatedDataset(synth_samples=20, transform=fair_eval_transform)
        sampler = create_demographic_sampler(ds, primary_attribute="Dark_Skin")

        self.assertEqual(len(sampler.weights), 20)
        self.assertTrue(all(w > 0 for w in sampler.weights))

    def test_csv_loading(self):
        # Create small temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({
                "path": ["face_1.jpg", "face_2.jpg", "face_3.jpg", "face_4.jpg"],
                "true_label": ["REAL", "FAKE", "REAL", "FAKE"],
                "Dark_Skin": [1, 0, 1, 0],
                "Heavy_Makeup": [0, 1, 0, 1],
                "Eyeglasses": [0, 0, 1, 1],
                "Senior": [1, 0, 0, 0]
            })
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            ds = AttributeAnnotatedDataset(csv_path=csv_path, transform=fair_eval_transform)
            self.assertEqual(len(ds), 4)
            sample = ds[0]
            self.assertEqual(sample["label"].item(), 0.0) # REAL = 0.0
            self.assertEqual(sample["attributes"]["Dark_Skin"].item(), 1.0)
            self.assertEqual(sample["attributes"]["Senior"].item(), 1.0)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


if __name__ == "__main__":
    unittest.main()
