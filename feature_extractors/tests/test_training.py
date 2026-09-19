"""Guard against feature misalignment and split/normalization leakage."""

from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np
import torch

from model.collaborative_gate import CollaborativeGateClassifier
from data_extract.base import split_audio
from data_extract.dataset import load_split
from trainer.data import load_data, normalize_from_train


class TrainingDataTests(unittest.TestCase):
    def write_archive(self, root, extractor, split, keys, labels, features):
        path = root / extractor / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.asarray(keys), labels=np.asarray(labels), features=np.asarray(features))

    def make_archives(self, root):
        for split, keys, labels in (
            ("train", ["x", "y"], [0, 1]),
            ("valid", ["z", "w"], [1, 0]),
            ("test", ["q", "r"], [0, 1]),
        ):
            values = [[1., 10.], [3., 30.]] if split == "train" else [[100., 100.], [200., 200.]]
            self.write_archive(root, "bert", split, keys, labels, values)
            self.write_archive(root, "wav2vec2", split, keys[::-1], labels[::-1], values[::-1])

    def test_alignment_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archives(root)
            splits = load_data(root, "base", ("T", "A"))
            np.testing.assert_array_equal(splits["train"].features["T"], splits["train"].features["A"])
            stats = normalize_from_train(splits)
            np.testing.assert_allclose(stats["T"]["mean"], [2., 20.])
            np.testing.assert_allclose(splits["train"].features["T"].mean(0), [0., 0.])
            self.assertGreater(splits["valid"].features["T"][0, 0], 10.)

    def test_label_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_archives(root)
            self.write_archive(root, "wav2vec2", "valid", ["w", "z"], [1, 1], [[2., 2.], [1., 1.]])
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                load_data(root, "base", ("T", "A"))

    def test_all_ablation_forward_shapes_with_single_item(self):
        for name in ("T", "A", "V", "T+A", "T+V", "A+V", "T+A+V"):
            modalities = name.split("+")
            model = CollaborativeGateClassifier({m: 4 for m in modalities}, shared_dim=8, projection_dim=4)
            output = model({m: torch.ones(1, 4) for m in modalities})
            self.assertEqual(tuple(output.shape), (1, 2))

    def test_mcsd1_schema_labels_and_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("audios", "videos"):
                (root / name).mkdir()
            (root / "audios/123.wav").touch()
            (root / "videos/123.mp4").touch()
            with (root / "train.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("File Name", "Transcriptions", "Labels"))
                writer.writerow(("123", "这句话是反讽。", "s"))
            for modality in ("T", "A", "V"):
                sample = load_split(root, "train", modality, "mcsd1")[0]
                self.assertEqual((sample.key, sample.text, sample.label), ("123", "这句话是反讽。", 1))
            self.assertEqual(sample.video, root / "videos/123.mp4")

    def test_audio_chunks_cover_full_long_utterance(self):
        waveform = np.arange(590, dtype=np.float32)
        chunks = split_audio(waveform, rate=10, max_seconds=20)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        np.testing.assert_array_equal(np.concatenate(chunks), waveform)


if __name__ == "__main__":
    unittest.main()
