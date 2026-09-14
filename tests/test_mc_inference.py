"""MC inference must preserve identity and each module's original mode."""

import unittest

import numpy as np
import torch
from src.deep import predict_with_mc_dropout
from torch import nn


class McInferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Linear(2, 1))
        self.model.train()
        self.model[2].eval()
        self.batch = (
            torch.ones(2, 2),
            torch.tensor([0.0, 1.0]),
            ["a", "b"],
            ["a.jpg", "b.jpg"],
        )
        self.modes = [m.training for m in self.model.modules()]
        self.buffers = {k: v.clone() for k, v in self.model.named_buffers()}

    def assert_restored(self):
        self.assertEqual([m.training for m in self.model.modules()], self.modes)
        for name, value in self.model.named_buffers():
            torch.testing.assert_close(value, self.buffers[name], rtol=0, atol=0)

    def test_success_restores_mixed_modes_and_batchnorm(self):
        result = predict_with_mc_dropout(
            self.model, [self.batch], torch.device("cpu"), 3
        )
        self.assertEqual(result["image_id"], ["a", "b"])
        self.assertEqual(result["all_probabilities"].shape, (3, 2))
        self.assert_restored()

    def test_changed_pass_identity_fails_and_restores(self):
        for field, replacement in (
            (1, torch.tensor([1.0, 0.0])),
            (2, ["b", "a"]),
            (3, ["b.jpg", "a.jpg"]),
        ):
            changed = list(self.batch)
            changed[field] = replacement
            batches = [self.batch, tuple(changed)]

            class ChangingLoader:
                def __iter__(self):
                    return iter([batches.pop(0)])

            with self.subTest(field=field), self.assertRaises(ValueError):
                predict_with_mc_dropout(
                    self.model, ChangingLoader(), torch.device("cpu"), 2
                )
            self.assert_restored()

    def test_errors_empty_duplicate_nonfinite_and_count(self):
        bad_batches = [
            [],
            [(self.batch[0], self.batch[1], ["a", "a"], self.batch[3])],
            [(torch.full((2, 2), np.nan), *self.batch[1:])],
            [(self.batch[0], torch.tensor([0.0, 0.5]), *self.batch[2:])],
        ]
        for batches in bad_batches:
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                predict_with_mc_dropout(self.model, batches, torch.device("cpu"), 2)
            self.assert_restored()
        for passes in (0, 1, 2.5, True):
            with self.subTest(passes=passes), self.assertRaises(ValueError):
                predict_with_mc_dropout(
                    self.model, [self.batch], torch.device("cpu"), passes
                )
            self.assert_restored()

    def test_duplicate_paths_fail_and_restore_state(self):
        duplicate = (*self.batch[:3], ["same.jpg", "same.jpg"])
        with self.assertRaises(ValueError):
            predict_with_mc_dropout(self.model, [duplicate], torch.device("cpu"), 2)
        self.assert_restored()

    def test_forward_exception_restores_modes(self):
        def fail(*_):
            raise RuntimeError("injected forward failure")

        handle = self.model[2].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                predict_with_mc_dropout(
                    self.model, [self.batch], torch.device("cpu"), 2
                )
        finally:
            handle.remove()
        self.assert_restored()
