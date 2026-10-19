"""Device-agnostic behaviour of the inference path.

Everything except the last class runs without an accelerator: detection order,
explicit device precedence, allocator cache dispatch, the embedding defaults and
the generator check. The last class compares CPU against XPU on a tiny
random-weight policy and is skipped when no XPU is present.
"""

import inspect
import types
import unittest
from unittest import mock

import torch

from hpt.utils import device as dev
from hpt.utils import utils
from hpt.utils.scheduling_ddpm import randn_tensor


def _patch_availability(*, xpu, cuda):
    def fake(device_type):
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    return mock.patch.object(dev, "is_available", side_effect=fake)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda"))

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))


class EmptyCacheTest(unittest.TestCase):
    def test_dispatch_follows_backend(self):
        with _patch_availability(xpu=True, cuda=True):
            with mock.patch.object(torch.xpu, "empty_cache") as xpu_cache, mock.patch.object(
                torch.cuda, "empty_cache"
            ) as cuda_cache:
                dev.empty_cache("xpu")
                xpu_cache.assert_called_once()
                cuda_cache.assert_not_called()
                dev.empty_cache("cuda:0")
                cuda_cache.assert_called_once()

    def test_cpu_and_missing_backend_are_noops(self):
        with _patch_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "empty_cache") as cuda_cache:
                dev.empty_cache("cpu")
                dev.empty_cache("cuda")
                cuda_cache.assert_not_called()


class EmbeddingDefaultsTest(unittest.TestCase):
    def test_no_embedding_helper_defaults_to_cuda(self):
        for name in ("get_image_embeddings", "get_dino_embeddings", "get_clip_embeddings",
                     "get_vit_embeddings", "get_resnet_embeddings", "get_r3m_embeddings",
                     "get_voltron_embeddings"):
            default = inspect.signature(getattr(utils, name)).parameters["device"].default
            self.assertIsNone(default, name)


class GeneratorCheckTest(unittest.TestCase):
    def test_cpu_generator_on_any_device_type(self):
        gen = torch.Generator("cpu").manual_seed(0)
        out = randn_tensor((2, 3), generator=gen, device=torch.device("cpu"))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_non_cpu_generator_for_other_device_is_rejected(self):
        fake_gen = types.SimpleNamespace(device=torch.device("xpu"))
        with self.assertRaises(ValueError):
            randn_tensor((2,), generator=fake_gen, device=torch.device("cpu"))


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU")
class XPUParityTest(unittest.TestCase):
    def test_policy_forward_cpu_vs_xpu(self):
        from omegaconf import OmegaConf

        from hpt.models.policy import Policy
        from hpt.utils.normalizer import LinearNormalizer

        def build():
            torch.manual_seed(0)
            policy = Policy(embed_dim=64, num_blocks=2, num_heads=4, observation_horizon=1, action_horizon=1,
                            token_postprocessing="mean")
            stem = OmegaConf.create({
                "modalities": ["state"], "modality_embed_dim": 64, "normalize_state": False,
                "cross_attention": True, "precompute_feat": True, "image_encoder": "resnet",
                "crossattn_dim_head": 16, "crossattn_heads": 4, "crossattn_modality_dropout": 0.0,
                "num_blocks": 1, "observation_horizon": 1, "masked_autoencoding": False,
                "random_horizon_masking": False, "add_pos_embedding_to_state": False,
                "crossattn_latent": {"state": 4}, "state_embedding_dim": 1,
                "state": {"_target_": "hpt.models.policy_stem.MLP", "input_dim": 5, "output_dim": 64, "widths": [32]},
            })
            head = OmegaConf.create({
                "_target_": "hpt.models.policy_head.MLP", "input_dim": 64, "tanh_end": True,
                "output_dim": 6, "widths": [32], "normalize_action": False, "dropout": False,
            })
            policy.init_domain_stem("d", stem)
            normalizer = LinearNormalizer()
            normalizer.fit({"state": torch.randn(16, 5), "action": torch.rand(16, 6) * 2 - 1})
            policy.init_domain_head("d", normalizer, head)
            policy.finalize_modules()
            with torch.no_grad():  # no zero-initialised layer may hide a mismatch
                for p in policy.parameters():
                    p.copy_(torch.randn_like(p) * 0.1)
            policy.train_mode = False  # normally set by get_action
            return policy.eval()

        state = torch.randn(2, 1, 5)
        outs = {}
        for name in ("cpu", "xpu"):
            policy = build().to(name)
            with torch.no_grad():
                outs[name] = policy("d", {"state": state.to(name).clone()}).cpu()
        torch.testing.assert_close(outs["xpu"], outs["cpu"], atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
