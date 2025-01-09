import random

import torch

from adapters import AdapterConfig, LoRAConfig
from adapters.methods.lora import LoRALayer
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class LoRATestMixin(AdapterMethodBaseTestMixin):
    def test_add_lora(self):
        model = self.get_model()
        self.run_add_test(model, LoRAConfig(), ["loras.{name}."])

    def test_leave_out_lora(self):
        model = self.get_model()
        self.run_leave_out_test(model, LoRAConfig(), self.leave_out_layers)

    def test_merging_with_other_adapters(self):
        model = self.get_model()
        model.add_adapter("lora", config="lora")

        # Add different adapters
        adapter_methods = [
            "seq_bn",
            "prompt_tuning",
            "prefix_tuning",
            "ia3",
            "unipelt",
            "mam",
            "compacter[phm_dim=2, reduction_factor=8]",
        ]

        for adapter_method in adapter_methods:
            config = AdapterConfig.load(adapter_method)
            if model.supports_adapter(config):
                model.add_adapter(adapter_method, config=config)

        # Merging adapters with different architectures with LoRA should raise a ValueError
        for adapter_method in adapter_methods:
            with self.subTest(adapter_architecture=adapter_method):
                if adapter_method not in model.adapters_config:
                    continue
                with self.assertRaises(ValueError):
                    model.average_adapter(
                        adapter_name=f"average_lora_{adapter_method}",
                        adapter_list=[adapter_method, "lora"],
                        weights=[0.5, 0.5],
                        combine_strategy="linear",
                    )

    def test_linear_average_lora(self):
        model = self.get_model()
        self.run_linear_average_test(model, LoRAConfig(), ["loras.{name}."])

    def test_linear_average_only_negate_b_lora(self):
        # This method tests that the linear average following the Zhang et al. 2023 paper works as expected.
        # Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html
        # This method is an adapted version of the `run_linear_average_test` method.
        model = self.get_model()
        model.eval()
        weights = [-1, 1.5, 0.5]

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(len(weights)):
            model.add_adapter(
                f"{name}_{i}",
                config=LoRAConfig(
                    dropout=random.random(),
                    init_weights=["bert", "lora"][i % 2],
                ),
            )

        averaged_weights = {}
        for i, w in enumerate(weights):
            this_filter_keys = [k.format(name=f"{name}_{i}") for k in ["loras.{name}."]]
            for k, v in self.filter_parameters(model, this_filter_keys).items():
                base_k = k.replace(f"{name}_{i}", name)
                # Only negate the lora_B weights and use the absolute value of the weight for lora_A weights.
                weight = abs(w) if "lora_A" in k else w
                if base_k not in averaged_weights:
                    averaged_weights[base_k] = weight * v
                else:
                    averaged_weights[base_k] += weight * v

        # average adapters
        model.average_adapter(
            name,
            [f"{name}_{i}" for i in range(len(weights))],
            weights=weights,
            combine_strategy="lora_linear_only_negate_b",
        )

        # adapter is correctly added to config
        self.assertTrue(name in model.adapters_config)
        config = model.adapters_config.get(name)
        self.assertEqual(LoRAConfig(dropout=config.dropout, init_weights=config.init_weights), config)

        # compare averaged weights to collected weights
        this_filter_keys = [k.format(name=name) for k in ["loras.{name}."]]
        for k, v in self.filter_parameters(model, this_filter_keys).items():
            self.assertTrue(torch.allclose(v, averaged_weights[k]), k)

    def _check_svd_weights(self, delta_w, merged_lora, svd_rank, atol=1e-5):
        # Compute SVD of the original delta_w
        u, s, v = torch.svd(delta_w)
        u = u[:, :svd_rank]
        s = s[:svd_rank]
        v = v[:, :svd_rank]

        # Reconstruct A and B matrices
        expected_A = v.t()
        expected_B = u @ torch.diag(s)

        # Compare with merged adapter
        self.assertTrue(torch.allclose(expected_A, merged_lora.lora_A, atol=atol))
        self.assertTrue(torch.allclose(expected_B, merged_lora.lora_B, atol=atol))

    def test_linear_delta_w_svd_average_lora(self):
        model = self.get_model()
        model.eval()
        model_supports_lora_delta_w_svd = model.base_model.support_lora_delta_w_svd
        weights = [-1, 1.5, 0.5]

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(len(weights)):
            model.add_adapter(
                f"{name}_{i}",
                config=LoRAConfig(
                    dropout=random.random(),
                    init_weights=["bert", "lora"][i % 2],
                ),
            )

        if not model_supports_lora_delta_w_svd:
            # Some models (GPT2, Deberta) don't support this merging method
            with self.assertRaises(ValueError):
                model.average_adapter(
                    "averaged_adapter",
                    [f"{name}_{i}" for i in range(len(weights))],
                    weights=weights,
                    combine_strategy="lora_delta_w_svd",
                )

            return

        # average adapters
        svd_rank = 16
        model.average_adapter(
            "averaged_adapter",
            [f"{name}_{i}" for i in range(len(weights))],
            weights=weights,
            combine_strategy="lora_delta_w_svd",
            svd_rank=svd_rank,
        )

        # adapter is correctly added to config
        self.assertTrue("averaged_adapter" in model.adapters_config)
        config = model.adapters_config.get("averaged_adapter")
        self.assertEqual(LoRAConfig(dropout=config.dropout, init_weights=config.init_weights, r=svd_rank), config)

        # Calculate the new weights: Matrix A and B are SVD of all the weighted delta_w matrices of the adapters.
        for i, layer in model.iter_layers():
            for module in layer.modules():
                if isinstance(module, LoRALayer):
                    # Check if this layer has the LoRA adapters
                    if not (
                        f"{name}_0" in module.loras
                        and f"{name}_1" in module.loras
                        and f"{name}_2" in module.loras
                        and name in module.loras
                    ):
                        continue

                    # Calculate the new weights
                    delta_w_1 = module.loras[name + "_0"].delta_w
                    delta_w_2 = module.loras[name + "_1"].delta_w
                    delta_w_3 = module.loras[name + "_2"].delta_w
                    delta_w = weights[0] * delta_w_1 + weights[1] * delta_w_2 + weights[2] * delta_w_3

                    self._check_svd_weights(delta_w, module.loras["averaged_adapter"], svd_rank)

    def test_edge_case_average_adapters_single_adapter(self):
        # If we merge only one adapter, the weights of the new adapter should be the same as the original adapter
        model = self.get_model()
        model.eval()
        model_supports_lora_delta_w_svd = model.base_model.support_lora_delta_w_svd

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(3):
            model.add_adapter(
                f"{name}_{i}",
                config=LoRAConfig(
                    dropout=random.random(),
                    init_weights=["bert", "lora"][i % 2],
                ),
            )

        # collect weights of the first adapter so we can compare them to the newly created adapters in the subsequent tests
        filter_keys_adapter_0 = [k.format(name=f"{name}_0") for k in ["loras.{name}."]]
        adapter_0 = self.filter_parameters(model, filter_keys_adapter_0)

        # Run tests for every combine strategy
        for combine_strategy in ["linear", "lora_linear_only_negate_b", "lora_delta_w_svd"]:
            if not model_supports_lora_delta_w_svd and combine_strategy == "lora_delta_w_svd":
                continue

            with self.subTest(combine_strategy=combine_strategy):
                svd_rank = LoRAConfig().r if combine_strategy == "lora_delta_w_svd" else None
                model.average_adapter(
                    adapter_name=f"{combine_strategy}_merged",
                    adapter_list=[f"{name}_0"],
                    weights=[1],
                    combine_strategy=combine_strategy,
                    svd_rank=svd_rank,
                )

                filter_keys = [k.format(name=f"{combine_strategy}_merged") for k in ["loras.{name}."]]

                if combine_strategy != "lora_delta_w_svd":
                    for k, v in self.filter_parameters(model, filter_keys).items():
                        adapter_0_key = k.replace(f"{combine_strategy}_merged", f"{name}_0")
                        self.assertTrue(torch.allclose(v, adapter_0[adapter_0_key]))
                else:
                    # For lora_delta_w_svd, we need to calculate the expected weights since lora_delta_w_svd performs an SVD
                    for i, layer in model.iter_layers():
                        for module in layer.modules():
                            if isinstance(module, LoRALayer):
                                if f"{name}_0" in module.loras and f"{combine_strategy}_merged" in module.loras:
                                    original_lora = module.loras[f"{name}_0"]
                                    merged_lora = module.loras[f"{combine_strategy}_merged"]
                                    self._check_svd_weights(original_lora.delta_w, merged_lora, svd_rank)

    def test_edge_case_average_adapters_multiple_adapters(self):
        # If we merge multiple adapters with weight 0 except one adapter with weight 1, the resulting adapter should be the same as the adapter with weight 1
        model = self.get_model()
        model.eval()
        model_supports_lora_delta_w_svd = model.base_model.support_lora_delta_w_svd

        # add adapters to average
        name = "test_adapter_" + LoRAConfig().__class__.__name__
        for i in range(3):
            model.add_adapter(
                f"{name}_{i}",
                config=LoRAConfig(
                    dropout=random.random(),
                    init_weights=["bert", "lora"][i % 2],
                ),
            )

        # collect weights of the first adapter so we can compare them to the newly created adapters in the subsequent tests
        filter_keys_adapter_0 = [k.format(name=f"{name}_0") for k in ["loras.{name}."]]
        adapter_0 = self.filter_parameters(model, filter_keys_adapter_0)

        # Run tests for every combine strategy
        for combine_strategy in ["linear", "lora_linear_only_negate_b", "lora_delta_w_svd"]:
            if not model_supports_lora_delta_w_svd and combine_strategy == "lora_delta_w_svd":
                continue

            with self.subTest(combine_strategy=combine_strategy):
                svd_rank = LoRAConfig().r if combine_strategy == "lora_delta_w_svd" else None

                # since normalize_weights is True, this should result in only the first adapter being used with a weight of 1
                model.average_adapter(
                    adapter_name=f"{combine_strategy}_merged",
                    adapter_list=[f"{name}_0", f"{name}_1", f"{name}_2"],
                    weights=[0.5, 0, 0],
                    combine_strategy=combine_strategy,
                    svd_rank=svd_rank,
                )

                filter_keys = [k.format(name=f"{combine_strategy}_merged") for k in ["loras.{name}."]]

                if combine_strategy != "lora_delta_w_svd":
                    for k, v in self.filter_parameters(model, filter_keys).items():
                        adapter_1_key = k.replace(f"{combine_strategy}_merged", f"{name}_0")
                        self.assertTrue(torch.allclose(v, adapter_0[adapter_1_key]))
                else:
                    # For lora_delta_w_svd, we need to calculate the expected weights since lora_delta_w_svd performs an SVD
                    for i, layer in model.iter_layers():
                        for module in layer.modules():
                            if isinstance(module, LoRALayer):
                                if f"{name}_0" in module.loras and f"{combine_strategy}_merged" in module.loras:
                                    original_lora = module.loras[f"{name}_0"]
                                    merged_lora = module.loras[f"{combine_strategy}_merged"]
                                    self._check_svd_weights(original_lora.delta_w, merged_lora, svd_rank)

    def test_delete_lora(self):
        model = self.get_model()
        self.run_delete_test(model, LoRAConfig(), ["loras.{name}."])

    def test_get_lora(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, LoRAConfig(intermediate_lora=True, output_lora=True), n_layers * 3)

    def test_forward_lora(self):
        model = self.get_model()
        for dtype in self.dtypes_to_test:
            with self.subTest(model_class=model.__class__.__name__, dtype=dtype):
                self.run_forward_test(
                    model, LoRAConfig(init_weights="bert", intermediate_lora=True, output_lora=True), dtype=dtype
                )

    def test_load_lora(self):
        self.run_load_test(LoRAConfig())

    def test_load_full_model_lora(self):
        self.run_full_model_load_test(LoRAConfig(init_weights="bert"))

    def test_train_lora(self):
        self.run_train_test(LoRAConfig(init_weights="bert"), ["loras.{name}."])

    def test_merge_lora(self):
        self.run_merge_test(LoRAConfig(init_weights="bert"))

    def test_reset_lora(self):
        self.run_reset_test(LoRAConfig(init_weights="bert"))
