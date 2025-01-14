from adapters import UniPELTConfig
from transformers.testing_utils import require_torch, torch_device

from .base import AdapterMethodBaseTestMixin


@require_torch
class UniPELTTestMixin(AdapterMethodBaseTestMixin):
    def test_add_unipelt(self):
        model = self.get_model()
        self.run_add_test(model, UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."])

    def test_linear_average_unipelt(self):
        model = self.get_model()
        self.run_linear_average_test(
            model, UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."]
        )

    def test_delete_unipelt(self):
        model = self.get_model()
        self.run_delete_test(model, UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}."])

    def test_get_unipelt(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        # In UniPELT, prefix tuning has gates in every layer
        n_prefix_layers = 1.5 * n_layers if model.config.is_encoder_decoder else n_layers
        self.run_get_test(model, UniPELTConfig(), n_layers * 2 + n_prefix_layers)

    def test_forward_unipelt(self):
        model = self.get_model()
        self.run_forward_test(model, UniPELTConfig())

    def test_load_unipelt(self):
        self.run_load_test(UniPELTConfig())

    def test_load_full_model_unipelt(self):
        self.run_full_model_load_test(UniPELTConfig())

    def test_train_unipelt(self):
        self.run_train_test(
            UniPELTConfig(), ["loras.{name}.", "adapters.{name}.", "prefix_tunings.{name}.", "prefix_gates.{name}."]
        )

    def test_output_adapter_gating_scores_unipelt(self):
        model = self.get_model()
        model.eval()

        adapter_config = UniPELTConfig()
        name = adapter_config.__class__.__name__
        model.add_adapter(name, config=adapter_config)
        model.to(torch_device)

        input_data = self.get_input_samples(config=model.config)

        model.set_active_adapters(name)
        output_1 = model(**input_data, output_adapter_gating_scores=True)

        self.assertEqual(len(output_1[0]), self.default_input_samples_shape[0])
        self.assertTrue(hasattr(output_1, "adapter_gating_scores"))
        gating_scores = output_1.adapter_gating_scores[name]
        self.assertEqual(len(list(model.iter_layers())), len(gating_scores))
        for k, per_layer_scores in gating_scores.items():
            self.assertGreaterEqual(len(per_layer_scores), 3)
            for k, v in per_layer_scores.items():
                self.assertEqual(self.default_input_samples_shape[0], v.shape[0], k)

    def test_unipelt_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(UniPELTConfig())
