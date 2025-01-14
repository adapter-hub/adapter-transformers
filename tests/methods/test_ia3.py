from adapters import IA3Config
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class IA3TestMixin(AdapterMethodBaseTestMixin):
    def test_add_ia3(self):
        model = self.get_model()
        self.run_add_test(model, IA3Config(), ["loras.{name}."])

    def test_leave_out_ia3(self):
        model = self.get_model()
        self.run_leave_out_test(model, IA3Config(), self.leave_out_layers)

    def test_linear_average_ia3(self):
        model = self.get_model()
        self.run_linear_average_test(model, IA3Config(), ["loras.{name}."])

    def test_delete_ia3(self):
        model = self.get_model()
        self.run_delete_test(model, IA3Config(), ["loras.{name}."])

    def test_get_ia3(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, IA3Config(intermediate_lora=True, output_lora=True), n_layers * 3)

    def test_forward_ia3(self):
        model = self.get_model()
        self.run_forward_test(model, IA3Config(init_weights="bert", intermediate_lora=True, output_lora=True))

    def test_load_ia3(self):
        self.run_load_test(IA3Config())

    def test_load_full_model_ia3(self):
        self.run_full_model_load_test(IA3Config(init_weights="bert"))

    def test_train_ia3(self):
        self.run_train_test(IA3Config(init_weights="bert"), ["loras.{name}."])

    def test_merge_ia3(self):
        self.run_merge_test(IA3Config(init_weights="bert"))

    def test_reset_ia3(self):
        self.run_reset_test(IA3Config(init_weights="bert"))

    def test_ia3_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(IA3Config())
