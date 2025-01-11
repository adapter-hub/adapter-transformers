from adapters import VeraConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class VeraTestMixin(AdapterMethodBaseTestMixin):
    def test_add_vera(self):
        model = self.get_model()
        self.run_add_test(model, VeraConfig(), ["loras.{name}."])

    def test_leave_out_vara(self):
        model = self.get_model()
        self.run_leave_out_test(model, VeraConfig(), self.leave_out_layers)

    def test_linear_average_vera(self):
        model = self.get_model()
        self.run_linear_average_test(model, VeraConfig(), ["loras.{name}."])

    def test_delete_vera(self):
        model = self.get_model()
        self.run_delete_test(model, VeraConfig(), ["loras.{name}."])

    def test_get_vera(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))
        self.run_get_test(model, VeraConfig(intermediate_lora=True, output_lora=True), n_layers * 3)

    def test_forward_vera(self):
        model = self.get_model()
        self.run_forward_test(model, VeraConfig(init_weights="vera", intermediate_lora=True, output_lora=True))

    def test_load_vera(self):
        self.run_load_test(VeraConfig())

    def test_load_full_model_vera(self):
        self.run_full_model_load_test(VeraConfig(init_weights="vera"))

    def test_train_vera(self):
        self.run_train_test(VeraConfig(init_weights="vera"), ["loras.{name}."])

    def test_merge_vera(self):
        self.run_merge_test(VeraConfig(init_weights="vera"))

    def test_reset_vera(self):
        self.run_reset_test(VeraConfig(init_weights="vera"))
