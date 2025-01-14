from adapters import PromptTuningConfig
from transformers.testing_utils import require_torch

from .base import AdapterMethodBaseTestMixin


@require_torch
class PromptTuningTestMixin(AdapterMethodBaseTestMixin):
    def test_add_prompt_tuning(self):
        model = self.get_model()
        self.run_add_test(model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_linear_average_prompt_tuning(self):
        model = self.get_model()
        self.run_linear_average_test(model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_delete_prompt_tuning(self):
        model = self.get_model()
        self.run_delete_test(model, PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_get_prompt_tuning(self):
        model = self.get_model()
        self.run_get_test(model, PromptTuningConfig(prompt_length=10), 1)

    def test_forward_prompt_tuning(self):
        model = self.get_model()
        for dtype in self.dtypes_to_test:
            with self.subTest(model_class=model.__class__.__name__, dtype=dtype):
                self.run_forward_test(model, PromptTuningConfig(prompt_length=10), dtype=dtype)

    def test_load_prompt_tuning(self):
        self.run_load_test(PromptTuningConfig(prompt_length=10))

    def test_load_full_model_prompt_tuning(self):
        self.run_full_model_load_test(PromptTuningConfig(prompt_length=10))

    def test_train_prompt_tuning(self):
        self.run_train_test(PromptTuningConfig(prompt_length=10), ["prompt_tunings.{name}."])

    def test_prompt_tuning_gradient_checkpointing_single_adapter(self):
        self.run_gradient_checkpointing_single_adapter_test(PromptTuningConfig(prompt_length=10))
