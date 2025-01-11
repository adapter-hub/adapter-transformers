from adapters.configuration import (
    CompacterConfig,
    ConfigUnion,
    LoRAConfig,
    ParBnConfig,
    PrefixTuningConfig,
    SeqBnConfig,
)
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class ConfigUnionAdapterTest(AdapterMethodBaseTestMixin):
    adapter_configs_to_test = [
        (
            ConfigUnion(
                PrefixTuningConfig(),
                ParBnConfig(phm_dim=1),
            ),
            ["adapters.{name}.", "prefix_tunings.{name}."],
        ),
        (
            ConfigUnion(
                CompacterConfig(phm_dim=1),
                LoRAConfig(),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
        (
            ConfigUnion(
                SeqBnConfig(phm_dim=1),
                LoRAConfig(),
            ),
            ["adapters.{name}.", "loras.{name}."],
        ),
    ]

    def test_add_union_adapter(self):
        # TODO: Discuss, why old tests were not working properly (could not work because we would add three times the same adapter name)
        # TODO: Discuss why these config unions are not working properly (must set phm_dim=1)
        for adapter_config, filter_keys in self.adapter_configs_to_test:
            model = self.get_model()
            model.eval()
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_union_adapter_forward(self):
        model = self.get_model()
        model.eval()

        for adapter_config, _ in self.adapter_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_forward_test(model, adapter_config)
