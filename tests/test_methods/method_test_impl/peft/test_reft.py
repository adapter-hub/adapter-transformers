from adapters import DiReftConfig, LoReftConfig, NoReftConfig
from tests.test_methods.method_test_impl.base import AdapterMethodBaseTestMixin
from transformers.testing_utils import require_torch


@require_torch
class ReftTestMixin(AdapterMethodBaseTestMixin):
    reft_configs_to_test = [
        (LoReftConfig(), ["refts.{name}."]),
        (NoReftConfig(prefix_positions=2, suffix_positions=2), ["refts.{name}."]),
        (DiReftConfig(tied_weights=True), ["refts.{name}."]),
    ]

    def test_add_reft(self):
        model = self.get_model()
        for adapter_config, filter_keys in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_add_test(model, adapter_config, filter_keys)

    def test_layers_reft(self):
        model = self.get_model()
        leave_out = self.leave_out_layers
        n_layers = len(list(model.iter_layers()))
        layers = list(set(range(n_layers)) - set(leave_out))
        for adapter_config, _ in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                model.eval()
                adapter_config = adapter_config.replace(layers=layers)
                name = "test_adapter_" + adapter_config.__class__.__name__
                model.add_adapter(name, config=adapter_config)
                model.set_active_adapters(name)

                # adapter is correctly added to config
                self._assert_adapter_available(model, name)

                adapter = model.get_adapter(name)

                self.assertNotEqual(len(adapter), 0)
                found_layers = list(adapter.keys())
                for layer in leave_out:
                    self.assertNotIn(layer, found_layers)

                model.delete_adapter(name)

    def test_average_reft(self):
        model = self.get_model()
        for adapter_config, filter_keys in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_linear_average_test(model, adapter_config, filter_keys)

    def test_delete_reft(self):
        model = self.get_model()
        for adapter_config, filter_keys in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_delete_test(model, adapter_config, filter_keys)

    def test_get_reft(self):
        model = self.get_model()
        n_layers = len(list(model.iter_layers()))

        for adapter_config, _ in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_get_test(model, adapter_config, n_layers)

    def test_forward_reft(self):
        model = self.get_model()
        for adapter_config, _ in self.reft_configs_to_test:
            with self.subTest(model_class=model.__class__.__name__, config=adapter_config.__class__.__name__):
                self.run_forward_test(model, adapter_config)

    def test_load_reft(self):
        self.run_load_test(LoReftConfig())

    def test_load_full_model_reft(self):
        self.run_full_model_load_test(LoReftConfig())

    def test_train_loreft(self):
        self.run_train_test(LoReftConfig(), ["refts.{name}."])

    def test_reft_generate(self):
        self.run_generate_test(LoReftConfig())
