import json
import logging
from abc import ABC, abstractmethod
from os import mkdir
from os.path import exists, isdir, isfile, join
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch
from packaging.version import Version


try:
    from safetensors.torch import load_file, save_file

    safetensors_available = True
except ImportError:
    safetensors_available = False

from .configuration import AdapterConfig, build_full_config
from .head_utils import STATIC_TO_FLEX_HEAD_MAP, get_head_config_and_rename_list
from .utils import (
    ACTIVATION_RENAME,
    ADAPTERFUSION_CONFIG_NAME,
    ADAPTERFUSION_WEIGHTS_NAME,
    CONFIG_NAME,
    HEAD_CONFIG_NAME,
    HEAD_WEIGHTS_NAME,
    SAFE_ADAPTERFUSION_WEIGHTS_NAME,
    SAFE_HEAD_WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    AdapterType,
    resolve_adapter_path,
)


logger = logging.getLogger(__name__)


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(
        self, model, weights_name, config_name, use_safetensors: bool = False, safe_weights_name: Optional[str] = None
    ):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name
        self.use_safetensors = use_safetensors
        if use_safetensors and not safetensors_available:
            raise ValueError("Safetensors package not available. Please install via `pip install safetensors`.")
        self.safe_weights_name = safe_weights_name or weights_name

    def state_dict(self, filter_func):
        return {k: v for (k, v) in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, *rename_funcs):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for rename_func in rename_funcs:
                new_k = rename_func(new_k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        # add meta information if given
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        # save to file system
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the module weights can be saved."

        # Get the state of all adapter modules for this task
        state_dict = self.state_dict(filter_func)
        # Save the adapter weights
        if self.use_safetensors:
            output_file = join(save_directory, self.safe_weights_name)
            save_file(state_dict, output_file)
        else:
            output_file = join(save_directory, self.weights_name)
            torch.save(state_dict, output_file)
        logger.info("Module weights saved in {}".format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info("Loading module configuration from {}".format(config_file))
        # Load the config
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        # For older versions translate the activation function to the new format
        if "version" not in loaded_config:
            if "config" in loaded_config and loaded_config["config"] is not None:
                if (
                    "non_linearity" in loaded_config["config"]
                    and loaded_config["config"]["non_linearity"] in ACTIVATION_RENAME
                ):
                    loaded_config["config"]["non_linearity"] = ACTIVATION_RENAME[
                        loaded_config["config"]["non_linearity"]
                    ]
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix=start_prefix)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys

    def load_weights(
        self,
        save_directory,
        filter_func,
        rename_func=None,
        loading_info=None,
        in_base_model=False,
    ):
        # Load the weights of the adapter
        try:
            if self.use_safetensors:
                weights_file = join(save_directory, self.safe_weights_name)
                if exists(weights_file):
                    state_dict = load_file(weights_file, device="cpu")
                else:
                    logger.info(f"No safetensors file found in {save_directory}. Falling back to torch.load...")
                    weights_file = join(save_directory, self.weights_name)
                    state_dict = torch.load(weights_file, map_location="cpu")
            else:
                weights_file = join(save_directory, self.weights_name)
                state_dict = torch.load(weights_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")
        logger.info("Loading module weights from {}".format(weights_file))

        return self.load_weights_from_state_dict(
            state_dict, filter_func, rename_func=rename_func, loading_info=loading_info, in_base_model=in_base_model
        )

    def load_weights_from_state_dict(
        self, state_dict, filter_func, rename_func=None, loading_info=None, in_base_model=False, start_prefix=""
    ):
        # Rename weights if needed
        if rename_func:
            if isinstance(rename_func, Sequence):
                state_dict = self.rename_state_dict(state_dict, *rename_func)
            else:
                state_dict = self.rename_state_dict(state_dict, rename_func)

        # Add the weights to the model
        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = self.model
        has_prefix_module = any(s.startswith(self.model.base_model_prefix) for s in state_dict.keys())
        if not start_prefix and not hasattr(self.model, self.model.base_model_prefix) and has_prefix_module:
            start_prefix = self.model.base_model_prefix + "."
        if in_base_model and hasattr(self.model, self.model.base_model_prefix) and not has_prefix_module:
            model_to_load = self.model.base_model

        missing_keys, unexpected_keys = self._load_module_state_dict(
            model_to_load, state_dict, start_prefix=start_prefix
        )

        missing_keys = [k for k in missing_keys if filter_func(k)]
        if len(missing_keys) > 0:
            logger.info(
                "Some module weights could not be found in loaded weights file: {}".format(", ".join(missing_keys))
            )
        if self.model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if k not in self.model._keys_to_ignore_on_load_unexpected]
        if len(unexpected_keys) > 0:
            logger.info(
                "Some weights of the state_dict could not be loaded into model: {}".format(", ".join(unexpected_keys))
            )

        if isinstance(loading_info, dict):
            if "missing_keys" not in loading_info:
                loading_info["missing_keys"] = []
            if "unexpected_keys" not in loading_info:
                loading_info["unexpected_keys"] = []
            loading_info["missing_keys"].extend(missing_keys)
            loading_info["unexpected_keys"].extend(unexpected_keys)

        return missing_keys, unexpected_keys


class WeightsLoader(ABC):
    """
    An abstract class providing basic methods for saving and loading weights of a model. Extend this class to build
    custom module weight loaders.
    """

    def __init__(
        self, model, weights_name, config_name, use_safetensors: bool = False, safe_weights_name: Optional[str] = None
    ):
        self.model = model
        self.weights_helper = WeightsLoaderHelper(
            model, weights_name, config_name, use_safetensors=use_safetensors, safe_weights_name=safe_weights_name
        )

    @abstractmethod
    def filter_func(self, name: str) -> Callable[[str], bool]:
        """
        The callable returned by this method is used to extract the module weights to be saved or loaded based on their
        names.

        Args:
            name (str): An identifier of the weights to be saved.

        Returns:
            Callable[str, bool]: A function that takes the fully qualified name of a module parameter and returns a
            boolean value that specifies whether this parameter should be extracted.
        """
        pass

    @abstractmethod
    def rename_func(self, old_name: str, new_name: str) -> Callable[[str], str]:
        """
        The callable returned by this method is used to optionally rename the module weights after loading.

        Args:
            old_name (str): The string identifier of the weights as loaded from file.
            new_name (str): The new string identifier to which the weights should be renamed.

        Returns:
            Callable[str, str]: A function that takes the fully qualified name of a module parameter and returns a new
            fully qualified name.
        """
        pass

    def save(self, save_directory, name, **kwargs):
        """
        Saves the module config and weights into the given directory. Override this method for additional saving
        actions.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str): An identifier of the weights to be saved. The details are specified by the implementor.
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where weights and configuration can be saved."

        config_dict = build_full_config(
            None,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )
        meta_dict = kwargs.pop("meta_dict", None)

        # Save the adapter configuration
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)

        # Save adapter weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None, **kwargs) -> Tuple[str, str]:
        """
        Loads the module weights from the given directory. Override this method for additional loading actions. If
        adding the loaded weights to the model passed to the loader class requires adding additional modules, this
        method should also perform the architectural changes to the model.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        if not exists(join(save_directory, self.weights_helper.weights_name)):
            raise ValueError("Loading path should be a directory where the weights are saved.")

        # Load config
        config = self.weights_helper.load_weights_config(save_directory)

        # Load head weights
        filter_func = self.filter_func(config["name"])
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_func, loading_info=loading_info
        )

        return save_directory, load_as or config["name"]


class AdapterLoader(WeightsLoader):
    """
    A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url.

    Model classes passed to this loader must implement the `ModelAdaptersMixin` class.
    """

    def __init__(self, model, adapter_type=None, use_safetensors: bool = False):
        super().__init__(
            model, WEIGHTS_NAME, CONFIG_NAME, use_safetensors=use_safetensors, safe_weights_name=SAFE_WEIGHTS_NAME
        )
        self.adapter_type = adapter_type
        if adapter_type and not AdapterType.has(self.adapter_type):
            raise ValueError("Invalid adapter type {}".format(self.adapter_type))

    def filter_func(self, adapter_name):
        return (
            lambda x: "_adapters.{}.".format(adapter_name) in x
            or ".adapters.{}.".format(adapter_name) in x
            or ".prefix_tunings.{}.".format(adapter_name) in x
            or ".prefix_gates.{}.".format(adapter_name) in x
            or ".loras.{}.".format(adapter_name) in x
            or ".refts.{}.".format(adapter_name) in x
            or ".prompt_tunings.{}.".format(adapter_name) in x
        )

    # This dict maps the original weight names to the currently used equivalents.
    # The mapping is used by rename_func() to support loading from older weights files.
    # Old adapters will be loaded and converted to the new format automatically.
    legacy_weights_mapping = {
        "attention_text_task_adapters": "adapters",
        "attention_text_lang_adapters": "adapters",
        "layer_text_task_adapters": "adapters",
        "layer_text_lang_adapters": "adapters",
        "invertible_lang_adapters": "invertible_adapters",
    }

    def _rename_legacy_weights(self, k):
        for old, new in self.legacy_weights_mapping.items():
            k = k.replace(old, new)
        return k

    def _fix_backward_compat(self, config):
        # Fix error in previous versions for LoRA/ (IA)^3
        ADAPTER_PREFIX = "adapters."
        MIN_VERSION = Version("1.1.0")

        version = config.get("version", "")
        if version.startswith(ADAPTER_PREFIX) and Version(version[len(ADAPTER_PREFIX) :]) < MIN_VERSION:
            if (
                config["config"].get("architecture", None) == "lora"
                and config["config"]["r"] != config["config"]["alpha"]
            ):
                logger.warning(
                    "Loading a LoRA trained using a faulty scaling implementation of a previous library version. Editing the configuration to make sure the adapter works as trained."
                    "See https://github.com/adapter-hub/adapters/pull/770 for more."
                )
                config["config"]["alpha"] = config["config"]["r"]

    # This method is used to remove unnecessary invertible adapters from task adapters using the old format.
    # In the old format, task adapters e.g. using seq_bn config specify inv. adapters but don't use them.
    # As inv. adapters would be incorrectly used in the new implementation,
    # catch this case here when loading pretrained adapters.
    def _fix_legacy_config(self, adapter_name, missing_keys):
        if self.adapter_type == AdapterType.text_task:
            inv_adapter_keys = [x for x in missing_keys if f"invertible_adapters.{adapter_name}." in x]
            if len(inv_adapter_keys) > 0:
                del self.model.base_model.invertible_adapters[adapter_name]
                missing_keys = [k for k in missing_keys if k not in inv_adapter_keys]
                # remove invertible_adapter from config
                adapter_config_name = self.model.adapters_config.adapters[adapter_name]
                if adapter_config_name in self.model.adapters_config.config_map:
                    adapter_config = self.model.adapters_config.config_map[adapter_config_name]
                    self.model.adapters_config.config_map[adapter_config_name] = adapter_config.replace(
                        inv_adapter=None, inv_adapter_reduction_factor=None
                    )
        return missing_keys

    def rename_func(self, old_name, new_name):
        return (
            lambda k: self._rename_legacy_weights(k)
            .replace("adapters.{}.".format(old_name), "adapters.{}.".format(new_name))
            .replace(".prefix_tunings.{}.".format(old_name), ".prefix_tunings.{}.".format(new_name))
            .replace(".prefix_gates.{}.".format(old_name), ".prefix_gates.{}.".format(new_name))
            .replace(".loras.{}.".format(old_name), ".loras.{}.".format(new_name))
            .replace(".refts.{}.".format(old_name), ".refts.{}.".format(new_name))
        )

    def save_to_state_dict(self, name: str):
        """
        Extracts the weights of a given adapter from the model and returns them as a state dict.

        Args:
            name (str): The name of the adapter to be saved.

        Returns:
            Tuple[dict, dict]: A tuple consisting of the state dict containing the adapter weights and the adapter
            configuration.
        """
        if name not in self.model.adapters_config.adapters:
            raise ValueError("No adapter of this type with the given name is part of this model.")

        adapter_config = self.model.adapters_config.get(name)

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        state_dict = self.weights_helper.state_dict(filter_func)

        return state_dict, config_dict

    def save(self, save_directory, name, meta_dict=None):
        """
        Saves an adapter and its configuration file to a directory, so that it can be reloaded using the `load()`
        method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the adapter to be saved
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where adapter and configuration can be saved."
        assert (
            name in self.model.adapters_config.adapters
        ), "No adapter of this type with the given name is part of this model."

        adapter_config = self.model.adapters_config.get(name)

        self.model.apply_to_adapter_layers(lambda _, layer: layer.pre_save_adapters())

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save the adapter configuration
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        self.weights_helper.save_weights(save_directory, filter_func)

    def load_from_state_dict(self, state_dict, name, load_as=None, loading_info=None, start_prefix=""):
        """
        Loads the weights of a given adapter from a state dict into the model.

        Args:
            state_dict (dict): The state dict from which to load the adapter weights.
            name (str): The name of the adapter to be loaded.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                saved will be used.
            loading_info (dict, optional):
                A dictionary to which loading information (missing and unexpected keys) will be added.
            start_prefix (str, optional): A custom prefix to be ignored in the given state dict.
        """
        new_adapter_name = load_as or name
        if new_adapter_name not in self.model.adapters_config.adapters:
            raise ValueError("No adapter of this type with the given name is part of this model.")

        # Load adapter weights
        filter_func = self.filter_func(name)
        rename_func = self.rename_func(name, new_adapter_name)
        missing_keys, _ = self.weights_helper.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
            start_prefix=start_prefix,
        )
        missing_keys = self._fix_legacy_config(new_adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

    def load(
        self,
        adapter_name_or_path,
        config=None,
        version=None,
        model_name=None,
        load_as=None,
        loading_info=None,
        leave_out=None,
        set_active=False,
        **kwargs,
    ):
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (str, optional): Deprecated.
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): Deprecated.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
             saved will be used.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        # Warn about deprecated arguments
        if config is not None or model_name is not None:
            logger.warning(
                "The 'config' and 'model_name' arguments are specific to the now unsupported legacy Hub repo and will"
                " be removed."
                "Please switch to only providing the HF Model Hub identifier.",
            )
        requested_config = AdapterConfig.load(config) if config else None
        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(
            adapter_name_or_path,
            model_name,
            adapter_config=requested_config,
            version=version,
            **kwargs,
        )

        # Load config of adapter
        config = self.weights_helper.load_weights_config(resolved_folder)
        if self.adapter_type and "type" in config:
            assert config["type"] == self.adapter_type, "Loaded adapter has to be a {} adapter.".format(
                self.adapter_type
            )
        elif "type" in config:
            self.adapter_type = config["type"]
        # post-loading drop of layers
        if leave_out is not None:
            if "leave_out" in config["config"] and config["config"]["leave_out"] is not None:
                # The conversion to a set and then back to a list removes all duplicates
                leave_out = list(set(leave_out + config["config"]["leave_out"]))
            config["config"]["leave_out"] = leave_out
        # Fix issues
        self._fix_backward_compat(config)

        adapter_name = load_as or config["name"]
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.adapters_config.adapters:
            self.model.add_adapter(adapter_name, config=config["config"], set_active=set_active)
        else:
            logger.warning("Overwriting existing adapter '{}'.".format(adapter_name))

        # Load adapter weights
        filter_func = self.filter_func(adapter_name)
        rename_func = self.rename_func(config["name"], adapter_name)
        missing_keys, _ = self.weights_helper.load_weights(
            resolved_folder, filter_func, rename_func=rename_func, loading_info=loading_info, in_base_model=True
        )
        missing_keys = self._fix_legacy_config(adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

        return resolved_folder, adapter_name


class AdapterFusionLoader(WeightsLoader):
    """
    A class providing methods for saving and loading AdapterFusion modules from the file system.

    """

    def __init__(self, model, error_on_missing=True, use_safetensors: bool = False):
        super().__init__(
            model,
            ADAPTERFUSION_WEIGHTS_NAME,
            ADAPTERFUSION_CONFIG_NAME,
            use_safetensors=use_safetensors,
            safe_weights_name=SAFE_ADAPTERFUSION_WEIGHTS_NAME,
        )
        self.error_on_missing = error_on_missing

    def filter_func(self, adapter_fusion_name):
        return lambda x: "adapter_fusion_layer.{}".format(adapter_fusion_name) in x

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace(
            "adapter_fusion_layer.{}".format(old_name), "adapter_fusion_layer.{}".format(new_name)
        )

    def save_to_state_dict(self, name: str):
        """
        Extracts the weights of a given AdapterFusion from the model and returns them as a state dict.

        Args:
            name (str): The name of the AdapterFusion to be saved.

        Returns:
            Tuple[dict, dict]: A tuple consisting of the state dict containing the AdapterFusion weights and the
            AdapterFusion configuration.
        """
        if name not in self.model.adapters_config.fusions:
            raise ValueError(f"No AdapterFusion with name '{name}' available.")

        adapter_fusion_config, _ = self.model.adapters_config.get_fusion(name)

        config_dict = build_full_config(
            adapter_fusion_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save adapter weights
        filter_func = self.filter_func(name)
        state_dict = self.weights_helper.state_dict(filter_func)

        return state_dict, config_dict

    def save(self, save_directory: str, name: str, meta_dict=None):
        """
        Saves a AdapterFusion module into the given directory.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str, optional): The AdapterFusion name.
        """

        if name not in self.model.adapters_config.fusions:
            if self.error_on_missing:
                raise ValueError(f"Unknown AdapterFusion '{name}'.")
            else:
                logger.debug(f"No AdapterFusion with name '{name}' available.")
                return

        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the head can be saved."

        adapter_fusion_config, adapter_names = self.model.adapters_config.get_fusion(name)

        # Save the adapter fusion configuration
        config_dict = build_full_config(
            adapter_fusion_config,
            self.model.config,
            name=name,
            adapter_names=adapter_names,
            model_name=self.model.model_name,
            model_class=self.model.__class__.__name__,
        )
        self.weights_helper.save_weights_config(save_directory, config_dict, meta_dict=meta_dict)

        # Save head weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load_from_state_dict(self, state_dict, name, load_as=None, loading_info=None, start_prefix=""):
        """
        Loads the weights of a given AdapterFusion module from a state dict into the model.

        Args:
            state_dict (dict): The state dict from which to load the AdapterFusion weights.
            name (str): The name of the AdapterFusion to be loaded.
            load_as (str, optional):
                Load the AdapterFusion using this name. By default, the name with which the AdapterFusion was saved
                will be used.
            loading_info (dict, optional):
                A dictionary to which loading information (missing and unexpected keys) will be added.
            start_prefix (str, optional): A custom prefix to be ignored in the given state dict.
        """
        new_adapter_fusion_name = load_as or name
        if new_adapter_fusion_name not in self.model.adapters_config.fusions:
            raise ValueError(f"No AdapterFusion with name '{new_adapter_fusion_name}' available.")

        # Load adapter weights
        filter_func = self.filter_func(name)
        rename_func = self.rename_func(name, new_adapter_fusion_name)
        self.weights_helper.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
            start_prefix=start_prefix,
        )

    def load(self, save_directory, load_as=None, loading_info=None, **kwargs):
        """
        Loads a AdapterFusion module from the given directory.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        if not exists(join(save_directory, ADAPTERFUSION_WEIGHTS_NAME)) and not exists(
            join(save_directory, SAFE_ADAPTERFUSION_WEIGHTS_NAME)
        ):
            if self.error_on_missing:
                raise ValueError("Loading path should be a directory where AdapterFusion is saved.")
            else:
                logger.debug("No matching adapter fusion found in '{}'".format(save_directory))
                return None, None

        config = self.weights_helper.load_weights_config(save_directory)

        adapter_fusion_name = load_as or config["name"]
        adapter_names = config.get("adapter_names", adapter_fusion_name)
        if adapter_fusion_name not in self.model.adapters_config.fusions:
            self.model.add_adapter_fusion(
                adapter_names,
                config["config"],
                name=adapter_fusion_name,
                overwrite_ok=True,
                set_active=kwargs.pop("set_active", True),
            )
        else:
            logger.warning("Overwriting existing adapter fusion module '{}'".format(adapter_fusion_name))

        # Load AdapterFusion weights
        filter_func = self.filter_func(adapter_fusion_name)
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_func, loading_info=loading_info
        )

        return save_directory, adapter_fusion_name


class PredictionHeadLoader(WeightsLoader):
    """
    A class providing methods for saving and loading prediction head modules from the file system.

    Model classes supporting configurable head modules via config files should provide a prediction head dict at
    `model.heads` and a method `add_prediction_head(head_name, config)`.
    """

    def __init__(self, model, error_on_missing=True, convert_to_flex_head=False, use_safetensors: bool = False):
        super().__init__(
            model,
            HEAD_WEIGHTS_NAME,
            HEAD_CONFIG_NAME,
            use_safetensors=use_safetensors,
            safe_weights_name=SAFE_HEAD_WEIGHTS_NAME,
        )
        self.error_on_missing = error_on_missing
        self.convert_to_flex_head = convert_to_flex_head

    def filter_func(self, head_name):
        # ToDo remove this workaround
        if self.model.config.model_type in ["t5", "mt5"]:
            if head_name:
                return (
                    lambda x: not x.startswith("encoder")
                    and not x.startswith("decoder")
                    and not x.startswith("shared")
                    and "heads.{}".format(head_name) in x
                )
            else:
                return (
                    lambda x: not x.startswith("encoder")
                    and not x.startswith("decoder")
                    and not x.startswith("shared")
                )

        if head_name:
            return lambda x: not x.startswith(self.model.base_model_prefix) and "heads.{}".format(head_name) in x
        else:
            return lambda x: not x.startswith(self.model.base_model_prefix)

    def rename_func(self, old_name, new_name):
        return lambda k: k.replace("heads.{}".format(old_name), "heads.{}".format(new_name))

    def save(self, save_directory: str, name: str = None):
        """
        Saves a prediction head module into the given directory.

        Args:
            save_directory (str): The directory to save the weights in.
            name (str, optional): The prediction head name.
        """

        if name:
            if hasattr(self.model, "heads"):
                if name not in self.model.heads:
                    if self.error_on_missing:
                        raise ValueError(f"Unknown head_name '{name}'.")
                    else:
                        logger.debug(f"No prediction head with name '{name}' available.")
                        return
            else:
                # we haven't found a prediction head configuration, so we assume there is only one (unnamed) head
                # (e.g. this is the case if we use a 'classic' Hf model with head)
                # -> ignore the name and go on
                name = None
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(save_directory), "Saving path should be a directory where the head can be saved."

        # if we use a custom head, save it
        if name and hasattr(self.model, "heads"):
            head = self.model.heads[name]
            head_config = head.config
        else:
            head_config = None

        # Save the adapter configuration
        config_dict = build_full_config(
            head_config,
            self.model.config,
            name=name,
            model_name=self.model.model_name,
            model_class=self.model.__class__.__name__,
            save_id2label=True,
        )
        # Add number of labels to config if present
        if head_config is None and hasattr(self.model.config, "num_labels"):
            config_dict["num_labels"] = self.model.config.num_labels

        self.weights_helper.save_weights_config(save_directory, config_dict)

        # Save head weights

        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(self, save_directory, load_as=None, loading_info=None, **kwargs):
        """
        Loads a prediction head module from the given directory.

        Args:
            save_directory (str): The directory from where to load the weights.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[str, str]: A tuple consisting of the local file system directory from which the weights where loaded
            and the name of the loaded weights.
        """
        if not exists(join(save_directory, HEAD_WEIGHTS_NAME)) and not exists(
            join(save_directory, SAFE_HEAD_WEIGHTS_NAME)
        ):
            if self.error_on_missing:
                raise ValueError("Loading path should be a directory where the head is saved.")
            else:
                logger.info("No matching prediction head found in '{}'".format(save_directory))
                return None, None
        # label2id map used to override default
        custom_id2label = kwargs.pop("id2label", None)
        if custom_id2label:
            custom_label2id = {label: id_ for id_, label in custom_id2label.items()}
        else:
            custom_label2id = None

        head_name = None
        conversion_rename_func = None

        # Load head config if available - otherwise just blindly try to load the weights
        if isfile(join(save_directory, HEAD_CONFIG_NAME)):
            config = self.weights_helper.load_weights_config(save_directory)
            # make sure that the model class of the loaded head matches the current class
            if not self.convert_to_flex_head and self.model.__class__.__name__ != config["model_class"]:
                error_msg = (
                    f"Model class '{config['model_class']}' of found prediction head does not match current model"
                    " class."
                )
                if self.error_on_missing:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)
                    return None, None

            # model with flex heads
            if hasattr(self.model, "heads"):
                # load head of same model class, no conversion needed
                if self.model.__class__.__name__ == config["model_class"]:
                    head_name = load_as or config["name"]
                    head_config = config["config"]
                elif config["model_class"].endswith("ModelWithHeads"):
                    this_class = self.model.__class__.__name__.replace("AdapterModel", "")
                    other_class = config["model_class"].replace("ModelWithHeads", "")
                    if this_class == other_class:
                        head_name = load_as or config["name"]
                        head_config = config["config"]
                    else:
                        raise ValueError(
                            f"Cannot automatically convert prediction head of model class {config['model_class']} to"
                            " flex head."
                        )
                # try to convert a static head to a flex head
                elif self.convert_to_flex_head and config["model_class"] in STATIC_TO_FLEX_HEAD_MAP:
                    head_name = kwargs.pop("main_load_name", load_as)
                    if head_name is None:
                        raise ValueError(
                            "Could not identify a name for the prediction head to be loaded. Please specify 'load_as'."
                        )
                    head_config, conversion_rename_func = get_head_config_and_rename_list(
                        config["model_class"],
                        head_name,
                        custom_label2id or config.get("label2id"),
                        num_labels=config.get("num_labels"),
                    )
                else:
                    raise ValueError(
                        f"Cannot automatically convert prediction head of model class {config['model_class']} to flex"
                        " head."
                    )
                if head_name in self.model.heads:
                    logger.warning("Overwriting existing head '{}'".format(head_name))

                # make sure the label2id map is correct
                custom_label2id = custom_label2id or head_config.get("label2id", None)
                if custom_label2id:
                    head_config["id2label"] = {int(id_): label for label, id_ in custom_label2id.items()}
                    head_config["label2id"] = {label: int(id_) for label, id_ in custom_label2id.items()}

                self.model.add_prediction_head_from_config(
                    head_name, head_config, overwrite_ok=True, set_active=kwargs.pop("set_active", True)
                )
            # model with static head
            else:
                if self.convert_to_flex_head:
                    raise ValueError("Cannot set convert_flex_head on model class with static head.")
                custom_label2id = custom_label2id or config.get("label2id", None)
                if custom_label2id:
                    self.model.config.id2label = {int(id_): label for label, id_ in custom_label2id.items()}
                    self.model.config.label2id = {label: int(id_) for label, id_ in custom_label2id.items()}

        # Load head weights
        filter_func = self.filter_func(head_name)
        rename_funcs = []
        if load_as:
            rename_funcs.append(self.rename_func(config["name"], load_as))
        if conversion_rename_func:
            rename_funcs.append(conversion_rename_func)
        self.weights_helper.load_weights(
            save_directory, filter_func, rename_func=rename_funcs, loading_info=loading_info
        )

        return save_directory, head_name

    def convert_static_to_flex_head(self, state_dict, load_as="default"):
        """
        Loads a prediction head module from the given state dict, which contains a static head checkpoint.

        Args:
            state_dict (dict): The static head checkpoint from which to load the head module. Can be None.
            load_as (str, optional): Load the weights with this name. Defaults to None.

        Returns:
            Tuple[dict, dict]: A tuple consisting of the head config and the state dict of the loaded weights.
        """
        assert self.convert_to_flex_head, "load_from_state_dict() can only be used with convert_to_flex_head=True."
        assert hasattr(self.model, "heads"), "load_from_state_dict() can only be used with flex heads model class."

        if state_dict is None:
            return None, None

        conversion_rename_func = None

        original_model_class = self.model.config.architectures[0] if self.model.config.architectures else None
        if original_model_class in STATIC_TO_FLEX_HEAD_MAP:
            head_config, conversion_rename_func = get_head_config_and_rename_list(
                original_model_class,
                load_as,
                getattr(self.model.config, "label2id"),
            )
        elif self.error_on_missing:
            raise ValueError(
                f"Cannot automatically convert prediction head of model class {original_model_class} to flex head."
            )
        else:
            return None, None

        # Load head weights
        if state_dict is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = conversion_rename_func(k)
                new_state_dict[new_k] = v
        else:
            new_state_dict = None
        return head_config, new_state_dict
