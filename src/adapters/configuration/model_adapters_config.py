import copy
import logging
from collections.abc import Collection, Mapping
from typing import List, Optional, Tuple, Union

from .. import __version__
from ..composition import AdapterCompositionBlock
from ..utils import get_adapter_config_hash
from .adapter_config import ADAPTER_CONFIG_MAP, DEFAULT_ADAPTER_CONFIG, AdapterConfig, ConfigUnion
from .adapter_fusion_config import ADAPTERFUSION_CONFIG_MAP, DEFAULT_ADAPTERFUSION_CONFIG


logger = logging.getLogger(__name__)


class ModelAdaptersConfig(Collection):
    """This class manages the setup and configuration of adapter modules in a pre-trained model."""

    def __init__(self, **kwargs):
        adapters_list = kwargs.pop("adapters", {})
        # this is for backwards compability: in v1.x, self.adapters values had shape (<type>, <config_name>)
        adapters_list = dict(
            map(lambda t: (t[0], t[1][1] or t[1][0] if isinstance(t[1], tuple) else t[1]), adapters_list.items())
        )
        self.adapters: Mapping[str, str] = adapters_list
        self.config_map = kwargs.pop("config_map", {})

        self.fusions: Mapping[str, str] = kwargs.pop("fusions", {})
        self.fusion_config_map = kwargs.pop("fusion_config_map", {})
        self.fusion_name_map = kwargs.pop("fusion_name_map", {})

        # TODO-V2 Save this with config?
        self.active_setup: Optional[AdapterCompositionBlock] = None
        self.skip_layers = None

    def __contains__(self, item):
        return item in self.adapters.keys()

    def __iter__(self):
        return iter(self.adapters)

    def __len__(self):
        return len(self.adapters)

    def get(self, adapter_name: str) -> Optional[dict]:
        """
        Gets the config dictionary for a given adapter.

        Args:
            adapter_name (str): The name of the adapter.

        Returns:
            Mapping: The adapter configuration.
        """
        if adapter_name in self.adapters:
            config_name = self.adapters[adapter_name]
            if config_name in self.config_map:
                config = self.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config = None
        return config

    def match(
        self,
        adapter_name: str,
        config_type: type,
        layer_idx: Optional[int] = None,
        location_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Tries to match the given criteria to an existing adapter. Return the adapter config if a match is found,
        otherwise None.
        """
        config = self.get(adapter_name)
        if config is None:
            return None
        elif not isinstance(config, AdapterConfig):
            config = AdapterConfig.load(config)

        if isinstance(config, config_type):
            leave_out = config.get("leave_out", [])
            if layer_idx is None or layer_idx not in leave_out:
                if location_key is None or config.get(location_key, False):
                    return config
        # if we have a config union, match with all child configs
        elif isinstance(config, ConfigUnion):
            results = []
            for c in config.configs:
                if isinstance(c, config_type):
                    leave_out = c.get("leave_out", [])
                    if layer_idx is None or layer_idx not in leave_out:
                        if location_key is None or c.get(location_key, False):
                            results.append(c)
            if len(results) == 1:
                return results[0]
            elif len(results) > 1:
                raise ValueError(
                    "Multiple adapter definitions conflict for adapter '{}' in layer {}. "
                    "Please make sure there is only one adaptation block used per location and adapter.".format(
                        adapter_name, layer_idx
                    )
                )

        return None

    def add(self, adapter_name: str, config: Optional[Union[str, dict]] = None):
        """
        Adds a new adapter of the name to the model config.

        Args:
            adapter_name (str): The name of the adapter.
            config (Optional[Union[str, dict]], optional): The adapter config. Defaults to None.
        """
        if adapter_name in self.adapters:
            raise ValueError(f"An adapter with the name '{adapter_name}' has already been added.")
        if config is None:
            config = DEFAULT_ADAPTER_CONFIG
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = AdapterConfig.load(config)
        else:
            raise ValueError("Invalid adapter config: {}".format(config))
        self.adapters[adapter_name] = config_name
        logger.info(f"Adding adapter '{adapter_name}'.")

    def get_fusion(self, fusion_name: Union[str, List[str]]) -> Tuple[Optional[dict], Optional[list]]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
            Optional[list]: The names of the adapters to fuse.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = ADAPTERFUSION_CONFIG_MAP.get(config_name, None)

            if fusion_name in self.fusion_name_map:
                adapter_names = self.fusion_name_map[fusion_name]
            else:
                adapter_names = fusion_name.split(",")

            return config, adapter_names
        else:
            return None, None

    def add_fusion(
        self, adapter_names: List[str], config: Optional[Union[str, dict]] = None, fusion_name: Optional[str] = None
    ):
        """
        Adds a new AdapterFusion.

        Args:
            adapter_names (List[str]): The names of the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
            fusion_name (Optional[str], optional): The name of the AdapterFusion. If not specified, will default to comma-separated adapter names.
        """
        if fusion_name is None:
            fusion_name = ",".join(adapter_names)
        else:
            self.fusion_name_map[fusion_name] = adapter_names
        if fusion_name in self.fusions:
            raise ValueError(f"An AdapterFusion with the name '{fusion_name}' has already been added.")
        if config is None:
            config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if config not in ADAPTERFUSION_CONFIG_MAP and config not in self.fusion_config_map:
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding AdapterFusion '{fusion_name}'.")

    def common_config_value(self, adapter_names: list, attribute: str):
        """
        Checks whether all adapters in a list share the same config setting for a given attribute and returns the
        shared value.

        Args:
            adapter_names (list): The adapters to check.
            attribute (str): The config attribute to check.
        """
        common_value = None
        for i, name in enumerate(adapter_names):
            config = self.get(name)
            if not config:
                raise ValueError(
                    f"No adapter with name '{name}' found. Make sure that an adapter with this name is loaded."
                )
            config_value = config.get(attribute, None)
            if i > 0 and config_value != common_value:
                raise ValueError(f"All given adapters must define the same value for config attribute {attribute}.")
            common_value = config_value
        return common_value

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        output_dict["config_map"] = {}
        for k, v in self.config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["config_map"][k] = v.to_dict()
            else:
                output_dict["config_map"][k] = copy.deepcopy(v)
        output_dict["fusions"] = copy.deepcopy(self.fusions)
        output_dict["fusion_config_map"] = {}
        for k, v in self.fusion_config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["fusion_config_map"][k] = v.to_dict()
            else:
                output_dict["fusion_config_map"][k] = copy.deepcopy(v)
        output_dict["fusion_name_map"] = copy.deepcopy(self.fusion_name_map)
        return output_dict

    def __eq__(self, other):
        return isinstance(other, ModelAdaptersConfig) and (self.__dict__ == other.__dict__)


def build_full_config(adapter_config, model_config, save_id2label=False, **kwargs):
    config_dict = {
        "model_type": model_config.model_type,
        # some models such as encoder-decoder don't have a model-wide hidden size
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    config_dict.update(kwargs)
    if not hasattr(model_config, "prediction_heads") and save_id2label:
        config_dict["label2id"] = model_config.label2id
    if isinstance(adapter_config, AdapterConfig):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    # add lib name before version to distinguish from adapter-transformers
    config_dict["version"] = "adapters." + __version__
    return config_dict
