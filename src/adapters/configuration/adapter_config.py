import logging
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from typing import List, Literal, Optional, Union

from ..utils import resolve_adapter_config


logger = logging.getLogger(__name__)


class AdapterConfig(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.

    Args:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture: Optional[str] = None

    def __init__(self):
        raise TypeError("AdapterConfig is an abstract class and cannot be instantiated.")

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Converts the config class to a Python dict."""
        return asdict(self)

    def replace(self, **changes):
        """Returns a new instance of the config class with the specified changes applied."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        """Creates a config class from a Python dict."""
        if isinstance(config, AdapterConfig):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _get_config_class(config_dict):
        """
        Returns the matching config class for the given config dict based on its "architecture" key.
        """
        architecture = config_dict.get("architecture", None)
        if architecture == "prefix_tuning":
            cls_new = PrefixTuningConfig
        elif architecture == "lora":
            cls_new = LoRAConfig
        elif architecture == "union":
            cls_new = ConfigUnion
        elif architecture == "prompt_tuning":
            cls_new = PromptTuningConfig
        elif architecture == "reft":
            cls_new = ReftConfig
        else:
            cls_new = BnConfig

        return cls_new

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
        """
        Loads a given adapter configuration specifier into a full AdapterConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        if not config:
            return None
        # if force_download is set, skip the local map
        if download_kwargs and download_kwargs.get("force_download", False):
            local_map = None
        else:
            local_map = ADAPTER_CONFIG_MAP
        if download_kwargs:
            config_dict = resolve_adapter_config(config, local_map=local_map, **download_kwargs)
        else:
            config_dict = resolve_adapter_config(config, local_map=local_map)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterConfig):
            cls_new = config_dict.__class__
            config_dict = config_dict.to_dict()
        else:
            cls_new = cls._get_config_class(config_dict)
        # The check for "None" is necessary because of the example script flags.
        config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
        return cls_new.from_dict(config_dict)


@dataclass(eq=False)
class BnConfig(AdapterConfig):
    """
    Base class that models the architecture of a bottleneck adapter.

    Args:
        mh_adapter (:obj:`bool`): If True, add adapter modules after the multi-head attention block of each layer.
        output_adapter (:obj:`bool`): If True, add adapter modules after the output FFN of each layer.
        reduction_factor (:obj:`float` or :obj:`Mapping`):
            Either a scalar float (> 0) specifying the reduction factor for all layers or a mapping from layer ID
            (starting at 0) to values specifying the reduction_factor for individual layers. If not all layers are
            represented in the mapping a default value should be given e.g. {'1': 8, '6': 32, 'default': 16}.
            Specifying a reduction factor < 1 will result in an up-projection layer.
        non_linearity (:obj:`str`): The activation function to use in the adapter bottleneck.
        original_ln_before (:obj:`bool`, optional):
            If True, apply layer pre-trained normalization and residual connection before the adapter modules. Defaults
            to False. Only applicable if :obj:`is_parallel` is False.
        original_ln_after (:obj:`bool`, optional):
            If True, apply pre-trained layer normalization and residual connection after the adapter modules. Defaults
            to True.
        ln_before (:obj:`bool`, optional): If True, add a new layer normalization before the adapter bottleneck.
            Defaults to False.
        ln_after (:obj:`bool`, optional): If True, add a new layer normalization after the adapter bottleneck.
            Defaults to False.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter" or "houlsby".
        is_parallel (:obj:`bool`, optional): If True, apply adapter transformations in parallel.
            By default (False), sequential application is used.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float), or the string "learned", in which case the scaling factor is learned, or the string
            "channel", in which case we initialize a scaling vector of the channel shape that is then learned.
            Defaults to 1.0.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        residual_before_ln (:obj:`bool` or :obj:`str`, optional):
            If True, take the residual connection around the adapter bottleneck before the layer normalization. If set
            to "post_add", take the residual connection around the adapter bottleneck after the previous residual
            connection. Only applicable if :obj:`original_ln_before` is True.
        adapter_residual_before_ln (:obj:`bool`, optional):
            If True, apply the residual connection around the adapter modules before the new layer normalization within
            the adapter. Only applicable if :obj:`ln_after` is True and :obj:`is_parallel` is False.
        inv_adapter (:obj:`str`, optional):
            If not None (default), add invertible adapter modules after the model embedding layer. Currently, this can
            be either "nice" or "glow".
        inv_adapter_reduction_factor (:obj:`float`, optional):
            The reduction to use within the invertible adapter modules. Only applicable if :obj:`inv_adapter` is not
            None.
        cross_adapter (:obj:`bool`, optional):
            If True, add adapter modules after the cross attention block of each decoder layer in an encoder-decoder
            model. Defaults to False.
        leave_out (:obj:`List[int]`, optional):
            The IDs of the layers (starting at 0) where NO adapter modules should be added.
        dropout (:obj:`float`, optional): The dropout rate used in the adapter layer. Defaults to 0.0.
        phm_layer (:obj:`bool`, optional): If True the down and up projection layers are a PHMLayer.
            Defaults to False
        phm_dim (:obj:`int`, optional): The dimension of the phm matrix.
            Only applicable if `phm_layer` is set to `True`. Defaults to 4.
        shared_phm_rule (:obj:`bool`, optional): Whether the phm matrix is shared across all layers.
            Defaults to True
        factorized_phm_rule (:obj:`bool`, optional):
            Whether the phm matrix is factorized into a left and right matrix. Defaults to False.
        learn_phm (:obj:`bool`, optional): Whether the phm matrix should be learned during training.
            Defaults to True
        factorized_phm_W (:
            obj:`bool`, optional): Whether the weights matrix is factorized into a left and right matrix. Defaults to
            True
        shared_W_phm (:obj:`bool`, optional): Whether the weights matrix is shared across all layers.
            Defaults to False.
        phm_c_init (:obj:`str`, optional): The initialization function for the weights of the phm matrix.
            The possible values are `["normal", "uniform"]`. Defaults to `normal`.
        phm_init_range (:obj:`float`, optional): std for initializing phm weights if `phm_c_init="normal"`.
            Defaults to 0.0001.
        hypercomplex_nonlinearity (:obj:`str`, optional):
            This specifies the distribution to draw the weights in the phm layer from. Defaults to `glorot-uniform`.
        phm_rank (:obj:`int`, optional):
            If the weight matrix is factorized this specifies the rank of the matrix. E.g. the left matrix of the down
            projection has the shape (phm_dim, _in_feats_per_axis, phm_rank) and the right matrix (phm_dim, phm_rank,
            _out_feats_per_axis). Defaults to 1
        phm_bias (:obj:`bool`, optional):
            If True the down and up projection PHMLayer has a bias term. If `phm_layer` is False this is ignored.
            Defaults to True
        stochastic_depth (:obj:`float`, optional):
            This value specifies the probability of the model dropping entire layers during
            training. This parameter should be only used for vision based tasks involving
            residual networks.
    """

    # Required options
    mh_adapter: bool
    output_adapter: bool

    reduction_factor: Union[float, Mapping]
    non_linearity: str

    # Options with defaults
    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    init_weights: str = "bert"
    is_parallel: bool = False
    scaling: Union[float, str] = 1.0
    use_gating: bool = False
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    inv_adapter: Optional[str] = None
    inv_adapter_reduction_factor: Optional[float] = None
    cross_adapter: bool = False
    leave_out: List[int] = field(default_factory=list)
    dropout: float = 0.0
    phm_layer: bool = False
    phm_dim: int = 4
    factorized_phm_W: Optional[bool] = True
    shared_W_phm: Optional[bool] = False
    shared_phm_rule: Optional[bool] = True
    factorized_phm_rule: Optional[bool] = False
    phm_c_init: Optional[str] = "normal"
    phm_init_range: Optional[float] = 0.0001
    learn_phm: Optional[bool] = True
    hypercomplex_nonlinearity: Optional[str] = "glorot-uniform"
    phm_rank: Optional[int] = 1
    phm_bias: Optional[bool] = True
    stochastic_depth: Optional[float] = 0.0

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        elif name == "invertible_adapter":
            # This is for backwards compatibility. In v1, invertible adapters were specified in a nested config dict.
            # Now, we have two config keys directly in the adapter config.
            if value:
                object.__setattr__(self, "inv_adapter", value["block_type"])
                object.__setattr__(self, "inv_adapter_reduction_factor", value["reduction_factor"])
        else:
            object.__setattr__(self, name, value)


@dataclass(eq=False)
class SeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class CompacterPlusPlusConfig(SeqBnConfig):
    """
    The Compacter++ architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class SeqBnInvConfig(SeqBnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class DoubleSeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Houlsby et al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = True
    output_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class CompacterConfig(DoubleSeqBnConfig):
    """
    The Compacter architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class DoubleSeqBnInvConfig(DoubleSeqBnConfig):
    """
    The adapter architecture proposed by Houlsby et. al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class ParBnConfig(BnConfig):
    """
    The parallel adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 2

    init_weights: str = "mam_adapter"
    is_parallel: bool = True
    scaling: Union[float, str] = 4.0


@dataclass(eq=False)
class AdapterPlusConfig(BnConfig):
    """
    The AdapterPlus config architecture proposed by Jan-Martin O, Steitz and Stefan Roth. See https://arxiv.org/pdf/2406.06820

    Please note that some configurations of the adapters parameters `original_ln_after`, `original_ln_before`, and
    `residual_before_ln` may result in performance issues when training.

    In the general case:
        1) At least one of `original_ln_before` or `original_ln_after` should be set to True in order to ensure that the original residual
           connection from pre-training is preserved.
        2) If `original_ln_after` is set to `False`, `residual_before_ln` must also be set to `False` to ensure convergence during training.
    """

    original_ln_after: bool = False
    original_ln_before: bool = True
    residual_before_ln: bool = False
    stochastic_depth: float = 0.1
    init_weights: str = "houlsby"
    scaling: Union[float, str] = "channel"

    mh_adapter: bool = False
    output_adapter: bool = True
    reduction_factor: Union[float, Mapping] = 96
    non_linearity: str = "gelu"


@dataclass(eq=False)
class PrefixTuningConfig(AdapterConfig):
    """
    The Prefix Tuning architecture proposed by Li & Liang (2021). See https://arxiv.org/pdf/2101.00190.pdf.

    Args:
        encoder_prefix (bool): If True, add prefixes to the encoder of an encoder-decoder model.
        cross_prefix (bool): If True, add prefixes to the cross attention of an encoder-decoder model.
        flat (bool): If True, train the prefix parameters directly. Otherwise, reparametrize using a bottleneck MLP.
        prefix_length (int): The length of the prefix.
        bottleneck_size (int): If flat=False, the size of the bottleneck MLP.
        non_linearity (str): If flat=False, the non-linearity used in the bottleneck MLP.
        dropout (float): The dropout rate used in the prefix tuning layer.
        leave_out (List[int]): The IDs of the layers (starting at 0) where NO prefix should be added.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        shared_gating (:
            obj:`bool`, optional): Whether to use a shared gate for the prefixes of all attention matrices. Only
            applicable if `use_gating=True`. Defaults to True.
    """

    architecture: Optional[str] = "prefix_tuning"

    encoder_prefix: bool = True
    cross_prefix: bool = True
    leave_out: List[int] = field(default_factory=list)

    flat: bool = False
    prefix_length: int = 30
    bottleneck_size: int = 512
    non_linearity: str = "tanh"
    dropout: float = 0.0
    use_gating: bool = False
    shared_gating: bool = True


@dataclass(eq=False)
class PromptTuningConfig(AdapterConfig):
    """
    The Prompt Tuning architecture proposed by Lester et al. (2021). See https://arxiv.org/pdf/2104.08691.pdf

    Args:
        prompt_length (int): The number of tokens in the prompt.
            Defaults to 10.
        prompt_init (str): The initialization method for the prompt. Can be either "random_uniform" or "from_string".
            Defaults to "random_uniform".
        prompt_init_text (str): The text to use for prompt initialization if prompt_init="from_string".
        random_uniform_scale (float): The scale of the random uniform initialization if prompt_init="random_uniform".
            Defaults to 0.5 as in the paper.
        combine (str):
            The method used to combine the prompt with the input. Can be either "prefix" or "prefix_after_bos".
            Defaults to "prefix".
    """

    architecture: str = "prompt_tuning"

    prompt_length: int = 10
    prompt_init: str = "random_uniform"
    prompt_init_text: Optional[str] = None
    random_uniform_scale = 0.5
    combine: str = "prefix"


@dataclass(eq=False)
class LoRAConfig(AdapterConfig):
    """
    The Low-Rank Adaptation (LoRA) architecture proposed by Hu et al. (2021). See https://arxiv.org/pdf/2106.09685.pdf.
    LoRA adapts a model by reparametrizing the weights of a layer matrix. You can merge the additional weights with the
    original layer weights using ``model.merge_adapter("lora_name")``.

    Args:
        selfattn_lora (bool, optional): If True, add LoRA to the self-attention weights of a model.
            Defaults to True.
        intermediate_lora (bool, optional): If True, add LoRA to the intermediate MLP weights of a model.
            Defaults to False.
        output_lora (bool, optional): If True, add LoRA to the output MLP weights of a model.
            Defaults to False.
        leave_out (:obj:`List[int]`, optional):
            The IDs of the layers (starting at 0) where NO adapter modules should be added.
        r (int, optional): The rank of the LoRA layer. Defaults to 8.
        alpha (int, optional): The hyperparameter used for scaling the LoRA reparametrization. Defaults to 8.
        dropout (float, optional): The dropout rate used in the LoRA layer. Defaults to 0.0.
        attn_matrices (List[str], optional): Determines which matrices of the self-attention module to adapt.
            A list that may contain the strings "q" (query), "k" (key), "v" (value). Defaults to ["q", "v"].
        composition_mode (str, optional):
            Defines how the injected weights are composed with the original model weights. Can be either "add"
            (addition of decomposed matrix, as in LoRA) or "scale" (element-wise multiplication of vector, as in
            (IA)^3). "scale" can only be used together with r=1. Defaults to "add".
        init_weights (:obj:`str`, optional): Initialization method for the weights of the LoRA modules.
            Currently, this can be either "lora" (default) or "bert", or "vera".
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False. Note that modules with use_gating=True cannot be merged using
            `merge_adapter()`.
        vera_d (:obj:`float`, optional):
            The value of d used in the VeraConfig. Defaults to None. Places a trainable
            scaling parameter `d` before the decomposition matrix A to allow scaling of the
            internal weights.

        vera_b (:obj:`float`, optional):
            The value of b used in the VeraConfig. Defaults to None. Places a trainable
            scaling parameter `b` before the decomposition matrix B to allow scaling of the
            internal weights.
        dtype (str, optional): torch dtype for reparametrization tensors. Defaults to None.
    """

    architecture: Optional[str] = "lora"

    selfattn_lora: bool = True
    intermediate_lora: bool = False
    output_lora: bool = False
    leave_out: List[int] = field(default_factory=list)

    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["q", "v"])
    composition_mode: str = "add"
    init_weights: str = "lora"
    use_gating: bool = False
    vera_d: float = None
    vera_b: float = None
    dtype: Optional[str] = None


@dataclass(eq=False)
class IA3Config(LoRAConfig):
    """
    The 'Infused Adapter by Inhibiting and Amplifying Inner Activations' ((IA)^3) architecture proposed by Liu et al.
    (2022). See https://arxiv.org/pdf/2205.05638.pdf. (IA)^3 builds on top of LoRA, however, unlike the additive
    composition of LoRA, it scales weights of a layer using an injected vector.
    """

    selfattn_lora: bool = True
    intermediate_lora: bool = True
    output_lora: bool = False
    leave_out: List[int] = field(default_factory=list)

    r: int = 1
    alpha: int = 1
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["k", "v"])
    composition_mode: str = "scale"
    init_weights: str = "ia3"
    use_gating: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class VeraConfig(LoRAConfig):
    """
    Lora Config that applies vector-based random matrix adaptation. It adds
    trainable matrices 'd' and 'b' while keeping the original LoRA matrices
    frozen, random, and shared across layers. See more through their paper:
    https://arxiv.org/pdf/2310.11454. Note that `r` will still be supplied
    since we are still initializing decomposition matrices A and B.
    The `composition_mode` parameter should also be set to `add`.
    """

    selfattn_lora: bool = True
    intermediate_lora: bool = False
    output_lora: bool = False

    r: int = 8
    vera_d: float = 0.1
    vera_b: float = 0.0
    init_weights: str = "vera"
    composition_mode: str = "add"
    dtype: Optional[str] = None


@dataclass(eq=False)
class ReftConfig(AdapterConfig):
    """
    Base class for Representation Fine-Tuning (ReFT) methods proposed in Wu et al. (2024). See https://arxiv.org/pdf/2404.03592.
    ReFT methods have in common that they add "interventions" after selected model layers and at selected sequence positions to adapt the representations produced by module outputs.

    Args:
        layers (Union[Literal["all"], List[int]]): The IDs of the layers where interventions should be added.
            If "all", interventions are added after all layers (default).
        prefix_positions (int): The number of prefix positions to add interventions to.
        suffix_positions (int): The number of suffix positions to add interventions to.
        r (int): The rank of the intervention layer.
        orthogonality (bool): If True, enforce an orthogonality constraint for the projection matrix.
        tied_weights (bool): If True, share intervention parameters between prefix and suffix positions in each layer.
        subtract_projection (bool): If True, subtract the projection of the input.
        dropout (float): The dropout rate used in the intervention layer.
        non_linearity (str): The activation function used in the intervention layer.
        dtype (str, optional): torch dtype for intervention tensors. Defaults to None.
    """

    layers: Union[Literal["all"], List[int]]
    prefix_positions: int
    suffix_positions: int
    r: int
    orthogonality: bool
    tied_weights: bool = False
    subtract_projection = True
    dropout: float = 0.05
    non_linearity: Optional[str] = None
    dtype: Optional[str] = None

    architecture: str = "reft"

    output_reft: bool = True


@dataclass(eq=False)
class LoReftConfig(ReftConfig):
    """
    Low-Rank Linear Subspace ReFT method proposed in Wu et al. (2024). See https://arxiv.org/pdf/2404.03592.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = True
    tied_weights: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class NoReftConfig(ReftConfig):
    """
    Variation of LoReft without orthogonality constraint.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = False
    tied_weights: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class DiReftConfig(ReftConfig):
    """
    Variation of LoReft without orthogonality constraint and projection subtraction as proposed in Wu et al. (2024). See https://arxiv.org/pdf/2404.03592.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = False
    tied_weights: bool = False
    subtract_projection = False
    dtype: Optional[str] = None


class ConfigUnion(AdapterConfig):
    """
    Composes multiple adaptation method configurations into one. This class can be used to define complex adaptation
    method setups.
    """

    architecture: Optional[str] = "union"

    configs: List[AdapterConfig]

    def __init__(self, *configs: List[AdapterConfig]):
        self.validate(configs)
        self.configs = configs

    @staticmethod
    def validate(configs):
        """
        Performs simple validations of a list of configurations to check whether they can be combined to a common
        setup.

        Args:
            configs (List[AdapterConfig]): list of configs to check.

        Raises:
            TypeError: One of the configurations has a wrong type. ValueError: At least two given configurations
            conflict.
        """
        # perform single config checks
        for config in configs:
            if not isinstance(config, AdapterConfig):
                raise TypeError(f"{config} is not an instance of AdapterConfig")
            elif isinstance(config, ConfigUnion):
                raise TypeError(f"{config} of type {type(config)} is not supported in a config union.")
        # perform pairwise check
        for c_a, c_b in [(c_a, c_b) for i, c_a in enumerate(configs) for j, c_b in enumerate(configs) if i > j]:
            if c_a.architecture != c_b.architecture:
                continue
            # if at least one config specifies a leave_out, we cannot make a final decision at this point
            elif c_a.get("leave_out", []) or c_b.get("leave_out", []):
                continue
            elif c_a.architecture is None or c_a.architecture == "bottleneck":
                is_valid = c_a.mh_adapter != c_b.mh_adapter and c_a.output_adapter != c_b.output_adapter
                if not is_valid:
                    raise ValueError(f"{c_a} and {c_b} cannot be combined.")
                else:
                    continue
            # at this point, we know that the architectures are the same
            raise ValueError(f"{c_a} and {c_b} have the same adapter architecture and cannot be combined.")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.configs[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            i, k = key.split(".")
            return self.configs[int(i)][k]

    def __iter__(self):
        for i, c in enumerate(self.configs):
            for k in iter(c):
                yield f"{i}.{k}"

    def __len__(self):
        return sum([len(c) for c in self.configs])

    def __eq__(self, other):
        return all([c_a == c_b for c_a, c_b in zip(self.configs, other.configs)])

    def to_dict(self):
        return {"architecture": self.architecture, "configs": [c.to_dict() for c in self.configs]}

    def replace(self, **changes):
        return ConfigUnion(*[c.replace(**changes) for c in self.configs])

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, AdapterConfig):
            return config

        configs = []
        for c in config["configs"]:
            config_class = cls._get_config_class(c)
            configs.append(config_class.from_dict(c))

        return cls(*configs)


class MAMConfig(ConfigUnion):
    """
    The Mix-And-Match adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    def __init__(self, prefix_tuning: Optional[PrefixTuningConfig] = None, adapter: Optional[BnConfig] = None):
        prefix_tuning = prefix_tuning or PrefixTuningConfig(bottleneck_size=800)
        adapter = adapter or ParBnConfig()

        assert isinstance(prefix_tuning, PrefixTuningConfig)
        assert isinstance(adapter, BnConfig)
        super().__init__(prefix_tuning, adapter)

    @property
    def prefix_tuning(self):
        return self[0]

    @property
    def adapter(self):
        return self[1]


class UniPELTConfig(ConfigUnion):
    """
    The UniPELT adapter architecture proposed by Mao et al. (2022). See https://arxiv.org/pdf/2110.07577.pdf.
    """

    def __init__(
        self,
        prefix_tuning: Optional[PrefixTuningConfig] = None,
        adapter: Optional[BnConfig] = None,
        lora: Optional[LoRAConfig] = None,
    ):
        components = [
            prefix_tuning or PrefixTuningConfig(prefix_length=10),
            adapter or SeqBnConfig(reduction_factor=16),
            lora or LoRAConfig(r=8, alpha=2),
        ]

        super().__init__(*[c.replace(use_gating=True) for c in components])


# IMPORTANT: When adding a new config here, also add it to docs/overview.md!
ADAPTER_CONFIG_MAP = {
    # DEPRECATED STRINGS
    "pfeiffer": SeqBnConfig(),
    "houlsby": DoubleSeqBnConfig(),
    "parallel": ParBnConfig(),
    "scaled_parallel": ParBnConfig(scaling="learned"),
    "pfeiffer+inv": SeqBnInvConfig(),
    "houlsby+inv": DoubleSeqBnInvConfig(),
    # CURRENT STRINGS
    "seq_bn": SeqBnConfig(),
    "double_seq_bn": DoubleSeqBnConfig(),
    "par_bn": ParBnConfig(),
    "scaled_par_bn": ParBnConfig(scaling="learned"),
    "seq_bn_inv": SeqBnInvConfig(),
    "double_seq_bn_inv": DoubleSeqBnInvConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "compacter": CompacterConfig(),
    "prefix_tuning": PrefixTuningConfig(),
    "prefix_tuning_flat": PrefixTuningConfig(flat=True),
    "prompt_tuning": PromptTuningConfig(),
    "lora": LoRAConfig(),
    "ia3": IA3Config(),
    "loreft": LoReftConfig(),
    "noreft": NoReftConfig(),
    "direft": DiReftConfig(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig(),
}

DEFAULT_ADAPTER_CONFIG = "seq_bn"
