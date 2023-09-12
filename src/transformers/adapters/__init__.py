# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The Adapter-Hub Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "3.2.1"

from typing import TYPE_CHECKING

from ..utils import _LazyModule


_import_structure = {
    "composition": [
        "AdapterCompositionBlock",
        "BatchSplit",
        "Fuse",
        "Parallel",
        "Split",
        "Stack",
        "parse_composition",
        "validate_composition",
    ],
    "configuration": [
        "ADAPTER_CONFIG_MAP",
        "ADAPTERFUSION_CONFIG_MAP",
        "DEFAULT_ADAPTER_CONFIG",
        "DEFAULT_ADAPTERFUSION_CONFIG",
        "AdapterConfig",
        "AdapterConfigBase",
        "AdapterFusionConfig",
        "CompacterConfig",
        "CompacterPlusPlusConfig",
        "ConfigUnion",
        "DynamicAdapterFusionConfig",
        "HoulsbyConfig",
        "HoulsbyInvConfig",
        "IA3Config",
        "LoRAConfig",
        "MAMConfig",
        "ModelAdaptersConfig",
        "ParallelConfig",
        "PfeifferConfig",
        "PfeifferInvConfig",
        "PrefixTuningConfig",
        "StaticAdapterFusionConfig",
        "UniPELTConfig",
    ],
    "context": [
        "AdapterSetup",
        "ForwardContext",
    ],
    "heads": [
        "BertStyleMaskedLMHead",
        "BiaffineParsingHead",
        "CausalLMHead",
        "ClassificationHead",
        "DependencyParsingOutput",
        "ModelWithFlexibleHeadsAdaptersMixin",
        "MultiHeadOutput",
        "MultiLabelClassificationHead",
        "MultipleChoiceHead",
        "PredictionHead",
        "QuestionAnsweringHead",
        "Seq2SeqLMHead",
        "TaggingHead",
    ],
    "layer": ["AdapterLayer", "AdapterLayerBase"],
    "model_mixin": [
        "EmbeddingAdaptersMixin",
        "InvertibleAdaptersMixin",
        "InvertibleAdaptersWrapperMixin",
        "ModelAdaptersMixin",
        "ModelWithHeadsAdaptersMixin",
    ],
    "models.albert": ["AlbertAdapterModel"],
    "models.auto": [
        "ADAPTER_MODEL_MAPPING",
        "MODEL_WITH_HEADS_MAPPING",
        "AutoAdapterModel",
        "AutoModelWithHeads",
    ],
    "models.bart": [
        "BartAdapterModel",
        "BartModelWithHeads",
    ],
    "models.beit": ["BeitAdapterModel"],
    "models.bert": [
        "BertAdapterModel",
        "BertModelWithHeads",
    ],
    "models.bert_generation": ["BertGenerationAdapterModel"],
    "models.deberta": ["DebertaAdapterModel"],
    "models.debertaV2": ["DebertaV2AdapterModel"],
    "models.distilbert": [
        "DistilBertAdapterModel",
        "DistilBertModelWithHeads",
    ],
    "models.gpt2": [
        "GPT2AdapterModel",
        "GPT2ModelWithHeads",
    ],
    "models.gptj": ["GPTJAdapterModel"],
    "models.hubert": [
        "HubertAdapterModel",
        "HubertModelWithHeads",
    ],
    "models.mbart": [
        "MBartAdapterModel",
        "MBartModelWithHeads",
    ],
    "models.roberta": [
        "RobertaAdapterModel",
        "RobertaModelWithHeads",
    ],
    "models.t5": [
        "T5AdapterModel",
        "T5ModelWithHeads",
    ],
    "models.vit": ["ViTAdapterModel"],
    "models.xlm_roberta": [
        "XLMRobertaAdapterModel",
        "XLMRobertaModelWithHeads",
    ],
    "trainer": ["AdapterTrainer", "Seq2SeqAdapterTrainer"],
    "training": [
        "AdapterArguments",
        "setup_adapter_training",
    ],
    "utils": [
        "ADAPTER_CACHE",
        "AdapterInfo",
        "AdapterType",
        "get_adapter_config_hash",
        "get_adapter_info",
        "list_adapters",
    ],
}


if TYPE_CHECKING:
    from .composition import (
        AdapterCompositionBlock,
        BatchSplit,
        Fuse,
        Parallel,
        Split,
        Stack,
        parse_composition,
        validate_composition,
    )
    from .configuration import (
        ADAPTER_CONFIG_MAP,
        ADAPTERFUSION_CONFIG_MAP,
        DEFAULT_ADAPTER_CONFIG,
        DEFAULT_ADAPTERFUSION_CONFIG,
        AdapterConfig,
        AdapterConfigBase,
        AdapterFusionConfig,
        CompacterConfig,
        CompacterPlusPlusConfig,
        ConfigUnion,
        DynamicAdapterFusionConfig,
        HoulsbyConfig,
        HoulsbyInvConfig,
        IA3Config,
        LoRAConfig,
        MAMConfig,
        ModelAdaptersConfig,
        ParallelConfig,
        PfeifferConfig,
        PfeifferInvConfig,
        PrefixTuningConfig,
        StaticAdapterFusionConfig,
        UniPELTConfig,
    )
    from .context import AdapterSetup, ForwardContext
    from .heads import (
        BertStyleMaskedLMHead,
        BiaffineParsingHead,
        CausalLMHead,
        ClassificationHead,
        DependencyParsingOutput,
        ModelWithFlexibleHeadsAdaptersMixin,
        MultiHeadOutput,
        MultiLabelClassificationHead,
        MultipleChoiceHead,
        PredictionHead,
        QuestionAnsweringHead,
        Seq2SeqLMHead,
        TaggingHead,
    )
    from .layer import AdapterLayer, AdapterLayerBase
    from .model_mixin import (
        EmbeddingAdaptersMixin,
        InvertibleAdaptersMixin,
        InvertibleAdaptersWrapperMixin,
        ModelAdaptersMixin,
        ModelWithHeadsAdaptersMixin,
    )
    from .models.albert import AlbertAdapterModel
    from .models.auto import ADAPTER_MODEL_MAPPING, MODEL_WITH_HEADS_MAPPING, AutoAdapterModel, AutoModelWithHeads
    from .models.bart import BartAdapterModel, BartModelWithHeads
    from .models.beit import BeitAdapterModel
    from .models.bert import BertAdapterModel, BertModelWithHeads
    from .models.bert_generation import BertGenerationAdapterModel
    from .models.deberta import DebertaAdapterModel
    from .models.debertaV2 import DebertaV2AdapterModel
    from .models.distilbert import DistilBertAdapterModel, DistilBertModelWithHeads
    from .models.gpt2 import GPT2AdapterModel, GPT2ModelWithHeads
    from .models.gptj import GPTJAdapterModel
    from .models.hubert import HubertAdapterModel, HubertModelWithHeads
    from .models.mbart import MBartAdapterModel, MBartModelWithHeads
    from .models.roberta import RobertaAdapterModel, RobertaModelWithHeads
    from .models.t5 import T5AdapterModel, T5ModelWithHeads
    from .models.vit import ViTAdapterModel
    from .models.xlm_roberta import XLMRobertaAdapterModel, XLMRobertaModelWithHeads
    from .trainer import AdapterTrainer, Seq2SeqAdapterTrainer
    from .training import AdapterArguments, setup_adapter_training
    from .utils import (
        ADAPTER_CACHE,
        AdapterInfo,
        AdapterType,
        get_adapter_config_hash,
        get_adapter_info,
        list_adapters,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )
