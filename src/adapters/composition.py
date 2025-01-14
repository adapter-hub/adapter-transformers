import itertools
import sys
import warnings
from collections.abc import Sequence
from typing import List, Optional, Set, Tuple, Union

import torch


class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return all([c1 == c2 for c1, c2 in zip(self.children, o.children)])
        else:
            return False

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    @property
    def parallel_channels(self):
        return max([b.parallel_channels if isinstance(b, AdapterCompositionBlock) else 1 for b in self.children])

    def flatten(self) -> Set[str]:
        return set(itertools.chain(*[[b] if isinstance(b, str) else b.flatten() for b in self.children]))

    def _get_save_kwargs(self):
        return None

    def to_dict(self):
        save_dict = {
            "type": self.__class__.__name__,
            "children": [
                c.to_dict() if isinstance(c, AdapterCompositionBlock) else {"type": "single", "children": [c]}
                for c in self.children
            ],
        }
        if kwargs := self._get_save_kwargs():
            save_dict["kwargs"] = kwargs
        return save_dict

    @classmethod
    def from_dict(cls, data):
        children = []
        for child in data["children"]:
            if child["type"] == "single":
                children.append(child["children"][0])
            else:
                children.append(cls.from_dict(child))
        return getattr(sys.modules[__name__], data["type"])(*children, **data.get("kwargs", {}))


class Parallel(AdapterCompositionBlock):
    def __init__(self, *parallel_adapters: List[str]):
        """
        Can be used to perform inference for multiple tasks (i.e., adapters) in parallel (for the same input).

        See AdapterDrop https://arxiv.org/abs/2010.11918
        """
        super().__init__(*parallel_adapters)

    @property
    def parallel_channels(self):
        return len(self.children)


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(self, *fuse_stacks: List[Union[AdapterCompositionBlock, str]], name: Optional[str] = None):
        super().__init__(*fuse_stacks)
        self._name = name

    # TODO-V2 pull this up to all block classes?
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return ",".join([c if isinstance(c, str) else c.last() for c in self.children])


class Split(AdapterCompositionBlock):
    def __init__(self, *split_adapters: List[Union[AdapterCompositionBlock, str]], splits: Union[List[int], int]):
        super().__init__(*split_adapters)
        self.splits = splits if isinstance(splits, list) else [splits] * len(split_adapters)

    def _get_save_kwargs(self):
        return {"splits": self.splits}


class BatchSplit(AdapterCompositionBlock):
    def __init__(self, *split_adapters: List[Union[AdapterCompositionBlock, str]], batch_sizes: Union[List[int], int]):
        super().__init__(*split_adapters)
        self.batch_sizes = batch_sizes if isinstance(batch_sizes, list) else [batch_sizes] * len(split_adapters)

    def _get_save_kwargs(self):
        return {"batch_sizes": self.batch_sizes}


class Average(AdapterCompositionBlock):
    def __init__(
        self,
        *average_adapters: List[Union[AdapterCompositionBlock, str]],
        weights: Optional[List[float]] = None,
        normalize_weights: bool = True,
    ):
        super().__init__(*average_adapters)
        if weights is not None:
            # normalize weights
            if normalize_weights:
                sum_weights = sum(weights) if weights else 1
                self.weights = [w / sum_weights for w in weights]
            else:
                self.weights = weights
        else:
            self.weights = [1 / len(average_adapters)] * len(average_adapters)

    def _get_save_kwargs(self):
        return {"weights": self.weights}


# Mapping each composition block type to the allowed nested types
ALLOWED_NESTINGS = {
    Stack: [str, Fuse, Split, Parallel, BatchSplit, Average],
    Fuse: [str, Stack],
    Split: [str, Split, Stack, BatchSplit, Average],
    Parallel: [str, Stack, BatchSplit, Average],
    BatchSplit: [str, Stack, Split, BatchSplit, Average],
    Average: [str, Stack, Split, BatchSplit],
}

# Some composition blocks might not be supported by all models.
# Add a whitelist of models for those here.
SUPPORTED_MODELS = {
    Parallel: [
        "albert",
        "bert",
        "roberta",
        "distilbert",
        "deberta-v2",
        "deberta",
        "bart",
        "mbart",
        "mt5",
        "plbart",
        "gpt2",
        "gptj",
        "t5",
        "vit",
        "xlm-roberta",
        "bert-generation",
        "llama",
        "mistral",
        "electra",
        "whisper",
        "xmod",
    ],
}


def validate_composition(adapter_composition: AdapterCompositionBlock, level=0, model_type=None):
    if level > 1 and not (isinstance(adapter_composition, Stack) or isinstance(adapter_composition, str)):
        raise ValueError(f"Adapter setup is too deep. Cannot have {adapter_composition} at level {level}.")
    if isinstance(adapter_composition, AdapterCompositionBlock):
        block_type = type(adapter_composition)
        if model_type and block_type in SUPPORTED_MODELS:
            if model_type not in SUPPORTED_MODELS[block_type]:
                raise ValueError(
                    f"Models of type {model_type} don't support adapter composition using {block_type.__name__}."
                )
        for child in adapter_composition:
            if not type(child) in ALLOWED_NESTINGS[type(adapter_composition)]:
                raise ValueError(f"Adapter setup is invalid. Cannot nest {child} in {adapter_composition}")
            # recursively validate children
            validate_composition(child, level=level + 1)


def parse_composition(adapter_composition, level=0, model_type=None) -> AdapterCompositionBlock:
    """
    Parses and validates a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        level (int, optional): If set to none, disables validation. Defaults to 0.
    """
    if not adapter_composition:
        return None
    elif isinstance(adapter_composition, AdapterCompositionBlock):
        if level is not None:
            validate_composition(adapter_composition, level=level, model_type=model_type)
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # Functionality of adapter-transformers v1.x
        warnings.warn(
            "Passing list objects for adapter activation is deprecated. Please use Stack or Fuse explicitly.",
            category=FutureWarning,
        )
        # for backwards compatibility
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)


def parse_heads_from_composition(adapter_composition, reference_heads: list = None):
    """
    Parses a potential head configuration from a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        reference_heads: The list of available to validate the retrieved head configuration against.
    """
    final_block = adapter_composition
    if isinstance(final_block, Stack):
        final_block = final_block.children[-1]

    if isinstance(final_block, str) and (reference_heads is None or final_block in reference_heads):
        return final_block
    elif isinstance(final_block, Parallel):
        return [a if isinstance(a, str) else a.last() for a in final_block.children]
    elif isinstance(final_block, BatchSplit):
        # Convert BatchSplit of adapters to a BatchSplit of heads.
        blocks = [block.last() if isinstance(block, AdapterCompositionBlock) else block for block in final_block]
        head_setup = BatchSplit(*blocks, batch_sizes=final_block.batch_sizes)
        if reference_heads is None or all(head in reference_heads for head in head_setup):
            return head_setup
        else:
            raise ValueError(
                "Missing at least one head for the given BatchSplit setup. Expected heads: {}".format(blocks)
            )
    else:
        return None


def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)


def adjust_tensors_for_parallel_(hidden_states, *tensors):
    """
    In-place version of adjust_tensors_for_parallel().
    """
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            tensor.set_(new_tensor)


def match_attn_matrices_for_parallel(query, key, value) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches the shapes of query, key and value matrices for parallel composition.
    """
    max_bsz = max(query.shape[0], key.shape[0], value.shape[0])

    query = query.repeat(max_bsz // query.shape[0], *([1] * len(query.shape[1:])))
    key = key.repeat(max_bsz // key.shape[0], *([1] * len(key.shape[1:])))
    value = value.repeat(max_bsz // value.shape[0], *([1] * len(value.shape[1:])))

    return query, key, value
