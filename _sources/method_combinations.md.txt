# Method Combinations

_Configuration class_: [`ConfigUnion`](adapters.ConfigUnion)

While different efficient fine-tuning methods and configurations have often been proposed as standalone, combining them for joint training might be beneficial. 
To make this process easier, `adapters` provides the possibility to group multiple configuration instances using the [`ConfigUnion`](adapters.ConfigUnion) class.

For example, this could be used to define different reduction factors for the adapter modules placed after the multi-head attention and the feed-forward blocks:

```python
from adapters import BnConfig, ConfigUnion

config = ConfigUnion(
    BnConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
    BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
)
model.add_adapter("union_adapter", config=config)
```

## Mix-and-Match Adapters

_Configuration class_: [`MAMConfig`](adapters.MAMConfig)

[He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) study various variants and combinations of efficient fine-tuning methods.
They propose _Mix-and-Match Adapters_ as a combination of Prefix Tuning and parallel bottleneck adapters.
This configuration is supported by `adapters` out-of-the-box:

```python
from adapters import MAMConfig

config = MAMConfig()
model.add_adapter("mam_adapter", config=config)
```

and is identical to using the following `ConfigUnion`:

```python
from adapters import ConfigUnion, ParBnConfig, PrefixTuningConfig

config = ConfigUnion(
    PrefixTuningConfig(bottleneck_size=800),
    ParBnConfig(),
)
model.add_adapter("mam_adapter", config=config)
```

_Papers:_
- [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/pdf/2110.04366.pdf) (He et al., 2021)

## UniPELT

_Configuration class_: [`UniPELTConfig`](adapters.UniPELTConfig)

```{eval-rst}
.. figure:: img/unipelt.png
    :height: 300
    :align: center
    :alt: Illustration of UniPELT.

    Illustration of the UniPELT method within one Transformer layer. Trained components are colored in shades of magenta.
```

An approach similar to the work of [He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) is taken by [Mao et al. (2022)](https://arxiv.org/pdf/2110.07577.pdf) in their _UniPELT_ framework.
They, too, combine multiple efficient fine-tuning methods, namely LoRA, Prefix Tuning and bottleneck adapters, in a single unified setup.
_UniPELT_ additionally introduces a gating mechanism that controls the activation of the different submodules.

Concretely, for each adapted module $m$, UniPELT adds a trainable gating value $\mathcal{G}_m \in (0, 1)$ that is computed via a feed-forward network ($W_{\mathcal{G}_m}$) and sigmoid activation ($\sigma$) from the Transformer layer input states ($x$):

$$\mathcal{G}_m \leftarrow \sigma(W_{\mathcal{G}_m} \cdot x)$$

These gating values are then used to scale the output activations of the injected adapter modules, e.g., for a LoRA layer:

$$
h \leftarrow W_0 x + \mathcal{G}_{LoRA} B A x
$$

In the configuration classes of `adapters`, these gating mechanisms can be activated via `use_gating=True`.
The full UniPELT setup can be instantiated using `UniPELTConfig`[^unipelt]:

[^unipelt]: Note that the implementation of UniPELT in `adapters` follows the implementation in the original code, which is slightlty different from the description in the paper. See [here](https://github.com/morningmoni/UniPELT/issues/1) for more.

```python
from adapters import UniPELTConfig

config = UniPELTConfig()
model.add_adapter("unipelt", config=config)
```

which is identical to the following `ConfigUnion`:

```python
from adapters import ConfigUnion, LoRAConfig, PrefixTuningConfig, SeqBnConfig

config = ConfigUnion(
    LoRAConfig(r=8, alpha=2, use_gating=True),
    PrefixTuningConfig(prefix_length=10, use_gating=True),
    SeqBnConfig(reduction_factor=16, use_gating=True),
)
model.add_adapter("unipelt", config=config)
```

Finally, as the gating values for each adapter module might provide interesting insights for analysis, `adapters` comes with an integrated mechanism of returning all gating values computed during a model forward pass via the `output_adapter_gating_scores` parameter:

```python
outputs = model(**inputs, output_adapter_gating_scores=True)
gating_scores = outputs.adapter_gating_scores
```
Note that this parameter is only available to base model classes and [AdapterModel classes](prediction_heads.md#adaptermodel-classes).
In the example, `gating_scores` holds a dictionary of the following form:
```
{
    '<adapter_name>': {
        <layer_id>: {
            '<module_location>': np.array([...]),
            ...
        },
        ...
    },
    ...
}
```

_Papers:_
- [UNIPELT: A Unified Framework for Parameter-Efficient Language Model Tuning](https://arxiv.org/pdf/2110.07577.pdf) (Mao et al., 2022)
