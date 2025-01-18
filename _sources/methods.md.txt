# Adapter Methods

On this page, we present all adapter methods currently integrated into the `adapters` library.
A tabular overview of adapter methods is provided [here](overview.md#table-of-adapter-methods). 
Additionally, options to combine multiple adapter methods in a single setup are presented [on the next page](method_combinations.md).

## Bottleneck Adapters

_Configuration class_: [`BnConfig`](adapters.BnConfig)

Bottleneck adapters introduce bottleneck feed-forward layers in each layer of a Transformer model.
Generally, these adapter layers consist of a down-projection matrix $W_{down}$ that projects the layer hidden states into a lower dimension $d_{bottleneck}$, a non-linearity $f$, an up-projection $W_{up}$ that projects back into the original hidden layer dimension and a residual connection $r$:

$$
h \leftarrow W_{up} \cdot f(W_{down} \cdot h) + r
$$

Depending on the concrete adapter configuration, these layers can be introduced at different locations within a Transformer block. Further, residual connections, layer norms, activation functions and bottleneck sizes ,etc., can be configured.

The most important configuration hyperparameter to be highlighted here is the bottleneck dimension $d_{bottleneck}$.
In adapters, this bottleneck dimension is specified indirectly via the `reduction_factor` attribute of a configuration.
This `reduction_factor` defines the ratio between a model's layer hidden dimension and the bottleneck dimension, i.e.:

$$
\text{reduction_factor} = \frac{d_{hidden}}{d_{bottleneck}}
$$

A visualization of further configuration options related to the adapter structure is given in the figure below. For more details, we refer to the documentation of `BnConfig`](adapters.BnConfig).


```{eval-rst}
.. figure:: img/architecture.png
    :width: 350
    :align: center
    :alt: Adapter architectures

    Visualization of possible adapter configurations with corresponding dictionary keys.
```

`adapters` comes with pre-defined configurations for some bottleneck adapter architectures proposed in literature:

- [`DoubleSeqBnConfig`](adapters.DoubleSeqBnConfig), as proposed by [Houlsby et al. (2019)](https://arxiv.org/pdf/1902.00751.pdf) places adapter layers after both the multi-head attention and feed-forward block in each Transformer layer.
- [`SeqBnConfig`](adapters.SeqBnConfig), as proposed by [Pfeiffer et al. (2020)](https://arxiv.org/pdf/2005.00052.pdf) places an adapter layer only after the feed-forward block in each Transformer layer.
- [`ParBnConfig`](adapters.ParBnConfig), as proposed by [He et al. (2021)](https://arxiv.org/pdf/2110.04366.pdf) places adapter layers in parallel to the original Transformer layers.
- [`AdapterPlusConfig`](adapters.AdapterPlusConfig), as proposed by [Steitz and Roth (2024)](https://arxiv.org/pdf/2406.06820) places adapter layers adapter layers after the multi-head attention and has channel wise scaling and houlsby weight initialization
_Example_:
```python
from adapters import BnConfig

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
model.add_adapter("bottleneck_adapter", config=config)
```

_Papers:_

* [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf) (Houlsby et al., 2019)
* [Simple, Scalable Adaptation for Neural Machine Translation](https://arxiv.org/pdf/1909.08478.pdf) (Bapna and Firat, 2019)
* [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://aclanthology.org/2021.eacl-main.39.pdf) (Pfeiffer et al., 2021)
* [Adapters Strike Back](https://arxiv.org/pdf/2406.06820) (Steitz and Roth., 2024)
* [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779.pdf) (Pfeiffer et al., 2020)

```{eval-rst}
.. note::
    The two parameters ``original_ln_before`` and ``original_ln_after`` inside bottleneck adapters control both the addition of the residual input and the application of the pretrained layer norm. If the original model does not apply a layer norm function at a specific position of the forward function (e.g after the FFN layer), the two bottleneck parameters of the adapter set at that same position will only control the application of the residual input.
```

## Language Adapters - Invertible Adapters

_Configuration class_: [`SeqBnInvConfig`](adapters.SeqBnInvConfig), [`DoubleSeqBnInvConfig`](adapters.DoubleSeqBnInvConfig)

The MAD-X setup ([Pfeiffer et al., 2020](https://arxiv.org/pdf/2005.00052.pdf)) proposes language adapters to learn language-specific transformations.
After being trained on a language modeling task, a language adapter can be stacked before a task adapter for training on a downstream task.
To perform zero-shot cross-lingual transfer, one language adapter can simply be replaced by another.

In terms of architecture, language adapters are largely similar to regular bottleneck adapters, except for an additional _invertible adapter_ layer after the LM embedding layer.
Embedding outputs are passed through this invertible adapter in the forward direction before entering the first Transformer layer and in the inverse direction after leaving the last Transformer layer.
Invertible adapter architectures are further detailed in [Pfeiffer et al. (2020)](https://arxiv.org/pdf/2005.00052.pdf) and can be configured via the `inv_adapter` attribute of the `BnConfig` class.

_Example_:
```python
from adapters import SeqBnInvConfig

config = SeqBnInvConfig()
model.add_adapter("lang_adapter", config=config)
```

_Papers:_
- [MAD-X: An Adapter-based Framework for Multi-task Cross-lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf) (Pfeiffer et al., 2020)

```{eval-rst}
.. note::
    V1.x of adapters made a distinction between task adapters (without invertible adapters) and language adapters (with invertible adapters) with the help of the ``AdapterType`` enumeration.
    This distinction was dropped with v2.x.
```

## Prefix Tuning

_Configuration class_: [`PrefixTuningConfig`](adapters.PrefixTuningConfig)

```{eval-rst}
.. figure:: img/prefix.png
    :height: 300
    :align: center
    :alt: Illustration of Prefix Tuning.

    Illustration of the Prefix Tuning method within one Transformer layer. Trained components are colored in shades of magenta.
```

Prefix Tuning ([Li and Liang, 2021](https://aclanthology.org/2021.acl-long.353.pdf)) introduces new parameters in the multi-head attention blocks in each Transformer layer.
More specifically, it prepends trainable prefix vectors $P^K$ and $P^V$ to the keys and values of the attention head input, each of a configurable prefix length $l$ (`prefix_length` attribute):

$$
head_i = \text{Attention}(Q W_i^Q, [P_i^K, K W_i^K], [P_i^V, V W_i^V])
$$

Following the original authors, the prefix vectors in $P^K$ and $P^V$ are not optimized directly but reparameterized via a bottleneck MLP.
This behavior is controlled via the `flat` attribute of the configuration.
Using `PrefixTuningConfig(flat=True)` will create prefix tuning vectors that are optimized without reparameterization.

_Example_:
```python
from adapters import PrefixTuningConfig

config = PrefixTuningConfig(flat=False, prefix_length=30)
model.add_adapter("prefix_tuning", config=config)
```

As reparameterization using the bottleneck MLP is not necessary for performing inference on an already trained Prefix Tuning module, `adapters` includes a function to "eject" a reparameterized Prefix Tuning into a flat one:
```python
model.eject_prefix_tuning("prefix_tuning")
```
This will only retain the necessary parameters and reduces the size of the trained Prefix Tuning.

_Papers:_
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190.pdf) (Li and Liang, 2021)

## Compacter

_Configuration class_: [`CompacterConfig`](adapters.CompacterConfig), [`CompacterPlusPlusConfig`](adapters.CompacterPlusPlusConfig)

```{eval-rst}
.. figure:: img/compacter.png
    :height: 300
    :align: center
    :alt: Illustration of Compacter.

    Illustration of the Compacter method within one Transformer layer. Trained components are colored in shades of magenta.
```

The Compacter architecture proposed by [Mahabadi et al., 2021](https://arxiv.org/pdf/2106.04647.pdf)
is similar to the bottleneck adapter architecture. It only exchanges the linear down- and 
up-projection with a PHM layer. Unlike the linear layer, the PHM layer constructs its weight matrix from two smaller matrices, which reduces the number of parameters.
 These matrices can be factorized and shared between all adapter layers. You can exchange the down- and up-projection layers from any of the bottleneck adapters described in the previous section
for a PHM layer by specifying `use_phm=True` in the config.

The PHM layer has the following additional properties: `phm_dim`, `shared_phm_rule`, `factorized_phm_rule`, `learn_phm`, 
`factorized_phm_W`, `shared_W_phm`, `phm_c_init`, `phm_init_range`, `hypercomplex_nonlinearity`

For more information, check out the [`BnConfig`](adapters.BnConfig) class.

To add a Compacter to your model, you can use the predefined configs:
```python
from adapters import CompacterConfig

config = CompacterConfig()
model.add_adapter("dummy", config=config)
```
_Papers:_
- [COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers](https://arxiv.org/pdf/2106.04647.pdf) (Mahabadi, Henderson and Ruder, 2021)

## LoRA

_Configuration class_: [`LoRAConfig`](adapters.LoRAConfig)

```{eval-rst}
.. figure:: img/lora.png
    :height: 300
    :align: center
    :alt: Illustration of LoRA.

    Illustration of the LoRA method within one Transformer layer. Trained components are colored in shades of magenta.
```

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique proposed by [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf).
LoRA injects trainable low-rank decomposition matrices into the layers of a pre-trained model.
For any model layer expressed as a matrix multiplication of the form $h = W_0 x$, it performs a reparameterization, such that:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

Here, $A \in \mathbb{R}^{r\times k}$ and $B \in \mathbb{R}^{d\times r}$ are the decomposition matrices and $r$, the low-dimensional rank of the decomposition, is the most important hyperparameter.

While, in principle, this reparameterization can be applied to any weight matrix in a model, the original paper only adapts the attention weights of the Transformer self-attention sub-layer with LoRA.
`adapters` additionally allows injecting LoRA into the dense feed-forward layers in the intermediate and output components of a Transformer block.
You can configure the locations where LoRA weights should be injected using the attributes in the [`LoRAConfig`](adapters.LoRAConfig) class.

_Example_:
```python
from adapters import LoRAConfig

config = LoRAConfig(r=8, alpha=16)
model.add_adapter("lora_adapter", config=config)
```

In the design of LoRA, Hu et al. (2021) also pay special attention to keeping the inference latency overhead compared to full fine-tuning at a minimum.
To accomplish this, the LoRA reparameterization can be merged with the original pre-trained weights of a model for inference.
Thus, the adapted weights are directly used in every forward pass without passing activations through an additional module.
In `adapters`, this can be realized using the built-in [`merge_adapter()`](adapters.ModelAdaptersMixin.merge_adapter)  method:
```python
model.merge_adapter("lora_adapter")
```

To continue training on this LoRA adapter or to deactivate it entirely, the merged weights first have to be reset again:
```python
model.reset_adapter()
```

_Papers:_
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) (Hu et al., 2021)

## (IA)^3

_Configuration class_: [`IA3Config`](adapters.IA3Config)

```{eval-rst}
.. figure:: img/ia3.png
    :height: 300
    :align: center
    :alt: Illustration of (IA)^3.

    Illustration of the (IA)^3 method within one Transformer layer. Trained components are colored in shades of magenta.
```

_Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3)_ is an efficient fine-tuning method proposed within the _T-Few_ fine-tuning approach by [Liu et al. (2022)](https://arxiv.org/pdf/2205.05638.pdf).
(IA)^3 introduces trainable vectors $l_W$ into different components of a Transformer model, which perform element-wise rescaling of inner model activations.
For any model layer expressed as a matrix multiplication of the form $h = W x$, it therefore performs an element-wise multiplication with $l_W$, such that:

$$
h = l_W \odot W x
$$

Here, $\odot$ denotes element-wise multiplication where the entries of $l_W$ are broadcasted to the shape of $W$.

_Example_:
```python
from adapters import IA3Config

config = IA3Config()
model.add_adapter("ia3_adapter", config=config)
```

The implementation of (IA)^3, as well as the [`IA3Config`](adapters.IA3Config) class, are derived from the implementation of [LoRA](#lora), with a few main modifications.
First, (IA)^3 uses multiplicative composition of weights instead of additive composition, as in LoRA.
Second, the added weights are not further decomposed into low-rank matrices.
These modifications are controlled via the `composition_mode` configuration attribute by setting `composition_mode="scale"`.
Additionally, as the added weights are already of rank 1, `r=1` is set.

Beyond that, both methods share the same configuration attributes that allow you to specify in which Transformer components rescaling vectors will be injected.
Following the original implementation, [`IA3Config`](adapters.IA3Config) adds rescaling vectors to the self-attention weights (`selfattn_lora=True`) and the final feed-forward layer (`output_lora=True`).
Further, you can modify which matrices of the attention mechanism to rescale by leveraging the `attn_matrices` attribute.
By default, (IA)^3 injects weights into the key ('k') and value ('v') matrices but not in the query ('q') matrix.

Finally, similar to LoRA, (IA)^3 also allows merging the injected parameters with the original weight matrices of the Transformer model.
E.g.:
```python
# Merge (IA)^3 adapter
model.merge_adapter("ia3_adapter")

# Reset merged weights
model.reset_adapter()
```

_Papers:_
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/pdf/2205.05638.pdf) (Liu et al., 2022)

## Prompt Tuning
Prompt Tuning is an efficient fine-tuning technique proposed by Lester et al. (2021). Prompt tuning adds tunable tokens, called soft-prompts, that are prepended to the input text.
First, the input sequence ${x_1, x_2, \dots, x_n }$ gets embedded, resulting in the matrix $X_e \in \mathbb{R}^{n \times e}$ where $e$ is the dimension of
the embedding space. The soft-prompts with length $p$ are represented as $P_e \in \mathbb{R}^{p \times e}$.
$P_e$ and $X_e$ get concatenated, forming the input of the following encoder or decoder:

$$
\left[P_e; X_e\right] \in \mathbb{R}^{\left(p + n\right) \times e}
$$

The `PromptTuningConfig` has the properties:
- `prompt_length`: to set the soft-prompts length $p$ 
- `prompt_init`: to set the weight initialisation method, which is either "random_uniform" or "from_string" to initialize each prompt token with an embedding drawn from the model’s vocabulary.
    - `prompt_init_text` as the text use for initialisation if `prompt_init="from_string"`
- `combine`: To define if the prefix should be added before the embedded input sequence or after the BOS token

To add Prompt Tuning to your model, you can use the predefined configs:
```python
from adapters import PromptTuningConfig

config = PromptTuningConfig(prompt_length=10)
model.add_adapter("dummy", config=config)
```

_Papers:_
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/) (Lester et al., 2021)

## ReFT

_Configuration class_: [`ReftConfig`](adapters.ReftConfig)

Representation Fine-Tuning (ReFT), as first proposed by [Wu et al. (2024)](https://arxiv.org/pdf/2404.03592), leverages so-called interventions to adapt the pre-trained representations of a language model.
Within the context of ReFT, these interventions can intuitively be thought of as adapter modules placed after each Transformer layer.
In the general form, an intervention function $\Phi$ can thus be defined as follows:

$$
\Phi(h) = h + R^T (W h + b - R h)
$$

Here, $R \in \mathbb{R}^{r \times d}$ and $W \in \mathbb{R}^{r \times d}$ are low-rank matrices of rank $r$.
$h$ is the layer output hidden state at a single sequence position, i.e. interventions can be applied independently at each position.

Based on this general form, the ReFT paper proposes multiple instantiations of ReFT methods supported by _Adapters_:

- **LoReFT** enforces orthogonality of rows in $R$. Defined via [`LoReftConfig`](adapters.LoReftConfig) or via the `orthogonality` attribute as in the following example:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=True
)  # equivalent to LoreftConfig()
```

- **NoReFT** does not enforce orthogonality in $R$. Defined via [`NoReftConfig`](adapters.NoReftConfig) or equivalently:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=False
)  # equivalent to NoreftConfig()
```

- **DiReFT** does not enforce orthogonality in $R$ and additionally removes subtraction of $R h$ in the intervention, Defined via [`DiReftConfig`](adapters.DiReftConfig) or equivalently:
```python
config = ReftConfig(
    layers="all", prefix_positions=3, suffix_positions=0, r=1, orthogonality=False, subtract_projection=False
)  # equivalent to DireftConfig()
```

In addition, _Adapters_ supports configuring multiple hyperparameters tuned in the ReFT paper in `ReftConfig`, including:
- `prefix_positions`: number of prefix positions
- `suffix_positions`: number of suffix positions
- `layers`: The layers to intervene on. This can either be `"all"` or a list of layer ids
- `tied_weights`: whether to tie parameters between prefixes and suffixes

_Papers:_

* [ReFT: Representation Finetuning for Language Models](https://arxiv.org/pdf/2404.03592) (Wu et al., 2024)
