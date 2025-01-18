# Overview and Configuration

Large pre-trained Transformer-based language models (LMs) have become the foundation of NLP in recent years.
While the most prevalent method of using these LMs for transfer learning involves costly *full fine-tuning* of all model parameters, a series of *efficient* and *lightweight* alternatives have recently been established.
Instead of updating all parameters of the pre-trained LM towards a downstream target task, these methods commonly introduce a small number of new parameters and only update these while keeping the pre-trained model weights fixed.

```{admonition} Why use Efficient Fine-Tuning?
Efficient fine-tuning methods offer multiple benefits over the full fine-tuning of LMs:

- They are **parameter-efficient**, i.e., they only update a tiny subset (often under 1%) of a model's parameters.
- They often are **modular**, i.e., the updated parameters can be extracted and shared independently of the base model parameters.
- They are easy to share and deploy due to their **small file sizes**, e.g., having only ~3MB per task instead of ~440MB for sharing a full model.
- They **speed up training**, i.e., efficient fine-tuning often requires less training time than fully fine-tuning LMs.
- They are **composable**, e.g., multiple adapters trained on different tasks can be stacked, fused, or mixed to leverage their combined knowledge.
- They often provide **on-par performance** with full fine-tuning.
```

More specifically, let the parameters of a LM be composed of a set of pre-trained parameters $\Theta$ (frozen) and a set of (newly introduced) parameters $\Phi$.
Then, efficient fine-tuning methods optimize only $\Phi$ according to a loss function $L$ on a dataset $D$:

$$
\Phi^* \leftarrow \arg \min_{\Phi} L(D; \{\Theta, \Phi\})
$$

Efficient fine-tuning might insert parameters $\Phi$ at different locations of a Transformer-based LM.
One early and successful method, (bottleneck) adapters, introduces bottleneck feed-forward layers in each layer of a Transformer model.
While these adapters have laid the foundation of the `adapters` library, multiple alternative methods have been introduced and integrated since.

```{eval-rst}
.. important::
    In literature, different terms are used to refer to efficient fine-tuning methods.
    The term "adapter" is usually only applied to bottleneck adapter modules.
    However, most efficient fine-tuning methods follow the same general idea of inserting a small set of new parameters and, by this, "adapting" the pre-trained LM to a new task.
    In ``adapters``, the term "adapter" thus may refer to any efficient fine-tuning method if not specified otherwise.
```

In the remaining sections, we will present how adapter methods can be configured in `adapters`.
The next two pages will then present the methodological details of all currently supported adapter methods.

## Table of Adapter Methods

The following table gives an overview of all adapter methods supported by `adapters`.
Identifiers and configuration classes are explained in more detail in the [next section](#configuration).

| Identifier | Configuration class | More information
| --- | --- | --- |
| `seq_bn` | `SeqBnConfig()` | [Bottleneck Adapters](methods.html#bottleneck-adapters) |
| `double_seq_bn` | `DoubleSeqBnConfig()` | [Bottleneck Adapters](methods.html#bottleneck-adapters) |
| `par_bn` | `ParBnConfig()` | [Bottleneck Adapters](methods.html#bottleneck-adapters) |
| `scaled_par_bn` | `ParBnConfig(scaling="learned")` | [Bottleneck Adapters](methods.html#bottleneck-adapters) |
| `seq_bn_inv` | `SeqBnInvConfig()` | [Invertible Adapters](methods.html#language-adapters---invertible-adapters) |
| `double_seq_bn_inv` | `DoubleSeqBnInvConfig()` | [Invertible Adapters](methods.html#language-adapters---invertible-adapters) |
| `compacter` | `CompacterConfig()` | [Compacter](methods.html#compacter) |
| `compacter++` | `CompacterPlusPlusConfig()` | [Compacter](methods.html#compacter) |
| `prefix_tuning` | `PrefixTuningConfig()` | [Prefix Tuning](methods.html#prefix-tuning) |
| `prefix_tuning_flat` | `PrefixTuningConfig(flat=True)` | [Prefix Tuning](methods.html#prefix-tuning) |
| `lora` | `LoRAConfig()` | [LoRA](methods.html#lora) |
| `ia3` | `IA3Config()` | [IA³](methods.html#ia-3) |
| `mam` | `MAMConfig()` | [Mix-and-Match Adapters](method_combinations.html#mix-and-match-adapters) |
| `unipelt` | `UniPELTConfig()` | [UniPELT](method_combinations.html#unipelt) |
| `prompt_tuning` | `PromptTuningConfig()` | [Prompt Tuning](methods.html#prompt-tuning) |
| `loreft` | `LoReftConfig()` | [ReFT](methods.html#reft) |
| `noreft` | `NoReftConfig()` | [ReFT](methods.html#reft) |
| `direft` | `DiReftConfig()` | [ReFT](methods.html#reft) |

## Configuration

All supported adapter methods can be added, trained, saved and shared using the same set of model class functions (see [class documentation](adapters.ModelAdaptersMixin)).
Each method is specified and configured using a specific configuration class, all of which derive from the common [`AdapterConfig`](adapters.AdapterConfig) class.
E.g., adding one of the supported adapter methods to an existing model instance follows this scheme:
```python
model.add_adapter("name", config=<ADAPTER_CONFIG>)
```

Here, `<ADAPTER_CONFIG>` can either be:
- a configuration string, as described below
- an instance of a configuration class, as listed in the table above
- a path to a JSON file containing a configuration dictionary

### Configuration strings

Configuration strings are a concise way of defining a specific adapter method configuration.
They are especially useful when adapter configurations are passed from external sources such as the command-line, when using configuration classes is not an option.

In general, a configuration string for a single method takes the form `<identifier>[<key>=<value>, ...]`.
Here, `<identifier>` refers to one of the identifiers listed in [the table above](#table-of-adapter-methods), e.g. `par_bn`.
In square brackets after the identifier, you can set specific configuration attributes from the respective configuration class, e.g. `par_bn[reduction_factor=2]`.
If all attributes remain at their default values, this can be omitted.

Finally, it is also possible to specify a [method combination](method_combinations.md) as a configuration string by joining multiple configuration strings with `|`, e.g.:
```python
config = "prefix_tuning[bottleneck_size=800]|parallel"
```

is identical to the following `ConfigUnion`:

```python
config = ConfigUnion(
    PrefixTuningConfig(bottleneck_size=800),
    ParBnConfig(),
)
```
