# Merging Adapters

The adapters library allows new adapters to be created by combining the parameters of multiple trained adapters, i.e. merging multiple existing adapters into a new one. This allows efficient domain, language and task transfer. Adapter Merging is a form of Task Arithmetics ([Ilharco et al., 2023](https://arxiv.org/abs/2212.04089); [Zhang et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html)) and hence allows increasing or unlearning specific skills. Unlearning is done by using negative weights.

The `average_adapter()` method provides this merging functionality:

```python
model.add_adapter("bottleneck_1", "seq_bn")
model.add_adapter("bottleneck_2", "seq_bn")
model.add_adapter("bottleneck_3", "seq_bn")

model.average_adapter(adapter_name="avg", adapter_list=["bottleneck_1", "bottleneck_2", "bottleneck_3"], weights=[-1, 1.2, 0.8])
```
In this example, the parameters of the three added bottleneck adapters are merged (with weights `-1`, `1.2` and `0.8`, respectively) to create a new adapter `avg`.
Note that for this to succeed, all averaged adapters must use the same adapter configuration. Compared to the [output averaging](adapter_composition.md#output-averaging) composition block, merging parameters of adapters has the advantage of not inducing any additional inference time relative to using a single adapter.

All [adapter methods](model_overview.md#table-of-adapter-methods) support linear merging. In linear merging, the weights of the trained adapters are linearly combined: Let us have *N* adapters and let $\Phi_i$ be all the parameters of the *i*-th adapter, and $\lambda_i$ be the corresponding weight that determines how strongly we weigh this adapter. The merged adapter parameters $\Phi_{merged}$ are calculated as:

$$
\Phi_{merged} = \sum_{i=0}^{N} \lambda_i \Phi_i
$$

The `average_adapter` method only merges the weights of the adapters but does not create a new head. To average the weights of heads, use the `average_head` method. Since heads are usually linear layers, the `average_head` method uses linear merging:

```python
model.add_masked_lm_head("head_1")
model.add_masked_lm_head("head_2")

model.average_head(head_name="avg_head", head_list=["head_1", "head_2"], weights=[0.2, 0.8])
```

#### Merging LoRA Adapters
LoRA introduces $A$ and $B$ matrices with $\Delta W = BA$. Since the B and A matrices are strongly dependent on each other, there are several ways to merge the weights of LoRA adapters. You can choose the combination method by passing the `combine_strategy` parameter to the `average_adapter` method:

1. `combine_strategy = "linear"`: Linear Combination (default). This has been proposed for LoRA by [Chronopoulou et al. (2023)](https://arxiv.org/abs/2311.09344). With $\Phi = \{A, B\}$:
    
    $$
    \Phi_{merged} = \sum_{i=0}^{N} \lambda_i \Phi_i
    $$

2. `combine_strategy = "lora_linear_only_negate_b"` Following [Zhang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html), this method only uses negative weights for the B-matrix if the weight is negative:

    $$
    A_{merged} &= \sum_{i=0}^{N} |\lambda_i| A_i\\
    B_{merged} &= \sum_{i=0}^{N} \lambda_i B_i
    $$

3. `combine_strategy = "lora_delta_w_svd"`: This method merges the $\Delta W_i$ of each adapter and then performs a singular value decomposition (SVD) to obtain the *A* and *B* LoRA matrices:
    1. For every adapter *i* we calculate: $\Delta W_i = B_i \cdot A_i$
    2. $\Delta W_{new} = \sum_{i=0}^N \lambda_i \cdot W_i$ 
    3. Perform SVD on $\text{SVD}(\Delta W_{new})$ to obtain $A_{new}$ and $B_{new}$

`lora_delta_w_svd` is not supported by Deberta and GPT-2. Example usage of these LoRA-specific merging strategies:

```python
model.add_adapter("lora_1", "seq_bn")
model.add_adapter("lora_2", "seq_bn")
model.add_adapter("lora_3", "seq_bn")

model.average_adapter(
    adapter_name="lora_avg",
    adapter_list=["lora_1", "lora_2", "lora_3"],
    weights=[1, -1, 1],
    combine_strategy="lora_delta_w_svd",
    svd_rank=8
)
# Note that "lora_delta_w_svd" requires the "svd_rank" parameter, which determines the r (rank) of the resulting LoRA adapter after SVD
```

For both output and parameter averaging, passed weights are normalized by default. To disable normalization, pass `normalize_weights=False`.
For more detailed examples and explanations, refer to our [Task Arithmetic notebook](https://github.com/adapter-hub/adapters/tree/main/notebooks/task_arithmetics_in_adapter.ipynb).


```{eval-rst}
.. tip::
    Adding more adapter merging methods is easy: You have to simply modify the ``average_adapter`` method. Most adapter-methods use the default implementation that only supports linear merging in `model_mixin.py <https://github.com/adapter-hub/adapters/blob/main/src/adapters/model_mixin.py>`_. Others like LoRA overwrite this method to add new merging methods like "lora_delta_w_svd", have a look at `lora.py <https://github.com/adapter-hub/adapters/blob/main/src/adapters/methods/lora.py>`_.
```