BEiT
======

The Bidirectional Encoder representation from Image Transformers (BEiT) model was proposed in `BERT Pre-Training of Image 
Transformers <https://arxiv.org/abs/2106.08254>`__ by Hangbo Bao, Li Dong, Songhao Piao, Furu Wei.


The abstract from the paper is the following:

*We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation 
from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image 
modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image 
patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into 
visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training 
objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we 
directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. 
Experimental results on image classification and semantic segmentation show that our model achieves competitive results 
with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, 
significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 
86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%).*

BeitAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.BeitAdapterModel
    :members:
    :inherited-members: BeitPreTrainedModel
