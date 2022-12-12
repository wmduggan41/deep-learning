---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- clinc_oos
metrics:
- accuracy
model-index:
- name: kd-distilBERT-clinc
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: clinc_oos
      type: clinc_oos
      config: plus
      split: train
      args: plus
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9164516129032259
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# kd-distilBERT-clinc

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the clinc_oos dataset.
It achieves the following results on the evaluation set:

-   Loss: 0.7849
-   Accuracy: 0.9165

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

-   learning_rate: 2e-05
-   train_batch_size: 48
-   eval_batch_size: 48
-   seed: 42
-   optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
-   lr_scheduler_type: linear
-   num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
| :-----------: | :---: | :--: | :-------------: | :------: |
|     4.2571    |  1.0  |  318 |      3.2374     |  0.7126  |
|     2.591     |  2.0  |  636 |      1.8546     |  0.8477  |
|     1.539     |  3.0  |  954 |      1.1561     |  0.8939  |
|     1.0142    |  4.0  | 1272 |      0.8673     |   0.91   |
|     0.806     |  5.0  | 1590 |      0.7849     |  0.9165  |

### Framework versions

-   Transformers 4.25.1
-   Pytorch 1.13.0+cu116
-   Datasets 2.7.1
-   Tokenizers 0.13.2


@misc {william_m_duggan_2022,
	author       = { {William M Duggan} {David Aponte} {Pooja Suthar}},
	title        = { kd-distilBERT-clinc (Revision 6bc3b2b) },
	year         = 2022,
	url          = { <https://huggingface.co/wmduggan41/kd-distilBERT-clinc> },
	doi          = { 10.57967/hf/0171 },
	publisher    = { Hugging Face }
}
