### `Model description`

In supervised learning, a classification model is generally trained to predict an instance class by maximizing the estimated probability of gold labels. A standard training objective thus involves minimizing the `cross-entropy` between the model’s predicted distribution and the one-hot empirical distribution of training labels. A model performing well on the training set will predict an output distribution with high probability on the correct class and with near-zero probabilities on other classes. But some of these "near-zero" probabilities are larger than others and reflect, in part, the generalization capabilities of the model and how well it will perform on the test set.

As a large language model, BERT is a concatenation of the English Wikipedia and the Toronto Book Corpus comprised of 12 encoders with 12 bidirectional self-attention heads totaling 110 million parameters. One can expect to replicate `BERT base` on an 8 GPU machine within about `10 to 17 days`. On a standard, affordable GPU machine with 4 GPUs one can expect to train BERT base for about 34 days using 16-bit or about 11 days using 8-bit. Order of magnitudes less, `DistilBERT` was trained on eight 16GB, V100 GPUs for approximately `90 hours`. 

DistilBERT architecture has the same general architecture as its teacher based model, BERT. The token-type embeddings and the pooler are removed while the number of layers is reduced by a factor of 2. Most of the operations used in the Transformer architecture (linear layer and layer normalization) are highly optimized in modern linear algebra frameworks, and variations on the last dimension of the tensor (hidden size dimension) have a smaller impact on computation efficiency (for fixed parameter budget).

Student initialization, in addition to the previously described optimization and architectural choices, is an important element in the training procedure for sub-network convergance. Taking advantage of the common dimensionality between teacher and student networks, we initialize the student from the teacher by removing one of two layers, greatly reducing the data volume. DistilBERT was already built on very large batches (sampling 4,000 examples per batch) leveraging gradient accumulation using dynamic masking and without the next sentence prediction objective, a far more suitable way to train for smaller teams.


### `Training procedure`
The primary goal was to improve the model's ability to understand and categorize text data, which is essential for many real-world applications. Here is the workflow of the project:

-	`Licence`: Our project is shared under the Apache 2.0 license, which allows other developers to freely use and modify our work, provided they give appropriate credit.
-	`Tags`: The project has been tagged as "generated_from_trainer," indicating that the model was fine-tuned using a training algorithm.
-	`Dataset`: CLINC_OOS dataset. This dataset contains text samples that need to be categorized, making it perfect for our text classification task.
-	`Metrics`: The model's performance was evaluated using the accuracy metric. This is a common measure used in machine learning that quantifies the proportion of correct predictions made by the model.
-	`Model Index`: We named our fine-tuned model "kd-distilBERT-clinc," which represents the base model we used (DistilBERT) and the dataset we trained it on (CLINC).
-	`Task`: Our specific task was `Text Classification`, which involved categorizing given text data into predefined groups.
-	`Dataset Information`: The model was fine-tuned on the `train` split of the CLINC OOS dataset using the "plus" configuration. This split of the dataset was used to adjust the parameters of the model.
-	`Performance Metrics`: The model achieved an accuracy score of `91.645%`, indicating that it classified this proportion of the test data correctly.


### `Hyperparameters`
The following hyperparameters were used during training:
```
-   learning_rate: 2e-05
-   train_batch_size: 48
-   eval_batch_size: 48
-   seed: 42
-   optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
-   lr_scheduler_type: linear
-   num_epochs: 5
```

### `Training results`

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
| :-----------: | :---: | :--: | :-------------: | :------: |
|     4.2571    |  1.0  |  318 |      3.2374     |  0.7126  |
|     2.591     |  2.0  |  636 |      1.8546     |  0.8477  |
|     1.539     |  3.0  |  954 |      1.1561     |  0.8939  |
|     1.0142    |  4.0  | 1272 |      0.8673     |   0.91   |
|     0.806     |  5.0  | 1590 |      0.7849     |  0.9165  |
  
### `Framework versions`
```
-   Transformers 4.25.1
-   Pytorch 1.13.0+cu116
-   Datasets 2.7.1
-   Tokenizers 0.13.2
```

### `Model card`
[kd-distilBERT-clinc](https://huggingface.co/distilbert-base-uncased)
```
-	23,700 queries (22,500 in-scope queries covering 150 intents), grouped into 10 general categorical domains. 
-	1,200 out-of-scope queries

@misc {knowledge_distillation,
	author       = { {William M Duggan} {David Aponte} {Pooja Suthar}},
	title        = { kd-distilBERT-clinc (Revision 6bc3b2b) },
	year         = 2022,
	url          = { <https://huggingface.co/wmduggan41/kd-distilBERT-clinc> },
	doi          = { 10.57967/hf/0171 },
	publisher    = { Hugging Face }
}
```
### `Video Presentation`
[![Knowledge Distillation](../images/kdn.jpg)](https://youtu.be/w-WfHlZ8TN4)
