# Sarcasm-Detection-in-Text
## Overview

This project explores sarcasm detection in text by fine-tuning pre-trained Transformer models and by training Bi-RNNs with Flat Attention Networks (FAN) and Hierarchical Attention Networks (HAN). It involves training DeBERTa, RoBERTa, FAN and HAN models on three distinct datasets: iSarcasmEval, Sarcasm_Corpus_V2, and a combined dataset(iSarcasmEval + Sarcasm_Corpus_V2). The trained models are then evaluated on their respective test sets and further assessed on the BigBench SNARKS benchmark to test their ability to discern sarcasm in a pairwise comparison task.

The primary goals include:
* Fine-tuning robust sarcasm detection models.
* Implementing strategies to handle class imbalance (e.g., weighted loss).
* Evaluating model performance thoroughly using standard metrics and specialized benchmarks.
* Comparing the performance of models trained on different sarcasm datasets.

## Models Used

* **DeBERTa**: `microsoft/deberta-v3-base`
* **RoBERTa**: `roberta-base`
* **Bi-RNN with Flat Attention Networks (FAN)**
* **Bi-RNN with Hierarchical Attention Networks (HAN)**

## Datasets

1.  **Fine-tuning Datasets:**
    * **iSarcasmEval**: Contains training, validation, and test sets for sarcasm detection. This dataset exhibited some class imbalance, addressed using a weighted loss function for the RoBERTa and DeBERTa models.
    * **Sarcasm\_Corpus\_V2**: Includes training, validation, and test sets. This dataset was noted to be more balanced.
    * **Combined Dataset**: A merger of the iSarcasmEval and Sarcasm\_Corpus\_V2 training/validation/test sets.
    Each dataset is expected in CSV format with 'text' and 'sarcastic' (0 or 1) columns.

2.  **Evaluation Benchmark:**
    * **BigBench SNARKS**: Used for evaluating the fine-tuned models on an unseen, out-of-distribution task. The SNARKS task involves determining which of two provided statements, (a) or (b), is more sarcastic. The `task.json` file from this benchmark is used for evaluation.

## Dataset preparation, Model Training and Evaluation File Structure

* `dataset_preparation.ipynb`: Used for preparing train-test-validation splits
* `dataset_stats`: Used to check the class distribution across training, testing and validation sets of all three datasets
* `DeBERTA-model.ipynb`: Jupyter Notebook for fine-tuning and evaluating the DeBERTa-v3-base model on the iSarcasmEval, Sarcasm_Corpus_V2, and combined datasets.
* `RoBERTa_model.ipynb`: Jupyter Notebook for fine-tuning and evaluating a RoBERTa-base model.
* `Bi-RNN-FAN.ipynb`: Used to train FAN models (with and without GloVe) on three datasets (total 6 model configurations)
* `Bi-RNN-HAN.ipynb`: Used to train HAN models (with and without GloVe) on three datasets (total 6 model configurations)
* `Bigbench-Snarks-evaluation.ipynb`: Script/Notebook to evaluate the sarcasm detection models (trained by the above notebooks) on the BigBench SNARKS dataset.
* Following folder contains the data used for training, testing and validation:
    * `/iSarcasmEval/`
        * `train.csv`
        * `val.csv`
        * `test.csv`
    * `/Sarcasm_Corpus_V2/`
        * `train.csv`
        * `val.csv`
        * `test.csv`
    * `/combined_data/`
        * `train.csv`
        * `val.csv`
        * `test.csv`

## Model Files and Evaluation Artifacts

The following directories contain the **trained model files**, **training-validation loss curves**, and **confusion matrices** depicting the model's performance on the respective test sets.
*(Note: Only FAN and HAN models are included.)*

### Included Folders:

* `/isarc_fan_16_glove/`
* `/isarc_fan_16/`
* `/scv2_fan_16_glove/`
* `/scv2_fan_16/`
* `/comb_fan_16_glove/`
* `/comb_fan_16/`
* `/isarc_han_8_4_glove/`
* `/isarc_han_8_4/`
* `/scv2_han_8_4_glove/`
* `/scv2_han_8_4/`
* `/comb_han_8_4_glove/`
* `/comb_han_8_4/`

### Folder Naming Convention:

Each folder name follows the format:
`{dataset_abbreviation}_{model}_{model_configuration}_{glove_flag}`

#### Components:

* **`dataset_abbreviation`**:

  * `isarc`: iSarcasmEval dataset
  * `scv2`: Sarcasm Corpus V2
  * `comb`: Combined dataset (iSarcasmEval + Sarcasm Corpus V2)

* **`model`**:

  * `fan`: FAN model
  * `han`: HAN model

* **`model_configuration`**:

  * `16`: 16 hidden units (used in FAN models)
  * `8_4`: 8 hidden units for word-level attention/encoding and 4 hidden units for sentence-level attention/encoding (used in HAN models)

* **`glove_flag`**:

  * `glove`: GloVe embeddings were used
  * *(empty)*: GloVe embeddings were **not** used


### Evaluation

* The evaluation data used in this project was sourced from the [BigBench SNARKS task](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/snarks).

  * Original file: `task.json`

* This JSON file was converted to CSV format and saved as:
  `BIG-Bench-SNARKS/test.csv`

* The folder `BIG-Bench-SNARKS/results` contains **confusion matrices** representing the evaluation performance of both FAN and HAN models.

#### Confusion Matrix File Naming Convention:

Each file follows the format:
`{dataset_abbreviation}_{model}_{model_configuration}_{glove_flag}_confusion_matrix.png`

Refer to the [Folder Naming Convention](#folder-naming-convention) section above for details on each component.

## Other

* `Model Scores.xlsx` contains the **raw score tables** for all evaluated models.

* `Other Resources/HAN diagram.png` is a visual representation of the **Hierarchical Attention Network (HAN)** architecture, taken from the original paper:
  [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

* To run models that use GloVe embeddings, make sure to download the following file:
  [`glove.6B.100d.txt`](https://nlp.stanford.edu/data/glove.6B.zip)
  
  *(Extract it from the ZIP archive linked above.)*

### Raw Data

The datasets used for training and evaluation were obtained from the following sources:

* `\isarcasm-data`
  Source: [iSarcasmEval Dataset](https://github.com/iabufarha/iSarcasmEval/tree/main)

* `\sarcasm_v2`
  Source: [Sarcasm Corpus V2](https://github.com/soraby/sarcasm2)

These folders contain the original data used to create the processed training, validation, and test splits.

## Setup & Requirements

Ensure you have **Python 3.x** installed. The key Python libraries required for this project include:

* `transformers`
* `torch` *(PyTorch, preferably with CUDA support for GPU acceleration)*
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

### Installation

You can install the required packages using `pip`:

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
```

> ⚠️ For GPU acceleration, make sure to install the appropriate version of PyTorch compatible with your CUDA version. Visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.