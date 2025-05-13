# Sarcasm-Detection-in-Text
## Overview

This project explores sarcasm detection in text by fine-tuning pre-trained Transformer models and by training Hierarchical Attention Networks (HAN). It involves training DeBERTa, RoBERTa and HAN models on three distinct datasets: iSarcasmEval, Sarcasm_Corpus_V2, and a combined dataset(iSarcasmEval + Sarcasm_Corpus_V2). The trained models are then evaluated on their respective test sets and further assessed on the BigBench SNARKS benchmark to test their ability to discern sarcasm in a pairwise comparison task.

The primary goals include:
* Fine-tuning robust sarcasm detection models.
* Implementing strategies to handle class imbalance (e.g., weighted loss).
* Evaluating model performance thoroughly using standard metrics and specialized benchmarks.
* Comparing the performance of models trained on different sarcasm datasets.

## Models Used

* **DeBERTa**: `microsoft/deberta-v3-base`
* **RoBERTa**: `roberta-base`

## Datasets

1.  **Fine-tuning Datasets:**
    * **iSarcasmEval**: Contains training, validation, and test sets for sarcasm detection. This dataset exhibited some class imbalance, addressed using a weighted loss function for the RoBERTa and DeBERTa models.
    * **Sarcasm\_Corpus\_V2**: Includes training, validation, and test sets. This dataset was noted to be more balanced.
    * **Combined Dataset**: A merger of the iSarcasmEval and Sarcasm\_Corpus\_V2 training/validation/test sets.
    Each dataset is expected in CSV format with 'text' and 'sarcastic' (0 or 1) columns.

2.  **Evaluation Benchmark:**
    * **BigBench SNARKS**: Used for evaluating the fine-tuned models on an unseen, out-of-distribution task. The SNARKS task involves determining which of two provided statements, (a) or (b), is more sarcastic. The `task.json` file from this benchmark is used for evaluation.

## File Structure

* `DeBERTA-model.ipynb`: Jupyter Notebook for fine-tuning and evaluating the DeBERTa-v3-base model on the iSarcasmEval, Sarcasm_Corpus_V2, and combined datasets.
* `RoBERTa_model.ipynb`: Jupyter Notebook for fine-tuning and evaluating a RoBERTa-base model.
* `Bigbench-Snarks-evaluation.ipynb`: Script/Notebook to evaluate the sarcasm detection models (trained by the above notebooks) on the BigBench SNARKS dataset.
* `/data/` (Example directory structure, please adapt)
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
    * `task.json` (from BigBench SNARKS)
* `/results_iSarcasm/`, `/results_Sarcasm_Corpus_V2/`, `/results_combined/`: Output directories where trained model checkpoints, tokenizers, and logs are saved.

## Setup & Requirements

Ensure you have Python 3 installed. Key libraries include:

* `transformers`
* `torch` (PyTorch, preferably with CUDA support for GPU acceleration)
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`

You can typically install these using pip:
```bash
pip install transformers torch torchvision torchaudio scikit-learn pandas numpy matplotlib seabornv








Raw Data:
- isarcasm-data
- sarcasm_v2

Normalised Data:
- iSarcasmEval
- Sarcasm_Corpus_V2
- combined
