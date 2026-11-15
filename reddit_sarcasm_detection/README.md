# Reddit Sarcasm Detection w/ BERT

This project studies binary sarcasm classification on Reddit comments from the SARC-based Kaggle dataset. The notebook compares a lightweight TF-IDF + Logistic Regression baseline with fine-tuned `bert-base-uncased`, evaluates whether parent-comment context helps, and examines errors across subreddits and vote metadata.

## Experiments

- Stratified 80/10/10 train/validation/test split shared by all model variants.
- TF-IDF word uni/bi-grams + Logistic Regression baseline.
- BERT fine-tuning with and without the parent comment as context.
- Held-out accuracy, sarcastic-class F1, inference latency, confusion matrix, and metadata slices.

The notebook intentionally reports measured outcomes rather than claiming BERT or context is necessarily better. Subreddit and voting slices are descriptive and can reflect dataset composition and sampling.

## Results

On the shared 10,000-comment test set (CPU, seed 42, 3 epochs per BERT run):

| Model                        | Context        | Accuracy | Sarcastic F1 | Latency / comment |
| ---------------------------- | -------------- | -------- | ------------ | ----------------- |
| TF-IDF + Logistic Regression | Reply only     | 68.93%   | 0.6809       | 0.0228 ms         |
| BERT (`bert-base-uncased`)   | Reply only     | 74.29%   | 0.7406       | not measured      |
| BERT (`bert-base-uncased`)   | Parent + reply | 74.67%   | 0.7482       | 42.93 ms          |

- **BERT vs. baseline:** reply-only BERT gains 5.36 points in accuracy and 0.0597 F1; parent-context BERT gains 5.74 points and 0.0673 F1.
- **Parent context:** +0.38 points accuracy and +0.0076 F1 over reply-only BERT. Each setting was trained once, no significance test was run, and the pair input shared the 64-token limit, so treat this as suggestive rather than conclusive.
- **Training:** reply-only BERT took about 2.9 hours per epoch on CPU. Its best validation F1 was 0.7485 at epoch 2, and validation loss rose in epoch 3. Training time and per-epoch validation metrics for the parent-context run were not recorded.
- **Latency:** the 42.93 ms figure is for the parent-context model on CPU at batch size 64, including tokenization, from a single run. It is roughly 1,900 times the baseline's latency. Reply-only BERT latency was not measured.
- **Error analysis (parent-context model):** 2,533 of 10,000 test comments were misclassified. Accuracy rose with reply length, from 71.8% (1-5 words) to 77.8% (21-40 words). Error rates varied little across score and upvote quantiles; the downvote column collapsed into a single bucket. Subreddit slices are small and descriptive only.
- **Confusion matrix (reply-only model):** 3,758 TN, 1,212 FP, 1,359 FN, 3,671 TP (sarcastic precision 0.7518, recall 0.7298).

The confusion matrix and precision/recall describe the reply-only model, while the latency, length, subreddit, and vote analyses ran on the parent-context model, which was the model loaded when those cells executed. The results are preserved in `results_snapshot.json` and `results_report.md` goes into more detail on analysis and the overall process.

## Project contents

- `sarcasm_detection_bert.ipynb` — experiment notebook
- `train.py` — standalone baseline and BERT training/evaluation script
- `results_report.md` — full analysis with figures covering the baseline and both BERT runs
- `requirements.txt` — Python dependencies
- `results_snapshot.json` — saved metrics and run status from the latest notebook outputs

## Run

Use Python 3.10+ in Jupyter or Google Colab. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook in Jupyter or Colab, or use the standalone training script:

```bash
python train.py --samples 100000 --contexts both --output-dir outputs
```

By default, the script downloads `danofer/sarcasm` through KaggleHub; configure Kaggle credentials if prompted. To use a local dataset file, pass `--csv path/to/train-balanced-sarcasm.csv`. A CUDA GPU is strongly recommended for the two BERT runs. Reduce `--samples` or `--epochs` for quicker experiments; `--samples 0` uses all cleaned rows. See `python train.py --help` for configuration options.

## Reproducibility and outputs

The experiments use seed 42 by default, stratified splits, and validation-F1 checkpoint selection. `train.py` saves both BERT checkpoints, test predictions, `baseline_metrics.csv`, `comparison_metrics.csv`, and `results.json` under the chosen output directory. The notebook also saves context-specific checkpoints when the corresponding training cells complete. Set `--samples 0` (or notebook `N_SAMPLES = None`) to use all cleaned rows, subject to available compute and memory. Latency depends on hardware and batch size; record both when reporting it.

## Dataset and reference

The dataset is not included in this repository.

- [Kaggle: Sarcasm (SARC)](https://www.kaggle.com/datasets/danofer/sarcasm).

## Author and License

_Author: Lucy Giang (lucy-giang3), updated Aug. 2026_

This code is shared for free for learning and personal use only. Commercial use is prohibited.
