# Duplicate Question Detection — Single-file Demo

This repository contains a compact, single-file demo for detecting duplicate questions using classical NLP features (TF-IDF) and a simple Logistic Regression classifier.

Files
- [duplication_question.py](duplication_question.py) — main single-file script (train, evaluate, interactive predict).

Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install scikit-learn pandas numpy
```

Quick start

- Train and evaluate on the included tiny demo dataset:

```bash
python3 duplication_question.py --train
```

- Enter interactive prediction mode (will train first if needed):

```bash
python3 duplication_question.py --predict
```

- Run with no flags to run a demo train followed by interactive prompts:

```bash
python3 duplication_question.py
```

Notes
- The demo uses a small in-file example dataset for illustration. For real tasks, replace the data-loading section with a larger labeled dataset (e.g., Quora Question Pairs CSV) and increase the feature/model complexity.
- To persist a trained model or add more features (word embeddings, syntactic features), extend the script or convert it into a small package.

Next steps (suggested)
- Add a `requirements.txt` and dataset loader.
- Save/load model with `joblib` or `pickle`.
- Add unit tests and a small web UI for quick demos.
# NLP-project