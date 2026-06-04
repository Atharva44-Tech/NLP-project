#!/usr/bin/env python3
"""
Single-file Duplicate Question Detection

Requirements:
  pip install scikit-learn pandas numpy

Usage examples:
  - Train & evaluate demo dataset:
	  python3 duplication_question.py --train
  - Interactive prediction:
	  python3 duplication_question.py --predict

This file contains a tiny example dataset, TF-IDF feature extraction,
and a simple logistic regression classifier using cosine similarity
and L1 difference of TF-IDF vectors as features.
"""
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


SAMPLE_DATA = [
	("How do I cook rice?", "What is the best way to cook rice?", 1),
	("How to lose weight?", "What is the fastest way to lose weight?", 1),
	("What is AI?", "Define artificial intelligence.", 1),
	("How to bake a cake?", "What is the capital of France?", 0),
	("Best way to learn Python", "How can I learn Java?", 0),
	("How to tie a tie?", "How to tie a bow tie?", 1),
	("What causes rain?", "How is rain formed?", 1),
	("Where is the Eiffel Tower?", "How tall is the Eiffel Tower?", 0),
	("How to lose belly fat?", "How do I reduce belly fat quickly?", 1),
	("What is quantum mechanics?", "Explain quantum physics briefly.", 1),
	("How to invest in stocks?", "Tips for saving money?", 0),
	("How to fix a bike flat tire?", "How to change a bike tube?", 1),
]


def normalize(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = text.lower().strip()
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"[^\w\s]", "", text)
	return text


def build_dataframe(pairs):
	df = pd.DataFrame(pairs, columns=["q1", "q2", "label"]) if not isinstance(pairs, pd.DataFrame) else pairs
	df = df.copy()
	df["q1"] = df["q1"].astype(str).map(normalize)
	df["q2"] = df["q2"].astype(str).map(normalize)
	return df


def extract_features(df, vectorizer=None):
	texts = pd.concat([df["q1"], df["q2"]])
	if vectorizer is None:
		vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit(texts)
	v1 = vectorizer.transform(df["q1"])
	v2 = vectorizer.transform(df["q2"])
	cos_sim = np.array([cosine_similarity(v1[i], v2[i])[0, 0] for i in range(v1.shape[0])])
	l1 = np.abs(v1 - v2).sum(axis=1).A1
	X = np.vstack([cos_sim, l1]).T
	return X, vectorizer


def train_and_evaluate(df):
	X, vectorizer = extract_features(df)
	y = df["label"].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	clf = LogisticRegression(solver="liblinear")
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	print("Evaluation on held-out set:")
	print(f"  Accuracy: {acc:.3f}")
	print(f"  F1-score: {f1:.3f}")
	print("\nDetailed report:")
	print(classification_report(y_test, y_pred))
	return clf, vectorizer


def interactive_predict(clf, vectorizer):
	print("Enter two questions (or empty to quit)")
	while True:
		q1 = input("Question 1: ").strip()
		if not q1:
			break
		q2 = input("Question 2: ").strip()
		if not q2:
			break
		df = build_dataframe([(q1, q2, 0)])
		X, _ = extract_features(df, vectorizer=vectorizer)
		prob = clf.predict_proba(X)[0, 1] if hasattr(clf, "predict_proba") else None
		pred = clf.predict(X)[0]
		print(f"  Predicted duplicate: {bool(pred)}; probability(dup)={prob:.3f}" if prob is not None else f"  Predicted duplicate: {bool(pred)}")


def demo_run():
	df = build_dataframe(pd.DataFrame(SAMPLE_DATA, columns=["q1", "q2", "label"]))
	clf, vec = train_and_evaluate(df)
	print("\nDemo interactive prediction (try similar/different questions):")
	interactive_predict(clf, vec)


def main():
	parser = argparse.ArgumentParser(description="Duplicate question detection (single-file demo)")
	parser.add_argument("--train", action="store_true", help="Train and evaluate on demo dataset")
	parser.add_argument("--predict", action="store_true", help="Enter interactive prediction mode")
	args = parser.parse_args()

	df = build_dataframe(pd.DataFrame(SAMPLE_DATA, columns=["q1", "q2", "label"]))
	clf, vec = train_and_evaluate(df) if args.train else (None, None)
	if args.predict:
		if clf is None or vec is None:
			clf, vec = train_and_evaluate(df)
		interactive_predict(clf, vec)
	if not args.train and not args.predict:
		print("No flag passed — running demo train & interactive session")
		demo_run()


if __name__ == "__main__":
	main()

