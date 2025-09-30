#!/usr/bin/env python3
import argparse, sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

MODEL_PATH = Path("sentiment_pipeline.joblib")
LABELS_PATH = Path("sentiment_labels.json")

# ---- tiny built-in dataset ----
BUILTIN = pd.DataFrame({
    "text": [
        "i love this product", "this is amazing", "absolutely fantastic",
        "works great and fast", "super happy with the service", "highly recommend",
        "i hate this", "this is terrible", "absolutely awful",
        "really disappointed", "not worth the money", "bad experience"
    ],
    "label": ["pos","pos","pos","pos","pos","pos","neg","neg","neg","neg","neg","neg"]
})

def load_dataset(user_csv: str | None):
    if user_csv:
        df = pd.read_csv(user_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: text,label")
        return df
    return BUILTIN.copy()

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=200))
    ])

def train(csv_path: str | None, test_size=0.2, random_state=42):
    df = load_dataset(csv_path)
    df["label"] = df["label"].astype(str)
    labs = sorted(df["label"].unique())
    if len(labs) != 2:
        raise ValueError(f"Expected exactly 2 labels, found: {labs}")
    label_map = {labs[0]: 0, labs[1]: 1}
    y = df["label"].map(label_map).values
    X = df["text"].astype(str).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = make_pipeline()

    # ✅ FIX: numpy array + correct variable (ytr)
    class_arr = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=class_arr, y=ytr)
    pipe.named_steps["clf"].set_params(class_weight={0: float(weights[0]), 1: float(weights[1])})

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    acc = accuracy_score(yte, yhat)
    print(f"\n[OK] Trained. Test accuracy: {acc:.3f}\n")
    print(classification_report(yte, yhat, target_names=[labs[0], labs[1]]))

    joblib.dump(pipe, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps({"idx2lab": {0: labs[0], 1: labs[1]}}))
    print(f"\n[Saved] {MODEL_PATH} and {LABELS_PATH}\n")

def _load():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Model files not found. Run training first.")
    pipe = joblib.load(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text())["idx2lab"]
    labels = {int(k): v for k, v in labels.items()}
    return pipe, labels

def predict_one(text: str):
    pipe, labels = _load()
    yhat = int(pipe.predict([text])[0])
    proba = pipe.predict_proba([text])[0] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    print("\nResult")
    print("------")
    print(f"prediction: {labels[yhat]}")
    if proba is not None:
        print("class probabilities:")
        for i, p in enumerate(proba):
            print(f"  {labels[i]:>3s}: {p:.3f}")
    print("")

def predict_batch(csv_in: str, out_csv: str):
    pipe, labels = _load()
    df = pd.read_csv(csv_in)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column")
    texts = df["text"].astype(str)
    preds = pipe.predict(texts)
    df["pred_label"] = [labels[int(i)] for i in preds]
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs = pipe.predict_proba(texts)
        df["prob_neg"] = probs[:,0]
        df["prob_pos"] = probs[:,1]
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

def evaluate(csv_path: str | None, test_size=0.2, random_state=42):
    df = load_dataset(csv_path)
    df["label"] = df["label"].astype(str)
    labs = sorted(df["label"].unique())
    label_map = {labs[0]: 0, labs[1]: 1}
    y = df["label"].map(label_map).values
    X = df["text"].astype(str).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else make_pipeline().fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    print(f"\nEval accuracy: {acc:.3f}\n")
    print(classification_report(yte, yhat, target_names=[labs[0], labs[1]]))

def menu():
    print("\nSentiment CLI (pos/neg)")
    print("-----------------------")
    while True:
        print("[T] Train  [E] Evaluate  [P] Predict one  [B] Batch CSV  [Q] Quit")
        c = input("> ").strip().lower()
        try:
            if c == "t":
                path = input("CSV path (or empty for built-in): ").strip().strip('"')
                train(path if path else None)
            elif c == "e":
                path = input("CSV path (or empty for built-in): ").strip().strip('"')
                evaluate(path if path else None)
            elif c == "p":
                text = input("Enter text: ").strip()
                predict_one(text)
            elif c == "b":
                incsv = input("Input CSV path (must have 'text' col): ").strip().strip('"')
                outcsv = input("Output CSV path: ").strip().strip('"')
                predict_batch(incsv, outcsv)
            elif c == "q":
                print("Bye!"); return
            else:
                print("  ⚠️ Use T/E/P/B/Q.")
        except Exception as e:
            print(f"  ⚠️ {e}")

def main():
    ap = argparse.ArgumentParser(description="Sentiment Analysis CLI (TF-IDF + Logistic Regression)")
    sub = ap.add_subparsers(dest="cmd")

    p_tr = sub.add_parser("train");  p_tr.add_argument("--csv", default=None)
    p_ev = sub.add_parser("eval");   p_ev.add_argument("--csv", default=None)
    p_po = sub.add_parser("predict"); p_po.add_argument("--text", required=True)
    p_pb = sub.add_parser("predict-batch"); p_pb.add_argument("--in", dest="incsv", required=True); p_pb.add_argument("--out", dest="outcsv", required=True)

    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive or args.cmd is None:
        menu(); return
    if args.cmd == "train": train(args.csv)
    elif args.cmd == "eval": evaluate(args.csv)
    elif args.cmd == "predict": predict_one(args.text)
    elif args.cmd == "predict-batch": predict_batch(args.incsv, args.outcsv)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted."); sys.exit(0)
