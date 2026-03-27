import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/shopping_behavior_updated.csv")
    parser.add_argument("--out_dir", default="reports/eda")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)

    # Limpieza mínima
    if "Customer ID" in df.columns:
        df = df.drop(columns=["Customer ID"])

    target_col = "Subscription Status"
    y = df[target_col].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col])

    # 1) Leakage / post-evento
    leakage_cols = ["Discount Applied", "Promo Code Used"]
    for c in leakage_cols:
        ct = pd.crosstab(df[c], df[target_col], normalize="index")
        ct.to_csv(os.path.join(args.out_dir, f"crosstab_{c}_vs_subscription.csv"))

    dup = pd.crosstab(df["Discount Applied"], df["Promo Code Used"])
    dup.to_csv(os.path.join(args.out_dir, "crosstab_discount_vs_promo.csv"))

    # 2) Comparaciones por grupos (tasa suscripción)
    group_cols = ["Category", "Season", "Shipping Type", "Payment Method",
                  "Frequency of Purchases", "Discount Applied"]

    for c in group_cols:
        tmp = df.groupby(c)[target_col].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
        tmp = tmp.reset_index()
        tmp.columns = [c, "subscription_rate"]
        tmp.to_csv(os.path.join(args.out_dir, f"subscription_rate_by_{c}.csv"), index=False)

        plt.figure()
        plt.bar(tmp[c].astype(str), tmp["subscription_rate"])
        plt.title(f"Subscription rate by {c}")
        plt.xticks(rotation=45, ha="right")
        save_fig(os.path.join(args.out_dir, f"plot_subscription_rate_by_{c}.png"))

    # 3) Ranking rápido (Mutual Information)
    X_enc = X.copy()
    discrete_mask = []
    for col in X_enc.columns:
        if X_enc[col].dtype == "object":
            X_enc[col], _ = pd.factorize(X_enc[col])
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)

    mi = mutual_info_classif(X_enc.values, y.values, discrete_features=discrete_mask, random_state=args.random_state)
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    mi_df.to_csv(os.path.join(args.out_dir, "mutual_information_ranking.csv"), index=False)

    plt.figure()
    top = mi_df.head(10)
    plt.bar(top["feature"], top["mutual_info"])
    plt.title("Top 10 features by Mutual Information (Subscription)")
    plt.xticks(rotation=45, ha="right")
    save_fig(os.path.join(args.out_dir, "plot_top10_mutual_info.png"))

    # 4) Segmentación operativa simple
    def freq_group(v: str) -> str:
        if v in ["Weekly", "Bi-Weekly", "Fortnightly"]:
            return "HighFreq"
        if v == "Monthly":
            return "MidFreq"
        return "LowFreq"

    seg = df.copy()
    seg["freq_group"] = seg["Frequency of Purchases"].apply(freq_group)
    med_prev = seg["Previous Purchases"].median()
    seg["prev_group"] = seg["Previous Purchases"].apply(lambda x: "HighPrev" if x >= med_prev else "LowPrev")
    seg["discount_flag"] = seg["Discount Applied"]

    seg_table = (
        seg.groupby(["freq_group", "prev_group", "discount_flag"])[target_col]
        .agg(n="count", subscription_rate=lambda s: (s == "Yes").mean())
        .reset_index()
    )
    seg_table["expected_subscribers"] = seg_table["n"] * seg_table["subscription_rate"]
    seg_table = seg_table.sort_values("expected_subscribers", ascending=False)
    seg_table.to_csv(os.path.join(args.out_dir, "segment_table.csv"), index=False)

    # 5) Baseline: Dummy vs Logistic Regression
    X_model = X.copy()
    if "Promo Code Used" in X_model.columns:
        X_model = X_model.drop(columns=["Promo Code Used"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    X_train_oh = pd.get_dummies(X_train, drop_first=False)
    X_test_oh = pd.get_dummies(X_test, drop_first=False)
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train_oh, y_train)
    base_proba = baseline.predict_proba(X_test_oh)[:, 1]
    base_pred = baseline.predict(X_test_oh)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_oh, y_train)
    lr_proba = logreg.predict_proba(X_test_oh)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)

    metrics = []
    def add_metrics(name, proba, pred):
        metrics.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba)
        })

    add_metrics("dummy_most_frequent", base_proba, base_pred)
    add_metrics("logreg_balanced", lr_proba, lr_pred)

    pd.DataFrame(metrics).to_csv(os.path.join(args.out_dir, "baseline_vs_logreg_metrics.csv"), index=False)

    cm = confusion_matrix(y_test, lr_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    save_fig(os.path.join(args.out_dir, "plot_confusion_matrix_logreg.png"))

    print("✅ EDA analítico listo. Revisa:", args.out_dir)


if __name__ == "__main__":
    main()