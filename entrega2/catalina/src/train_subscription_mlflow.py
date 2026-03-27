import os
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/shopping_behavior_updated.csv")
    parser.add_argument("--experiment", default="shopping-subscription-e2")
    parser.add_argument("--model", choices=["dummy", "logreg", "rf"], default="logreg")
    parser.add_argument("--drop_discount", action="store_true",
                        help="Si se activa, se excluye Discount Applied y Promo Code Used (escenario pre-intervención).")

    # hiperparámetros
    parser.add_argument("--C", type=float, default=1.0)                 # logreg
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--n_estimators", type=int, default=400)        # rf
    parser.add_argument("--max_depth", type=int, default=12)            # rf

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_model_path", default="models/subscription_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    # Limpieza
    if "Customer ID" in df.columns:
        df = df.drop(columns=["Customer ID"])

    y = df["Subscription Status"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Subscription Status"])

    # Promo Code Used es redundante con Discount Applied -> quitar por defecto
    if "Promo Code Used" in X.columns:
        X = X.drop(columns=["Promo Code Used"])

    feature_set = "with_discount"
    if args.drop_discount:
        if "Discount Applied" in X.columns:
            X = X.drop(columns=["Discount Applied"])
        feature_set = "no_discount"  # pre-intervención

    # columnas num/cat
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if args.model == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
        params = {"strategy": "most_frequent"}
    elif args.model == "logreg":
        cw = None if args.class_weight == "none" else "balanced"
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight=cw)
        params = {"C": args.C, "class_weight": args.class_weight}
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}

    pipe = Pipeline([("prep", preprocessor), ("model", clf)])

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.model}_{feature_set}"):
        mlflow.set_tag("feature_set", feature_set)
        mlflow.log_param("model", args.model)
        for k, v in params.items():
            mlflow.log_param(k, v)

        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", auc)

        os.makedirs("artifacts", exist_ok=True)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        cm_path = "artifacts/confusion_matrix.png"
        save_fig(cm_path)
        mlflow.log_artifact(cm_path)

        # ROC curve artifact
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        roc_path = "artifacts/roc_curve.png"
        save_fig(roc_path)
        mlflow.log_artifact(roc_path)

        # Log model + save for later (tablero)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
        joblib.dump(pipe, args.save_model_path)
        mlflow.log_artifact(args.save_model_path)

        print("✅ Run listo:", {"accuracy": acc, "f1": f1, "roc_auc": auc, "feature_set": feature_set})


if __name__ == "__main__":
    main()
