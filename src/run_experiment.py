"""
Main experiment runner.

STEP 1 (once, after downloading the real CSV):
    python run_experiment.py --inspect /path/to/dod_construction.csv
    -> prints your file's real column names so you can fix config.py if any
       field failed to auto-match.

STEP 2 (the actual experiment):
    python run_experiment.py --data /path/to/dod_construction.csv
"""
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, \
    accuracy_score, f1_score, roc_auc_score

from data_pipeline import load_raw, match_columns, engineer_features, clean_and_split, add_risk_label
from mtl_model import MultiTaskNet
from baselines import train_single_task_baselines


def inspect(path):
    df = load_raw(path)
    print("\nColumn names in your file:\n")
    for c in df.columns:
        print(" -", c)
    print("\nCompare these against config.py's COLUMN_CANDIDATES and add any "
          "exact header text that's missing, then re-run with --data.")


class StdScalerTarget:
    """Standardize the cost_growth_ratio target after a signed-log transform
    (sign(x)*log1p(|x|)) — the raw ratio is extremely heavy-tailed even after
    winsorizing, which was hurting MSE-based training stability."""
    def fit(self, y):
        y_t = np.sign(y) * np.log1p(np.abs(y))
        self.mean, self.std = y_t.mean(), y_t.std() + 1e-9
        return self

    def transform(self, y):
        y_t = np.sign(y) * np.log1p(np.abs(y))
        return (y_t - self.mean) / self.std

    def inverse(self, y):
        y_t = y * self.std + self.mean
        return np.sign(y_t) * (np.expm1(np.abs(y_t)))


def run(path, prepared=False):
    if prepared:
        # already project-level (see prepare_real_data.py) — just add risk label
        feat = load_raw(path)
        feat = add_risk_label(feat)
    else:
        df = load_raw(path)
        cols = match_columns(df)
        feat = engineer_features(df, cols)
    X_train, X_test, y_train, y_test, feature_names = clean_and_split(feat)

    # log-signed-scale the cost target only; schedule and risk are now both
    # binary classification (0/1), no scaling needed
    cost_scaler = StdScalerTarget().fit(y_train["cost_growth_ratio"])
    y_cost_tr = cost_scaler.transform(y_train["cost_growth_ratio"])
    y_sched_tr = y_train["schedule_overrun_label"].astype(float)
    y_risk_tr = y_train["risk_label"].astype(float)

    print(f"\nFeatures used ({len(feature_names)}): {feature_names}\n")

    print("=== Training multi-task network (joint loss) ===")
    mtl = MultiTaskNet(n_features=X_train.shape[1])

    # weight the two classification heads (real signal, AUC ~0.65-0.70) more
    # than the still-noisy cost regression head
    
    mtl.fit(X_train, y_cost_tr, y_sched_tr, y_risk_tr,
            epochs=150, batch_size=1024, lr=2e-3, weights=(0.5, 1.5, 1.5))

    cost_pred_s, sched_pred, risk_pred = mtl.predict(X_test)
    cost_pred = cost_scaler.inverse(cost_pred_s)

    print("\n=== Training single-task baselines ===")
    baselines = train_single_task_baselines(X_train, y_train)

    results = {}

    def reg_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
        }

    def clf_metrics(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC": auc,
        }

    results["cost_MTL"] = reg_metrics(y_test["cost_growth_ratio"], cost_pred)
    results["cost_RF"] = reg_metrics(y_test["cost_growth_ratio"], baselines["cost_rf"].predict(X_test))
    results["cost_MLP_single"] = reg_metrics(y_test["cost_growth_ratio"], baselines["cost_mlp"].predict(X_test))

    results["sched_MTL"] = clf_metrics(y_test["schedule_overrun_label"], sched_pred)
    results["sched_RF"] = clf_metrics(y_test["schedule_overrun_label"], baselines["sched_rf"].predict_proba(X_test)[:, 1])
    results["sched_MLP_single"] = clf_metrics(y_test["schedule_overrun_label"], baselines["sched_mlp"].predict_proba(X_test)[:, 1])

    results["risk_MTL"] = clf_metrics(y_test["risk_label"], risk_pred)
    results["risk_RF"] = clf_metrics(y_test["risk_label"], baselines["risk_rf"].predict_proba(X_test)[:, 1])
    results["risk_MLP_single"] = clf_metrics(y_test["risk_label"], baselines["risk_mlp"].predict_proba(X_test)[:, 1])

    print("\n================ RESULTS ================")
    print(f"{'Model':<20}{'Metric 1':<20}{'Metric 2':<20}{'Metric 3':<20}")
    for name, m in results.items():
        vals = "  ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"{name:<20}{vals}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the DoD construction CSV")
    parser.add_argument("--inspect", type=str, help="Just print column names and exit")
    parser.add_argument("--prepared", action="store_true",
                         help="Set if --data is already project-level (output of prepare_real_data.py)")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
    elif args.data:
        run(args.data, prepared=args.prepared)
    else:
        parser.print_help()
