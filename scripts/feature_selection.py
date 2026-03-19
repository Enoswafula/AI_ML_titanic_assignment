"""
feature_selection.py
--------------------
Selects the most predictive features via:
  1. Correlation analysis  (remove highly-correlated redundant features)
  2. Random Forest feature importance
  3. Recursive Feature Elimination (RFE) — optional / extra credit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_engineered(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # Encode any remaining boolean columns
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=["Survived"], errors="ignore")
    y = df["Survived"] if "Survived" in df.columns else None
    return X, y


# ── 1. Correlation ─────────────────────────────────────────────────────────────
def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.90) -> list[str]:
    """Return list of features to DROP due to high pairwise correlation."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop


def plot_correlation_heatmap(X: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(X.corr(), dtype=bool))
    sns.heatmap(X.corr(), mask=mask, cmap="coolwarm", center=0,
                linewidths=0.5, annot=False)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved → {save_path}")
    plt.close()


# ── 2. Random Forest importance ────────────────────────────────────────────────
def random_forest_importance(X: pd.DataFrame, y: pd.Series,
                              n_estimators: int = 200,
                              top_n: int = 20) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances


def plot_feature_importance(importances: pd.Series, top_n: int = 20, save_path: str = None):
    top = importances.head(top_n)
    plt.figure(figsize=(10, 6))
    top.sort_values().plot(kind="barh", color="steelblue")
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Importance plot saved → {save_path}")
    plt.close()


# ── 3. RFE (extra credit) ──────────────────────────────────────────────────────
def rfe_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> list[str]:
    rf   = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe  = RFE(estimator=rf, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    return selected


# ── Pipeline ───────────────────────────────────────────────────────────────────
def select_features(engineered_path: str,
                    output_path: str = None,
                    plots_dir: str = ".") -> list[str]:
    import os
    os.makedirs(plots_dir, exist_ok=True)

    X, y = load_engineered(engineered_path)

    # Step 1 — drop high-correlation features
    to_drop = remove_correlated_features(X)
    print(f"High-correlation drop ({len(to_drop)}): {to_drop}")
    X_reduced = X.drop(columns=to_drop)

    # Correlation heatmap
    plot_correlation_heatmap(X_reduced, save_path=f"{plots_dir}/correlation_heatmap.png")

    # Step 2 — RF importance
    importances = random_forest_importance(X_reduced, y)
    plot_feature_importance(importances, save_path=f"{plots_dir}/feature_importance.png")
    print("\nTop 15 features by RF importance:")
    print(importances.head(15).to_string())

    # Step 3 — RFE
    rfe_features = rfe_selection(X_reduced, y, n_features=15)
    print(f"\nRFE selected features ({len(rfe_features)}): {rfe_features}")

    # Final feature list — union of top-15 RF and RFE selection
    top15_rf = importances.head(15).index.tolist()
    final_features = list(set(top15_rf) | set(rfe_features))
    print(f"\nFinal selected features ({len(final_features)}): {sorted(final_features)}")

    if output_path:
        pd.Series(sorted(final_features)).to_csv(output_path, index=False, header=["feature"])
        print(f"Feature list saved → {output_path}")

    return final_features


if __name__ == "__main__":
    select_features(
        engineered_path="../data/train_engineered.csv",
        output_path="../data/selected_features.csv",
        plots_dir="../notebooks/plots"
    )
