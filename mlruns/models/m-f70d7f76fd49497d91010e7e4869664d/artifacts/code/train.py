import logging
import os
import random
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from xgboost import XGBRegressor

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

logging.getLogger("mlflow").setLevel(logging.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as prep

SEED = 42
FOLDS = 10
TEST_SIZE = 0.2
EXPERIMENT = "Actividad5-Airbnb-NYC"
REGISTERED_MODEL = "airbnb-nyc-price"
TRACKING_URI = f"sqlite:///{prep.ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (prep.ROOT / "mlruns").as_uri()
EVIDENCE = prep.ROOT / "evidencias"
torch.set_num_threads(1)
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
CODE_PATHS = [str(Path(__file__).resolve()), str(Path(prep.__file__).resolve())]

COLORS = {"Ridge": "#2a78d6", "XGBoost": "#eb6834", "MLP": "#1baf7a"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "lines.linewidth": 2,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden=64, lr=1e-3, epochs=30, batch_size=512, dropout=0.1, random_state=SEED):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def _build(self, n_features):
        return nn.Sequential(
            nn.Linear(n_features, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), device=DEVICE)
        self.n_features_in_ = X.shape[1]
        model = self._build(self.n_features_in_).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        rng = np.random.default_rng(self.random_state)
        model.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(rng.permutation(len(X))).to(DEVICE)
            for start in range(0, len(X), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(X[batch]), y[batch])
                loss.backward()
                optimizer.step()
        self.model_ = model.cpu().eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(np.asarray(X, dtype=np.float32))
            return self.model_(X).numpy().ravel()


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), prep.NUMERIC),
        ("cat", OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=False), prep.CATEGORICAL),
    ])


def in_dollars(metric):
    return lambda y_true, y_pred: metric(np.expm1(y_true), np.expm1(y_pred))


SCORING = {
    "rmse": make_scorer(in_dollars(root_mean_squared_error), greater_is_better=False),
    "mae": make_scorer(in_dollars(mean_absolute_error), greater_is_better=False),
    "r2": make_scorer(in_dollars(r2_score)),
}


def model_specs():
    return {
        "Ridge": {
            "estimator": Ridge(random_state=SEED),
            "grid": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
            "n_jobs": -1,
        },
        "XGBoost": {
            "estimator": XGBRegressor(tree_method="hist", n_jobs=1, random_state=SEED),
            "grid": {
                "model__n_estimators": [300, 600, 1000],
                "model__max_depth": [4, 6, 8, 10],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            "n_jobs": -1,
        },
        "MLP": {
            "estimator": TorchMLPRegressor(),
            "grid": {"model__hidden": [64, 128, 256], "model__lr": [1e-3, 3e-3]},
            "n_jobs": 1,
        },
    }


def regression_metrics(y_true_log, y_pred_log):
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def short_params(params):
    return {key.replace("model__", ""): value for key, value in params.items()}


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)


def plot_predictions(name, y_true_log, y_pred_log):
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_true, y_pred, s=8, alpha=0.25, color=COLORS[name], edgecolors="none")
    axes[0].plot([0, prep.PRICE_MAX], [0, prep.PRICE_MAX], color=INK_SECONDARY, linewidth=1.2, linestyle="--")
    axes[0].set(title=f"{name}: precio real vs predicho", xlabel="Precio real (USD)", ylabel="Precio predicho (USD)", xlim=(0, prep.PRICE_MAX), ylim=(0, prep.PRICE_MAX))
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=80, color=COLORS[name], edgecolor=SURFACE, linewidth=0.5)
    axes[1].axvline(0, color=INK_SECONDARY, linewidth=1.2, linestyle="--")
    axes[1].set(title=f"{name}: distribución de residuales", xlabel="Real menos predicho (USD)", ylabel="Frecuencia", xlim=(-500, 500))
    path = EVIDENCE / f"pred_{name.lower()}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def train_model(name, spec, X_train, y_train, X_test, y_test, cv, data_sha):
    pipeline = Pipeline([("prep", preprocessor()), ("model", spec["estimator"])])
    search = GridSearchCV(pipeline, spec["grid"], scoring=SCORING, refit="rmse", cv=cv, n_jobs=spec["n_jobs"])
    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning", "device": DEVICE if name == "MLP" else "cpu"})
        mlflow.log_params({
            "model": name,
            "cv_folds": FOLDS,
            "test_size": TEST_SIZE,
            "seed": SEED,
            "target": "log1p(price)",
            "data_version": prep.DATA_VERSION,
            "data_sha256": data_sha,
            "grid": str(short_params(spec["grid"])),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        search.fit(X_train, y_train)
        results = pd.DataFrame(search.cv_results_)
        for i, row in results.iterrows():
            with mlflow.start_run(run_name=f"{name}-{i:02d}", nested=True):
                mlflow.set_tag("stage", "grid_candidate")
                mlflow.log_params(short_params(row["params"]))
                mlflow.log_metrics({
                    "cv_rmse_mean": -row["mean_test_rmse"],
                    "cv_rmse_std": row["std_test_rmse"],
                    "cv_mae_mean": -row["mean_test_mae"],
                    "cv_r2_mean": row["mean_test_r2"],
                    "fit_time_mean": row["mean_fit_time"],
                })
        best = search.best_index_
        fold_rmse = np.array([-results.loc[best, f"split{k}_test_rmse"] for k in range(FOLDS)])
        y_pred = search.predict(X_test)
        test = regression_metrics(y_test, y_pred)
        mlflow.log_params({f"best_{key}": value for key, value in short_params(search.best_params_).items()})
        mlflow.log_metrics({
            "cv_rmse_mean": fold_rmse.mean(),
            "cv_rmse_std": fold_rmse.std(),
            "cv_mae_mean": -results.loc[best, "mean_test_mae"],
            "cv_r2_mean": results.loc[best, "mean_test_r2"],
            "test_mae": test["mae"],
            "test_rmse": test["rmse"],
            "test_r2": test["r2"],
            "grid_size": len(results),
            "search_time_sec": float(results["mean_fit_time"].sum() * FOLDS),
        })
        cv_path = EVIDENCE / f"cv_results_{name.lower()}.csv"
        results.drop(columns="params").to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), "cv")
        mlflow.log_artifact(str(plot_predictions(name, y_test, y_pred)), "plots")
        signature = infer_signature(X_test.head(100), search.predict(X_test.head(100)))
        model_info = mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            code_paths=CODE_PATHS,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    print(f"{name}: mejores hiperparámetros {short_params(search.best_params_)}")
    print(f"{name}: RMSE CV {fold_rmse.mean():.2f} +/- {fold_rmse.std():.2f} | test MAE {test['mae']:.2f} RMSE {test['rmse']:.2f} R2 {test['r2']:.3f}")
    return {
        "name": name,
        "run_id": parent.info.run_id,
        "model_uri": model_info.model_uri,
        "best_params": short_params(search.best_params_),
        "fold_rmse": fold_rmse,
        "test": test,
        "cv_rmse_mean": fold_rmse.mean(),
        "cv_rmse_std": fold_rmse.std(),
        "cv_r2_mean": results.loc[best, "mean_test_r2"],
        "grid_size": len(results),
    }


def compare(results):
    folds = pd.DataFrame({r["name"]: r["fold_rmse"] for r in results})
    friedman = stats.friedmanchisquare(*[folds[c] for c in folds])
    rows = []
    for a, b in combinations(folds.columns, 2):
        t = stats.ttest_rel(folds[a], folds[b])
        w = stats.wilcoxon(folds[a], folds[b])
        rows.append({
            "comparacion": f"{a} vs {b}",
            "diferencia_media_rmse": (folds[a] - folds[b]).mean(),
            "t": t.statistic,
            "p_t_pareada": t.pvalue,
            "w": w.statistic,
            "p_wilcoxon": w.pvalue,
        })
    tests = pd.DataFrame(rows)
    return folds, friedman, tests


def plot_comparison(results, folds):
    names = [r["name"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, label in zip(axes, ["mae", "rmse", "r2"], ["MAE (USD)", "RMSE (USD)", "R²"]):
        values = [r["test"][metric] for r in results]
        bars = ax.bar(names, values, color=[COLORS[n] for n in names], width=0.55, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:.3f}" if metric == "r2" else f"{v:.1f}" for v in values], padding=3, color=INK)
        ax.set(title=f"{label} en test", ylabel=label)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.tight_layout()
    bars_path = EVIDENCE / "comparacion_metricas.png"
    fig.savefig(bars_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    box = ax.boxplot([folds[n] for n in names], tick_labels=names, patch_artist=True, widths=0.5, medianprops={"color": INK, "linewidth": 2})
    for patch, name in zip(box["boxes"], names):
        patch.set(facecolor=COLORS[name], alpha=0.35, edgecolor=COLORS[name], linewidth=2)
    for i, name in enumerate(names, start=1):
        jitter = np.random.default_rng(SEED).normal(0, 0.04, len(folds))
        ax.scatter(i + jitter, folds[name], s=40, color=COLORS[name], edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set(title=f"RMSE por fold ({FOLDS}-fold CV, mismos folds)", ylabel="RMSE (USD)")
    ax.grid(axis="x", visible=False)
    folds_path = EVIDENCE / "rmse_por_fold.png"
    fig.savefig(folds_path)
    plt.close(fig)
    return bars_path, folds_path


def log_comparison(results, folds, friedman, tests, plots):
    summary = pd.DataFrame([{
        "modelo": r["name"],
        "mejores_hiperparametros": str(r["best_params"]),
        "cv_rmse_media": r["cv_rmse_mean"],
        "cv_rmse_std": r["cv_rmse_std"],
        "cv_r2_media": r["cv_r2_mean"],
        "test_mae": r["test"]["mae"],
        "test_rmse": r["test"]["rmse"],
        "test_r2": r["test"]["r2"],
        "run_id": r["run_id"],
    } for r in results])
    paths = {
        "resumen_modelos.csv": summary,
        "rmse_por_fold.csv": folds,
        "pruebas_estadisticas.csv": tests,
    }
    with mlflow.start_run(run_name="Comparacion"):
        mlflow.set_tag("stage", "comparison")
        mlflow.log_params({"models": ",".join(folds.columns), "cv_folds": FOLDS, "data_version": prep.DATA_VERSION})
        mlflow.log_metrics({"friedman_chi2": friedman.statistic, "friedman_p": friedman.pvalue})
        for _, row in tests.iterrows():
            key = row["comparacion"].replace(" vs ", "_vs_")
            mlflow.log_metrics({f"p_ttest_{key}": row["p_t_pareada"], f"p_wilcoxon_{key}": row["p_wilcoxon"]})
        for filename, frame in paths.items():
            path = EVIDENCE / filename
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), "comparacion")
        for plot in plots:
            mlflow.log_artifact(str(plot), "plots")
    return summary


def register_best(results):
    best = min(results, key=lambda r: r["test"]["rmse"])
    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.update_model_version(REGISTERED_MODEL, version.version, f"{best['name']} con RMSE test {best['test']['rmse']:.2f} USD")
    print(f"Modelo registrado: {REGISTERED_MODEL} v{version.version} ({best['name']}) con alias champion")
    return best, version


def export_runs():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    path = EVIDENCE / "mlflow_runs.csv"
    runs.to_csv(path, index=False)
    return path


def run(models=None):
    set_seed()
    set_style()
    EVIDENCE.mkdir(exist_ok=True)
    setup_mlflow()
    df = prep.load_clean()
    data_sha = prep.sha256(prep.CLEAN_PATH)
    X, y = prep.split_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    print(f"Dispositivo para MLP: {DEVICE}")
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas | Folds: {FOLDS}")
    specs = model_specs()
    results = [train_model(name, specs[name], X_train, y_train, X_test, y_test, cv, data_sha) for name in (models or specs)]
    folds, friedman, tests = compare(results)
    plots = plot_comparison(results, folds)
    summary = log_comparison(results, folds, friedman, tests, plots)
    best, version = register_best(results)
    export_runs()
    print(f"Friedman: chi2 {friedman.statistic:.3f}, p {friedman.pvalue:.4g}")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return {"results": results, "summary": summary, "folds": folds, "friedman": friedman, "tests": tests, "best": best, "version": version}


if __name__ == "__main__":
    run()
