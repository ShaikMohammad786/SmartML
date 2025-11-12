import pandas as pd
import numpy as np
from preprocessing import Preproccessor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score,RandomizedSearchCV,StratifiedKFold
from typing import Dict, Any



class Classification_Training:

    
    def __init__(self,
                 X_train, y_train,
                 X_test, y_test,
                 X_val, y_val,
                 random_state: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state

        # candidate models (preprocessed)
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "SVM": SVC(random_state=self.random_state, probability=False),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state)
        }

        # sane default param grids (keep small for speed; expand if needed)
        self.param_grids = {
            "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
            "KNN": {"n_neighbors": [3,5,7], "weights": ["uniform","distance"]},
            "DecisionTree": {"max_depth": [None,5,10], "criterion": ["gini","entropy"]},
            "SVM": {"C": [0.1,1,10], "kernel": ["linear","rbf"]},
            "RandomForest": {"n_estimators": [100,200], "max_depth": [None,10]},
            "GradientBoosting": {"n_estimators": [100,200], "learning_rate": [0.01,0.1]}
        }

        # containers to hold results
        self.baselines: Dict[str, Any] = {}   # model name -> fitted baseline estimator
        self.tuned: Dict[str, Any] = {}       # model name -> tuned estimator (GridSearchCV.best_estimator_)
        self.results: Dict[str, Dict[str, Any]] = {}  # per-model summary
        self.best_model_name = None
        self.best_model = None
        self.best_val_acc = -np.inf

    # ---------------------------
    # 1) Fit baseline estimators on full training set (mandatory before tuning)
    # ---------------------------
    def train_models(self, cv: int = 5, verbose: bool = True):
        """
        Fit each model on X_train and record baseline train/cv/val accuracies.
        cv: used for reporting cross-validated training accuracy (not for tuning here).
        """
        if self.X_val is None or self.y_val is None:
            raise RuntimeError("Validation set (X_val, y_val) is required for this workflow.")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        rows = []

        for name, model in self.models.items():
            if verbose: print(f"[BASELINE] Fitting {name} on training set...")
            m = clone(model)
            m.fit(self.X_train, self.y_train)
            self.baselines[name] = m

            train_acc = float(accuracy_score(self.y_train, m.predict(self.X_train)))
            try:
                cv_scores = cross_val_score(m, self.X_train, self.y_train, cv=skf, scoring="accuracy", n_jobs=-1)
                cv_mean = float(np.mean(cv_scores))
            except Exception:
                cv_mean = float("nan")

            val_acc = float(accuracy_score(self.y_val, m.predict(self.X_val)))

            rows.append({"model": name, "train_accuracy": train_acc, "cv_train_accuracy": cv_mean, "val_accuracy": val_acc})
            if verbose:
                print(f"  -> {name}: train={train_acc:.4f}, cv_train={cv_mean:.4f}, val={val_acc:.4f}")

        self.results = {r["model"]: {"baseline_train_acc": r["train_accuracy"], "baseline_cv_acc": r["cv_train_accuracy"], "baseline_val_acc": r["val_accuracy"]} for r in rows}
        return pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    # ---------------------------
    # 2) Tune each model with inner CV on train, then evaluate on validation set
    # ---------------------------
    def tune_models(self,
                             inner_cv: int = 3,
                             search_type: str = "grid",
                             random_state: int = None,
                             n_iter: int = 20,
                             n_jobs: int = -1,
                             scoring: str = "accuracy",
                             verbose: bool = True):
        """
        For each model:
          - run GridSearchCV (or RandomizedSearchCV if search_type="random") on X_train (inner CV = inner_cv)
          - obtain best_estimator_ from search
          - evaluate it on X_val; record val accuracy and best params
        Finally select the model with highest validation accuracy.

        Returns a dict with per-model summaries and the selected best model.
        """
        if not self.baselines:
            raise RuntimeError("Call train_baselines() before tune_and_select_best().")

        if random_state is None:
            random_state = self.random_state

        skf_inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

        self.best_model_name = None
        self.best_val_acc = -np.inf
        self.best_model = None

        for name, base_model in self.models.items():
            grid = self.param_grids.get(name, None)
            if grid is None:
                if verbose: print(f"[TUNE] No grid for {name}, skipping tuning.")
                continue

            if verbose: print(f"[TUNE] Searching hyperparameters for {name} (search_type={search_type}) ...")

            try:
                if search_type == "grid":
                    searcher = GridSearchCV(estimator=clone(base_model),
                                            param_grid=grid,
                                            cv=skf_inner,
                                            scoring=scoring,
                                            n_jobs=n_jobs,
                                            refit=True,
                                            verbose=0)
                elif search_type == "random":
                    searcher = RandomizedSearchCV(estimator=clone(base_model),
                                                  param_distributions=grid,
                                                  n_iter=n_iter,
                                                  cv=skf_inner,
                                                  scoring=scoring,
                                                  n_jobs=n_jobs,
                                                  refit=True,
                                                  random_state=random_state,
                                                  verbose=0)
                else:
                    raise ValueError("search_type must be 'grid' or 'random'")

                searcher.fit(self.X_train, self.y_train)

                best_est = searcher.best_estimator_
                best_cv_score = float(searcher.best_score_)   # mean inner-CV score for best params
                val_acc = float(accuracy_score(self.y_val, best_est.predict(self.X_val)))

                # store
                self.tuned[name] = best_est
                self.results.setdefault(name, {})
                self.results[name].update({
                    "best_params": searcher.best_params_,
                    "best_inner_cv_score": best_cv_score,
                    "val_accuracy_after_tuning": val_acc
                })

                if verbose:
                    print(f"  -> {name}: inner-CV={best_cv_score:.4f}, val_acc={val_acc:.4f}, best_params={searcher.best_params_}")

                # select by validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_name = name
                    self.best_model = best_est

            except Exception as e:
                # record failure and continue
                self.results.setdefault(name, {})
                self.results[name]["error"] = str(e)
                if verbose:
                    print(f"  -> Tuning failed for {name}: {e}")
                continue

        if verbose:
            print(f"\n[SELECT] Best model by validation accuracy: {self.best_model_name} (val_acc={self.best_val_acc:.4f})")
        return {
            "per_model_results": self.results,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "best_val_accuracy": self.best_val_acc
        }




class Regression_Training:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.results = []

    def evaluate_model(self, grid: GridSearchCV, name):
        grid.fit(self.X_train, self.y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        preds = best_model.predict(self.X_val)
        self.results.append(
            {
                "Model": name,
                "Best Params": grid.best_params_,
                "R2": r2_score(self.y_val, preds),
                "MAE": mean_absolute_error(self.y_val, preds),
                "RMSE": root_mean_squared_error(self.y_val, preds),
            }
        )
        print(self.results[-1])
        return self

    def train_model(self):
        print("\n[START] Regression Training Pipeline Initiated...\n")

        model_pipelines = {
            "LinearRegression": Pipeline([("regressor", LinearRegression())]),
            "PolynomialRegression": Pipeline([
                ("PolyFeatures", PolynomialFeatures()),
                ("regressor", LinearRegression())
            ]),
            "Ridge": Pipeline([("regressor", Ridge())]),
            "Lasso": Pipeline([("regressor", Lasso(max_iter=10000))]),
            "ElasticNet": Pipeline([("regressor", ElasticNet(max_iter=10000))]),
            "KNN": Pipeline([("regressor", KNeighborsRegressor())]),
            "DecisionTree": Pipeline([("regressor", DecisionTreeRegressor())]),
            "SVR": Pipeline([("regressor", SVR())]),
            "RandomForest": Pipeline([("regressor", RandomForestRegressor())]),
            "GradientBoosting": Pipeline([("regressor", GradientBoostingRegressor())]),
        }

        param_grids = {
            "LinearRegression": {},
            "PolynomialRegression": {"PolyFeatures__degree": [2, 3, 4]},
            "KNN": {
                "regressor__n_neighbors": [3, 5, 7, 9],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],
            },
            "Ridge": {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Lasso": {"regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
            "ElasticNet": {
                "regressor__alpha": [0.001, 0.01, 0.1, 1.0],
                "regressor__l1_ratio": [0.1, 0.5, 0.9],
            },
            "DecisionTree": {
                "regressor__max_depth": [3, 5, 10, None],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
            },
            "SVR": {
                "regressor__kernel": ["linear", "rbf", "poly"],
                "regressor__C": [0.1, 1, 10],
                "regressor__epsilon": [0.01, 0.1, 1],
            },
            "RandomForest": {
                "regressor__n_estimators": [100, 200, 300],
                "regressor__max_depth": [5, 10, 20, None],
                "regressor__min_samples_split": [2, 5],
                "regressor__min_samples_leaf": [1, 2],
            },
            "GradientBoosting": {
                "regressor__n_estimators": [100, 200, 300],
                "regressor__learning_rate": [0.01, 0.05, 0.1],
                "regressor__max_depth": [3, 5, 8],
            },
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=18)

        print("[INFO] Models to be tuned:", ", ".join(model_pipelines.keys()), "\n")

        for name, model in model_pipelines.items():
            print(f"[TUNE] Now tuning model: {name}")

            if len(param_grids[name]) == 0:
                # No hyperparameters → fit directly
                print(f"  -> No params to tune for {name}. Fitting default model...")
                model.fit(self.X_train, self.y_train)
                preds = model.predict(self.X_val)
                result = {
                    "Model": name,
                    "Best Params": None,
                    "R2": r2_score(self.y_val, preds),
                    "MAE": mean_absolute_error(self.y_val, preds),
                    "RMSE": root_mean_squared_error(self.y_val, preds),
                }
                self.results.append(result)
                print(f"  -> Results: R2={result['R2']:.4f}, MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}\n")
                continue

            print(f"  -> Starting GridSearchCV for {name} with {len(param_grids[name])} params...")
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=kf,
                scoring="r2",
                n_jobs=-1,
            )
            # Use the same evaluate_model() for consistency
            self.evaluate_model(grid, name)
            print("")

        print("\n[SUMMARY] All Models Trained and Evaluated.\n")

        results_df = pd.DataFrame(self.results).sort_values("R2", ascending=False)
        print(results_df)

        best_row = results_df.iloc[0]
        print(f"\n[SELECT] ✅ Best Model: {best_row['Model']} (R2={best_row['R2']:.4f})")

        results_df["Dataset_ID"] = "dataset_01"
        results_df.to_csv("meta_dataset_results.csv", mode="a", index=False)

        print("\n[COMPLETE] Results saved to 'meta_dataset_results.csv'")
        return results_df



# Linear Regression (multiple , polynomial) , KNN , Guassian naive bayes ,
# D.T , SVM , Random forest , Gradient boost


if __name__ == "__main__":
    dataset_path = "datasets/regression/synthetic_car_prices.csv"
    preprocessor = Preproccessor(dataset_path, "Price")
    X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
        preprocessor.run_preprocessing()
    )
    trainer = Regression_Training(X_train, y_train, X_test, y_test, X_val, y_val)
    trainer.train_model()


    # dataset_path = "datasets/classification/phone_detection.csv"
    # preprocessor = Preproccessor(dataset_path, "price_range")
    # X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
    #     preprocessor.run_preprocessing()
    # )
    # trainer = Classification_Training(X_train, y_train, X_test, y_test, X_val, y_val,42)
    # trainer.train_models()
  
    # trainer.tune_models()
  
  
