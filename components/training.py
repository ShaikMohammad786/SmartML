import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


import pandas as pd
import numpy as np
import random
from components.preprocessing import Preproccessor
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

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score,RandomizedSearchCV,StratifiedKFold
from typing import Dict, Any, Optional, Iterable
from meta_learning.meta_features_extraction import meta_features_extract_class,meta_features_extract_reg,meta_features_extract_clust



class Classification_Training:

    
    def __init__(self,
                 X_train, y_train,
                 X_test, y_test,
                 X_val, y_val,
                 target_col,dataset_path,
                 random_state: int = 42,
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.dataset_path=dataset_path
        self.target_col=target_col

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
            "SVM": {"C": [0.1,1,10], "kernel": ["linear"]},
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

    
    def train_models(self, cv: int = 5, verbose: bool = True):
      
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

    def tune_models(self,
                             inner_cv: int = 3,
                             search_type: str = "grid",
                             random_state: int = None,
                             n_iter: int = 20,
                             n_jobs: int = -1,
                             scoring: str = "accuracy",
                             verbose: bool = True):
   
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
        meta_features_extract_class(self.dataset_path,self.target_col,self.best_model_name)

        if verbose:
            print(f"\n[SELECT] Best model by validation accuracy: {self.best_model_name} (val_acc={self.best_val_acc:.4f})")
        return {
            "per_model_results": self.results,
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "best_val_accuracy": self.best_val_acc
        }




class Regression_Training:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val,dataset_path,target_col):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.dataset_path=dataset_path
        self.target_col=target_col
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
                "regressor__kernel": ["linear"],
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
        
        meta_features_extract_reg(self.dataset_path,self.target_col,best_row['Model'])
        
        return results_df
        




class ClusteringTrainer:

    def __init__(self, X, random_state: int = 42):
      
        self.X = X if not isinstance(X, pd.DataFrame) else X.values
        self.random_state = random_state

        # Candidate algorithms (factory functions returning new estimators)
        self.algos = {
            "KMeans": lambda **p: KMeans(random_state=self.random_state, **p),
            "DBSCAN": lambda **p: DBSCAN(**p),
            "Agglomerative": lambda **p: AgglomerativeClustering(**p)
        }

        # Default parameter grids (small, extend as needed)
        self.param_grids = {
            "KMeans": {"n_clusters": [2, 3, 4, 5, 8, 10]},
            "DBSCAN": {"eps": [0.3, 0.5, 0.8, 1.0], "min_samples": [3, 5, 8]},
            "Agglomerative": {"n_clusters": [2, 3, 4, 5, 8], "linkage": ["ward", "complete", "average"]}
        }

        # results containers
        self.baselines: Dict[str, Any] = {}
        self.tuned: Dict[str, Any] = {}
        self.results: pd.DataFrame = pd.DataFrame()
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.best_score: float = -np.inf
        self.best_metric: str = "silhouette"  # default selection metric


    # helpers: scoring

    def _score_labels(self, X, labels):
        """Compute internal clustering metrics. Labels of -1 (noise) are accepted."""
        # Need at least 2 clusters for silhouette and CH; DB requires >=1 cluster.
        unique_labels = set(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        scores = {"n_clusters": n_clusters}

        if n_clusters >= 2:
            try:
                scores["silhouette"] = float(silhouette_score(X, labels))
            except Exception:
                scores["silhouette"] = float("nan")
            try:
                scores["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except Exception:
                scores["calinski_harabasz"] = float("nan")
        else:
            scores["silhouette"] = float("nan")
            scores["calinski_harabasz"] = float("nan")

        # Davies-Bouldin exists for n_clusters >= 2 as well, smaller is better
        if n_clusters >= 2:
            try:
                scores["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            except Exception:
                scores["davies_bouldin"] = float("nan")
        else:
            scores["davies_bouldin"] = float("nan")

        return scores

 
    # baseline fitting

    def fit_baselines(self, verbose: bool = True):
        """
        Fit baseline versions of each algorithm (first grid value or sensible defaults)
        and compute cluster metrics.
        """
        rows = []
        for name, factory in self.algos.items():
            # pick a default param set: first item in param grid or empty
            grid = self.param_grids.get(name, {})
            default_params = {k: v[0] for k, v in grid.items()} if grid else {}
            model = factory(**default_params)
            # KMeans/Agglomerative need n_clusters; ensure sensible default if not present
            try:
                labels = model.fit_predict(self.X)
            except Exception:
                # Some algorithms (e.g. DBSCAN) may not have fit_predict method for edge cases
                model.fit(self.X)
                labels = model.labels_ if hasattr(model, "labels_") else model.predict(self.X)
            scores = self._score_labels(self.X, labels)
            rows.append({"algo": name, "params": default_params, **scores})
            self.baselines[name] = {"model": clone(model), "labels": labels, "scores": scores}
            if verbose:
                print(f"[BASELINE] {name} params={default_params} -> n_clusters={scores['n_clusters']}, silhouette={scores['silhouette']}")
        self.results = pd.DataFrame(rows)
        return self.results.sort_values(by="silhouette", ascending=False).reset_index(drop=True)


    def search(self,
               algo_name: str,
               search_type: str = "grid",
               n_iter: int = 20,
               score_metric: str = "silhouette",
               random_state: Optional[int] = None,
               verbose: bool = True):
      
        if random_state is None:
            random_state = self.random_state

        if algo_name not in self.algos:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        grid = self.param_grids.get(algo_name, {})
        if not grid:
            raise ValueError(f"No parameter grid defined for {algo_name}")

        # build list of candidate param dicts
        keys = list(grid.keys())
        all_candidates = []
        if search_type == "grid":
            # cartesian product (simple)
            import itertools
            for vals in itertools.product(*(grid[k] for k in keys)):
                all_candidates.append(dict(zip(keys, vals)))
        else:
            # random sampling without replacement
            candidates_set = set()
            attempts = 0
            max_attempts = max(n_iter * 10, 1000)
            while len(all_candidates) < n_iter and attempts < max_attempts:
                cand = {}
                for k in keys:
                    cand[k] = random.choice(grid[k])
                tup = tuple(sorted(cand.items()))
                if tup not in candidates_set:
                    candidates_set.add(tup)
                    all_candidates.append(cand)
                attempts += 1

        rows = []
        best_score = -np.inf if score_metric != "davies_bouldin" else np.inf
        best_model = None
        best_labels = None
        best_params = None

        for params in all_candidates:
            # instantiate model
            model = self.algos[algo_name](**params)
            try:
                labels = model.fit_predict(self.X)
            except Exception:
                model.fit(self.X)
                labels = getattr(model, "labels_", None)
                if labels is None and hasattr(model, "predict"):
                    labels = model.predict(self.X)
                if labels is None:
                    # skip if we can't get labels
                    continue

            scores = self._score_labels(self.X, labels)
            # choose metric comparison
            metric_val = scores.get(score_metric)
            if score_metric == "davies_bouldin":
                is_better = metric_val < best_score
            else:
                is_better = metric_val > best_score

            if is_better or best_model is None:
                best_score = metric_val
                best_model = clone(model)
                best_labels = labels
                best_params = params

            rows.append({"algo": algo_name, "params": params, **scores})

            if verbose:
                print(f"[TRY] {algo_name} params={params} -> n_clusters={scores['n_clusters']}, silhouette={scores['silhouette']}, DB={scores['davies_bouldin']:.4f}")

        df_results = pd.DataFrame(rows).sort_values(by="silhouette", ascending=False).reset_index(drop=True)
        # store best
        self.tuned[algo_name] = {"model": best_model, "params": best_params, "labels": best_labels, "score": best_score, "metric": score_metric}
        # also append to global results
        self.results = pd.concat([self.results, df_results], ignore_index=True, sort=False).reset_index(drop=True)
        if verbose:
            print(f"[BEST] {algo_name} best_params={best_params} best_{score_metric}={best_score}")
        return df_results

   
    # select best across algorithms
   
    def select_best(self, metric: str = "silhouette"):
        """
        Select the best tuned model (or baseline if not tuned) based on metric.
        metric: 'silhouette', 'davies_bouldin' (lower is better), or 'calinski_harabasz'
        """
        best = None
        best_val = -np.inf if metric != "davies_bouldin" else np.inf
        best_name = None
        best_obj = None

        # check tuned first, then baselines
        candidates = []
        for name in set(list(self.tuned.keys()) + list(self.baselines.keys())):
            if name in self.tuned and self.tuned[name]["model"] is not None:
                entry = self.tuned[name]
                score = entry["score"]
            else:
                entry = self.baselines.get(name)
                score = entry["scores"].get(metric) if entry is not None else float("nan")

            if score is None or (isinstance(score, float) and np.isnan(score)):
                continue

            if metric == "davies_bouldin":
                better = score < best_val
            else:
                better = score > best_val

            if better or best is None:
                best = entry
                best_val = score
                best_name = name

        self.best_model_name = best_name
        self.best_model = best["model"] if best is not None else None
        self.best_score = best_val
        self.best_metric = metric
        print(f"[SELECT] Best algorithm: {self.best_model_name} (metric={metric}, value={best_val})")
        return {"best_name": self.best_model_name, "best_model": self.best_model, "best_score": self.best_score}

 

    def get_labels(self, model_obj, X=None):
       
        if X is None:
            X = self.X
        try:
            return model_obj.fit_predict(X)
        except Exception:
            try:
                model_obj.fit(X)
                return getattr(model_obj, "labels_", None)
            except Exception:
                raise RuntimeError("Model cannot produce labels on given data.")

    



if __name__ == "__main__":
    # dataset_path = "datasets/regression/synthetic_car_prices.csv"
    # preprocessor = Preproccessor(dataset_path, "Price")
    # X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
    #     preprocessor.run_preprocessing()
    # )
    # trainer = Regression_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path,"Price")
    # trainer.train_model()


    dataset_path = "datasets/classification/synthetic.csv"
    preprocessor = Preproccessor(dataset_path, "target")
    X_train, y_train, X_test, y_test, X_val, y_val, task_type = (
        preprocessor.run_preprocessing()
    )
    trainer = Classification_Training(X_train, y_train, X_test, y_test, X_val, y_val,dataset_path=dataset_path,target_col='target')
    trainer.train_models()
  
    trainer.tune_models()
  
  
