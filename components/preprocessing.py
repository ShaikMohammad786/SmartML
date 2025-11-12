# preprocessing order:
# 1 Remove duplicates
# 2 Split into train / val / test
# 3️ Impute missing values
# 4️ Remove outliers
# 5️ Encode categorical features
# 6️ Scale data (fit on train, transform val/test)
# 7️ Remove highly correlated features
# 8 Apply PCA
# 9 Apply SMOTE (only on training set)


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os,json
from datetime import datetime


class Preproccessor:
    def __init__(self, dataframe, target_col):
        self.df = pd.read_csv(dataframe,header=0)
        self.target_col = target_col
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.pca = None
        self.scaler = None
        self.task_type=None

    def check_task(self):
        if not self.target_col:
            self.task_type="clustering"
        elif self.df[self.target_col].nunique()<=20:
            self.task_type="classification"
        else:
            self.task_type="regression"
        
        return self
    
    
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)
    
    def splitting(self):
        df = self.df
        X = df.drop(columns=[self.target_col], axis=1)
        y = df[self.target_col]
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
    def imputing_null_values(self, discrete_threshold=20, n_neighbors=3):
        """
        Imputes missing values robustly using:
        - Mode for categorical columns
        - Mean for continuous numeric columns
        - KNN for discrete numeric columns
        - Fallback imputation for any remaining NaNs
        """

        print("\n🔹 Starting missing value imputation...")

        X_train = self.X_train.copy()
        X_test = self.X_test.copy() if self.X_test is not None else None
        X_val = self.X_val.copy() if self.X_val is not None else None

        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        # 1️⃣ Categorical imputation (mode)
        for col in cat_cols:
            mode_vals = X_train[col].mode()
            if not mode_vals.empty:
                mode_val = mode_vals.iloc[0]
                X_train[col] = X_train[col].fillna(mode_val)
                if X_test is not None:
                    X_test[col] = X_test[col].fillna(mode_val)
                if X_val is not None:
                    X_val[col] = X_val[col].fillna(mode_val)

        # 2️⃣ Separate numeric columns
        discrete_cols = [c for c in num_cols if X_train[c].nunique(dropna=True) <= discrete_threshold]
        continuous_cols = [c for c in num_cols if c not in discrete_cols]

        # 3️⃣ Continuous numeric imputation (mean)
        for col in continuous_cols:
            mean_val = X_train[col].mean()
            X_train[col] = X_train[col].fillna(mean_val)
            if X_test is not None:
                X_test[col] = X_test[col].fillna(mean_val)
            if X_val is not None:
                X_val[col] = X_val[col].fillna(mean_val)

        # 4️⃣ Discrete numeric imputation (KNN)
        if len(discrete_cols) > 0:
            X_train_discrete = X_train[discrete_cols].copy()
            if X_train_discrete.isna().all().all() or X_train_discrete.shape[1] == 0:
                print("⚠️ No valid discrete numeric columns to impute (all NaN or empty). Skipping KNN.")
            else:
                scaler = StandardScaler()
                imputer = KNNImputer(n_neighbors=n_neighbors)
                try:
                    knn_train_scaled = scaler.fit_transform(X_train_discrete)
                    imputed_train_scaled = imputer.fit_transform(knn_train_scaled)
                    imputed_train_unscaled = np.round(scaler.inverse_transform(imputed_train_scaled))

                    X_train[discrete_cols] = pd.DataFrame(imputed_train_unscaled, columns=discrete_cols, index=X_train.index)

                    def _impute_other(df):
                        if df is None or df.empty:
                            return df
                        df_discrete = df[discrete_cols].copy()
                        if df_discrete.shape[1] == 0:
                            return df
                        scaled = scaler.transform(df_discrete)
                        imputed_scaled = imputer.transform(scaled)
                        imputed_unscaled = np.round(scaler.inverse_transform(imputed_scaled))
                        df[discrete_cols] = pd.DataFrame(imputed_unscaled, columns=discrete_cols, index=df.index)
                        return df

                    X_test = _impute_other(X_test)
                    X_val = _impute_other(X_val)

                except ValueError as e:
                    print(f"⚠️ Skipping KNN imputation due to: {e}")
        else:
            print("ℹ️ No discrete numeric columns detected — skipping KNN imputation.")

        # 5️⃣ Final Fallback — fill any remaining NaNs (numeric → mean, categorical → mode)
        print("🔍 Checking for any remaining missing values...")
        all_cols = X_train.columns
        fallback_counts = {}
        for df_name, df in {"train": X_train, "val": X_val, "test": X_test}.items():
            if df is None:
                continue
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                fallback_counts[df_name] = nan_cols
                for col in nan_cols:
                    if col in num_cols:
                        fill_val = X_train[col].mean()
                    else:
                        mode_vals = X_train[col].mode()
                        fill_val = mode_vals.iloc[0] if not mode_vals.empty else 0
                    df[col] = df[col].fillna(fill_val)

        if fallback_counts:
            print(f"✅ Fallback imputation applied to remaining NaNs: {json.dumps(fallback_counts, indent=4)}")
        else:
            print("✅ No remaining missing values detected after main imputations.")

        # 6️⃣ Save back to instance
        self.X_train = X_train.reset_index(drop=True)
        if X_test is not None:
            self.X_test = X_test.reset_index(drop=True)
        if X_val is not None:
            self.X_val = X_val.reset_index(drop=True)

        print("🎯 Imputation complete. All datasets are now free of missing values.\n")
        return self


    def remove_outliers_iqr(self, factor=1.5, min_violations=3, min_frac=0.10, nan_threshold_frac=0.10):
    
        if self.X_train is None:
            raise ValueError("Call splitting() before remove_outliers_iqr().")

        X_train = self.X_train.copy()
        y_train = self.y_train.copy() if self.y_train is not None else None
        X_val = self.X_val.copy() if self.X_val is not None else None
        X_test = self.X_test.copy() if self.X_test is not None else None

        X_train = X_train.reset_index(drop=True)
        if y_train is not None:
            y_train = y_train.reset_index(drop=True)

        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            print("No numeric columns to check for outliers.")
            self.X_train, self.y_train = X_train, y_train
            return self

        # 🧠 Intelligent NaN handling
        nan_stats = X_train[num_cols].isna().mean()
        to_drop = nan_stats[nan_stats >= nan_threshold_frac].index.tolist()
        to_fill = nan_stats[(nan_stats > 0) & (nan_stats < nan_threshold_frac)].index.tolist()

        if to_drop:
            print(f"⚠️ Dropping {len(to_drop)} numeric columns with ≥{nan_threshold_frac*100:.0f}% NaNs: {to_drop}")
            X_train.drop(columns=to_drop, inplace=True)
            if X_val is not None:
                X_val.drop(columns=[c for c in to_drop if c in X_val.columns], inplace=True)
            if X_test is not None:
                X_test.drop(columns=[c for c in to_drop if c in X_test.columns], inplace=True)

            num_cols = [c for c in num_cols if c not in to_drop]

        if to_fill:
            print(f"🩹 Filling {len(to_fill)} columns with mean (NaNs < {nan_threshold_frac*100:.0f}%): {to_fill}")
            for col in to_fill:
                mean_val = X_train[col].mean()
                X_train[col].fillna(mean_val, inplace=True)
                if X_val is not None and col in X_val.columns:
                    X_val[col].fillna(mean_val, inplace=True)
                if X_test is not None and col in X_test.columns:
                    X_test[col].fillna(mean_val, inplace=True)

        # --- IQR Outlier Removal ---
        Q1 = X_train[num_cols].quantile(0.25)
        Q3 = X_train[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        is_low = X_train[num_cols].lt(lower)
        is_high = X_train[num_cols].gt(upper)
        violations = (is_low | is_high)
        viol_count = violations.sum(axis=1)

        n_numeric = len(num_cols)
        frac_threshold = max(1, int(np.ceil(min_frac * n_numeric)))
        threshold = min_violations
        remove_mask = (viol_count >= threshold) | (viol_count >= frac_threshold)
        n_removed = int(remove_mask.sum())

        col_viol_counts = violations.sum().sort_values(ascending=False)
        print(f"IQR factor={factor}. Numeric cols: {n_numeric}.")
        print(f"Per-row removal thresholds -> min_violations={min_violations}, min_frac={min_frac} (=> {frac_threshold} cols).")
        print("Top offending numeric columns (violation counts):")
        print(col_viol_counts.head(10))

        if n_removed > 0:
            X_train_filtered = X_train.loc[~remove_mask].reset_index(drop=True)
            if y_train is not None:
                y_train_filtered = y_train.loc[~remove_mask.values].reset_index(drop=True)
            else:
                y_train_filtered = None
            self.X_train = X_train_filtered
            self.y_train = y_train_filtered
            print(f"✅ Removed {n_removed} training rows (out of {len(X_train)}).")
        else:
            print("✅ No training rows removed (no outliers met criteria).")

        self.X_val = X_val
        self.X_test = X_test

        return self

    def universal_encoder(self, cardinality_threshold=10):
        X_train, X_val, X_test = (
            self.X_train.copy(),
            self.X_val.copy(),
            self.X_test.copy(),
        )
        y_train = self.y_train if self.target_col is not None else None
        encoders = {}

        for col in X_train.columns:
            if X_train[col].dtype == "object" or str(X_train[col].dtype) == "category":
                unique_vals = X_train[col].nunique()

                # 1️ Low-cardinality → One-Hot Encoding
                if unique_vals <= cardinality_threshold:
                    dummies_train = pd.get_dummies(
                        X_train[col], prefix=col, drop_first=True
                    )
                    dummies_val = pd.get_dummies(
                        X_val[col], prefix=col, drop_first=True
                    )
                    dummies_test = pd.get_dummies(
                        X_test[col], prefix=col, drop_first=True
                    )

                    all_cols = dummies_train.columns.union(dummies_val.columns).union(
                        dummies_test.columns
                    )
                    dummies_train = dummies_train.reindex(
                        columns=all_cols, fill_value=0
                    )
                    dummies_val = dummies_val.reindex(columns=all_cols, fill_value=0)
                    dummies_test = dummies_test.reindex(columns=all_cols, fill_value=0)

                    X_train = pd.concat(
                        [X_train.drop(columns=[col]), dummies_train], axis=1
                    )
                    X_val = pd.concat([X_val.drop(columns=[col]), dummies_val], axis=1)
                    X_test = pd.concat(
                        [X_test.drop(columns=[col]), dummies_test], axis=1
                    )
                    encoders[col] = {"type": "onehot", "columns": list(all_cols)}

                # 2️ High-cardinality + target → Target Encoding
                elif y_train is not None:
                    y = y_train.copy()
                    if y.dtype == "object" or str(y.dtype) == "category":
                        y = pd.Categorical(y).codes

                    df_temp = pd.DataFrame({col: X_train[col], self.target_col: y})
                    means = df_temp.groupby(col)[self.target_col].mean()
                    X_train[col] = X_train[col].map(means).fillna(means.mean())
                    X_val[col] = X_val[col].map(means).fillna(means.mean())
                    X_test[col] = X_test[col].map(means).fillna(means.mean())
                    encoders[col] = {"type": "target", "mapping": means.to_dict()}

                # 3️ High-cardinality + no target → Frequency Encoding
                else:
                    freqs = X_train[col].value_counts(normalize=True)
                    X_train[col] = X_train[col].map(freqs).fillna(0)
                    X_val[col] = X_val[col].map(freqs).fillna(0)
                    X_test[col] = X_test[col].map(freqs).fillna(0)
                    encoders[col] = {"type": "frequency", "mapping": freqs.to_dict()}

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.encoders = encoders
        print(" Encoding complete.")
        return self

    def scaling(self, discrete_threshold=20):

        X_train, X_val, X_test = (
            self.X_train.copy(),
            self.X_val.copy(),
            self.X_test.copy(),
        )

        # Detect numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        # Detect low-cardinality numeric columns (categorical-like)
        discrete_like_cols = [
            col
            for col in numeric_cols
            if X_train[col].nunique(dropna=True) <= discrete_threshold
        ]

        # Columns to scale = numeric columns minus discrete/categorical-like
        scale_cols = [col for col in numeric_cols if col not in discrete_like_cols]

        if len(scale_cols) > 0:
            scaler = StandardScaler()
            X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
            X_val[scale_cols] = scaler.transform(X_val[scale_cols])
            X_test[scale_cols] = scaler.transform(X_test[scale_cols])
            self.scaler = scaler
            print(f" Scaled {len(scale_cols)} continuous numeric columns.")
        else:
            print(" No continuous numeric columns found for scaling.")

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        return self

    def remove_high_correlation(self, threshold=0.95):
        X_train, X_val, X_test = (
            self.X_train.copy(),
            self.X_val.copy(),
            self.X_test.copy(),
        )
        num = X_train.select_dtypes(include=[np.number])

        if num.shape[1] == 0:
            print("No numeric columns to check for correlation.")
            return self

        corr = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]

        if to_drop:
            X_train.drop(columns=to_drop, inplace=True)
            X_val.drop(columns=[c for c in to_drop if c in X_val.columns], inplace=True)
            X_test.drop(
                columns=[c for c in to_drop if c in X_test.columns], inplace=True
            )
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            print(f"Dropped {len(to_drop)} highly correlated columns: {to_drop}")
        else:
            print("No highly correlated columns found.")
        return self


    def apply_pca(self, variance_threshold=0.95):
        print("\nNaN check before PCA:")
        print(self.X_train.isna().sum()[self.X_train.isna().sum() > 0])
        if self.X_train.shape[1]<50:
            return self
        if self.X_train is None:
            raise ValueError("Call splitting() and scaling() before apply_pca().")

        pca = PCA(n_components=variance_threshold, svd_solver='full', random_state=42)
        X_train_pca = pca.fit_transform(self.X_train)

        col_names = [f"PC{i+1}" for i in range(pca.n_components_)]

        self.X_train = pd.DataFrame(X_train_pca, columns=col_names, index=self.X_train.index)
        if self.X_val is not None:
            self.X_val = pd.DataFrame(pca.transform(self.X_val), columns=col_names, index=self.X_val.index)
        if self.X_test is not None:
            self.X_test = pd.DataFrame(pca.transform(self.X_test), columns=col_names, index=self.X_test.index)

       
        self.pca = pca

        explained = np.sum(pca.explained_variance_ratio_) * 100
        print(f"PCA applied dynamically. Retained {pca.n_components_} components explaining {explained:.2f}% variance.")

        return self


    def data_balancing(self, random_state, sampling_strategy="auto", k_neighbors=5):

        if(self.y_train.nunique()>=20):
            return self

        if self.X_train is None or self.y_train is None:
            raise ValueError("splittin is not done")

        X_train_was_df = isinstance(self.X_train, pd.DataFrame)

        if not X_train_was_df:
            Xtrain = pd.DataFrame(self.X_train)
        else:
            Xtrain = self.X_train.copy()

        ytrain = pd.Series(self.y_train).copy()

        sm = SMOTE(
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
        )

        Xres, yres = sm.fit_resample(Xtrain, ytrain)

        if X_train_was_df:
            Xres = pd.DataFrame(Xres, columns=Xtrain.columns)

        self.X_train = Xres
        self.y_train = yres

        print("data balancing was successful.new rows are added and data is balanced")


    def save_dataset(self, dataset_name=None, base_dir="processed_datasets"):
        """
        Save preprocessed train/val/test splits into a unique timestamped folder.
        Automatically avoids name collisions.
        """
        os.makedirs(base_dir, exist_ok=True)

        # Derive dataset name automatically if not given
        if dataset_name is None:
            # Try from original file path if available
            if hasattr(self, "dataframe"):
                base_name = os.path.splitext(os.path.basename(self.df_path))[0]
            else:
                base_name = "dataset"

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dataset_name = f"{base_name}_{timestamp}"

        dataset_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Combine X and y for saving
        def combine_xy(X, y):
            df = X.copy()
            if y is not None:
                df[self.target_col] = y.values
            return df

        train_df = combine_xy(self.X_train, self.y_train)
        val_df = combine_xy(self.X_val, self.y_val)
        test_df = combine_xy(self.X_test, self.y_test)

        # Save CSVs
        train_path = os.path.join(dataset_dir, "train.csv")
        val_path = os.path.join(dataset_dir, "val.csv")
        test_path = os.path.join(dataset_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Optional: metadata JSON
        meta_info = {
            "dataset_name": dataset_name,
            "target_col": self.target_col,
            "created_at": datetime.now().isoformat(),
            "rows": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df)
            },
            "features": list(train_df.columns),
        }

        with open(os.path.join(dataset_dir, "info.json"), "w") as f:
            json.dump(meta_info, f, indent=4)

        print(f" Dataset saved successfully to '{dataset_dir}'")
        print(f" Files:\n  - train.csv\n  - val.csv\n  - test.csv\n  - info.json")

        return self

    def run_preprocessing(self):
        self.remove_duplicates()
        self.check_task()
        self.splitting()
        self.imputing_null_values()
        self.remove_outliers_iqr()
        self.universal_encoder()
        self.scaling()
        self.remove_high_correlation()
        self.apply_pca()
        self.data_balancing(random_state=42)
        self.save_dataset()
        
        return self.X_train,self.y_train,self.X_test,self.y_test,self.X_val,self.y_val,self.task_type





if __name__ == "__main__":
   

    dataset_path = "../datasets/classification/phone_detection.csv"
    pp = Preproccessor(dataframe=dataset_path,target_col='price_range')
    pp.run_preprocessing()
    print("\n🚀 Preprocessing pipeline completed and saved successfully.")


