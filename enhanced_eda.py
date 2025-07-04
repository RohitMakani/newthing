import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import chi2, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, Any, Tuple, List
import os
import warnings

warnings.filterwarnings('ignore')


class AutoEDAPipeline:
    """Enhanced Automated EDA Pipeline with advanced visualizations"""

    def __init__(self):
        self.charts_dir = "static/charts"
        os.makedirs(self.charts_dir, exist_ok=True)

    async def run_analysis(self, df: pd.DataFrame, task_type: str, target_col: str) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        """Run comprehensive EDA analysis"""
        report = {
            "original_shape": df.shape,
            "task_type": task_type,
            "target_column": target_col
        }

        # Step 1: Data Quality Assessment
        quality_report = self._assess_data_quality(df)
        report["data_quality"] = quality_report

        # Step 2: Clean and preprocess
        cleaned_df = self._clean_data(df)
        report["cleaned_shape"] = cleaned_df.shape

        # Step 3: Feature engineering
        engineered_df = self._engineer_features(cleaned_df, target_col, task_type)

        # Step 4: Generate comprehensive visualizations
        visualizations = await self._generate_visualizations(engineered_df, target_col, task_type)
        report["visualizations"] = visualizations

        # Step 5: Statistical analysis
        stats = self._statistical_analysis(engineered_df, target_col, task_type)
        report["statistics"] = stats

        # Step 6: Feature selection and importance
        feature_importance = self._analyze_feature_importance(engineered_df, target_col, task_type)
        report["feature_importance"] = feature_importance

        return engineered_df, report

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_report = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "zero_values": (df == 0).sum().to_dict(),
            "negative_values": (df.select_dtypes(include=[np.number]) < 0).sum().to_dict()
        }

        # Identify potential issues
        issues = []
        for col in df.columns:
            if quality_report["missing_percentage"][col] > 50:
                issues.append(f"High missing values in {col}: {quality_report['missing_percentage'][col]:.1f}%")

            if df[col].dtype == 'object' and df[col].nunique() == len(df):
                issues.append(f"Potential ID column: {col}")

        quality_report["potential_issues"] = issues
        return quality_report

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning"""
        cleaned_df = df.copy()

        # Clean column names
        cleaned_df.columns = cleaned_df.columns.str.strip().str.replace(' ', '_')

        # Remove ID-like columns
        id_cols = [col for col in cleaned_df.columns
                   if any(keyword in col.lower() for keyword in ['id', 'index', 'key'])]
        cleaned_df = cleaned_df.drop(columns=id_cols, errors='ignore')

        # Remove high-null columns (>60% missing)
        high_null_cols = cleaned_df.columns[cleaned_df.isnull().mean() > 0.6]
        cleaned_df = cleaned_df.drop(columns=high_null_cols)

        # Convert string numbers to numeric
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
            if numeric_series.notna().sum() > len(cleaned_df) * 0.7:
                cleaned_df[col] = numeric_series

        return cleaned_df

    def _engineer_features(self, df: pd.DataFrame, target_col: str, task_type: str) -> pd.DataFrame:
        """Advanced feature engineering"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        engineered_df = df.copy()

        # Separate features and target
        y = engineered_df[target_col]
        X = engineered_df.drop(columns=[target_col])

        # Handle missing values using KNN imputation
        X_imputed = self._knn_impute(X)

        # Encode categorical variables
        X_encoded = self._encode_categorical(X_imputed)

        # Feature scaling for numerical variables
        X_scaled = self._scale_features(X_encoded)

        # Encode target if classification
        if task_type == "classification" and y.dtype == 'object':
            y_encoded = LabelEncoder().fit_transform(y)
            y = pd.Series(y_encoded, index=y.index, name=target_col)

        # Outlier removal
        X_clean, y_clean = self._remove_outliers(X_scaled, y)

        # Combine back
        final_df = pd.concat([X_clean, y_clean], axis=1)

        return final_df.dropna()

    def _knn_impute(self, df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """KNN imputation for missing values"""
        # Encode categorical for imputation
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        df_temp = df.copy()

        encoders = {}
        for col in cat_cols:
            encoder = LabelEncoder()
            df_temp[col] = encoder.fit_transform(df_temp[col].astype(str))
            encoders[col] = encoder

        # Impute
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_temp),
            columns=df_temp.columns,
            index=df_temp.index
        )

        # Decode categorical columns
        for col in cat_cols:
            df_imputed[col] = df_imputed[col].round().astype(int)
            df_imputed[col] = encoders[col].inverse_transform(df_imputed[col])

        return df_imputed

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced categorical encoding"""
        encoded_df = df.copy()

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                encoded_df = pd.concat([encoded_df.drop(col, axis=1), dummies], axis=1)
            else:
                # Label encoding for high cardinality
                encoded_df[col] = LabelEncoder().fit_transform(df[col])

        return encoded_df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature scaling"""
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    def _remove_outliers(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """IQR-based outlier removal"""
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create mask for outliers
        outlier_mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)

        return X[outlier_mask], y[outlier_mask]

    async def _generate_visualizations(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, str]:
        """Generate comprehensive visualizations"""
        visualizations = {}

        # Target distribution
        target_dist_path = await self._plot_target_distribution(df[target_col], task_type)
        visualizations["target_distribution"] = target_dist_path

        # Correlation heatmap
        corr_path = await self._plot_correlation_heatmap(df)
        visualizations["correlation_heatmap"] = corr_path

        # Feature distributions
        feat_dist_path = await self._plot_feature_distributions(df, target_col)
        visualizations["feature_distributions"] = feat_dist_path

        # Pairplot for key features
        pairplot_path = await self._plot_pairplot(df, target_col, task_type)
        visualizations["pairplot"] = pairplot_path

        # Box plots for outlier detection
        boxplot_path = await self._plot_outlier_detection(df, target_col)
        visualizations["outlier_detection"] = boxplot_path

        return visualizations

    async def _plot_target_distribution(self, target: pd.Series, task_type: str) -> str:
        """Plot target variable distribution"""
        plt.figure(figsize=(10, 6))

        if task_type == "classification":
            target.value_counts().plot(kind='bar')
            plt.title('Target Variable Distribution (Classification)')
            plt.ylabel('Count')
        else:
            plt.hist(target, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Target Variable Distribution (Regression)')
            plt.ylabel('Frequency')

        plt.xlabel('Target Value')
        plt.tight_layout()

        path = f"{self.charts_dir}/target_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    async def _plot_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))

        corr_matrix = df.select_dtypes(include=[np.number]).corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )

        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()

        path = f"{self.charts_dir}/correlation_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    async def _plot_feature_distributions(self, df: pd.DataFrame, target_col: str) -> str:
        """Plot feature distributions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        n_features = min(len(numeric_cols), 9)  # Limit to 9 features
        cols = 3
        rows = (n_features + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        for i, col in enumerate(numeric_cols[:n_features]):
            plt.subplot(rows, cols, i + 1)
            plt.hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

        plt.tight_layout()

        path = f"{self.charts_dir}/feature_distributions.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    async def _plot_pairplot(self, df: pd.DataFrame, target_col: str, task_type: str) -> str:
        """Plot pairplot for key features"""
        # Select top 5 most correlated features with target
        numeric_df = df.select_dtypes(include=[np.number])

        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations.head(6).index.tolist()  # Include target

            subset_df = df[top_features]

            plt.figure(figsize=(12, 10))

            if task_type == "classification":
                sns.pairplot(subset_df, hue=target_col, diag_kind='hist')
            else:
                sns.pairplot(subset_df, diag_kind='hist')

            plt.suptitle('Pairplot of Top Correlated Features', y=1.02)

            path = f"{self.charts_dir}/pairplot.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            return path

        return ""

    async def _plot_outlier_detection(self, df: pd.DataFrame, target_col: str) -> str:
        """Plot box plots for outlier detection"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        n_features = min(len(numeric_cols), 6)
        cols = 3
        rows = (n_features + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        for i, col in enumerate(numeric_cols[:n_features]):
            plt.subplot(rows, cols, i + 1)
            plt.boxplot(df[col].dropna())
            plt.title(f'Box Plot - {col}')
            plt.ylabel(col)

        plt.tight_layout()

        path = f"{self.charts_dir}/outlier_detection.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path

    def _statistical_analysis(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, Any]:
        """Comprehensive statistical analysis"""
        stats = {}

        # Basic statistics
        stats["descriptive"] = df.describe().to_dict()

        # Target variable analysis
        if target_col in df.columns:
            target = df[target_col]

            if task_type == "classification":
                stats["target"] = {
                    "unique_classes": int(target.nunique()),
                    "class_distribution": target.value_counts().to_dict(),
                    "class_balance": target.value_counts(normalize=True).to_dict()
                }
            else:
                stats["target"] = {
                    "mean": float(target.mean()),
                    "median": float(target.median()),
                    "std": float(target.std()),
                    "skewness": float(target.skew()),
                    "kurtosis": float(target.kurtosis())
                }

        return stats

    def _analyze_feature_importance(self, df: pd.DataFrame, target_col: str, task_type: str) -> Dict[str, Any]:
        """Analyze feature importance using statistical tests"""
        if target_col not in df.columns:
            return {}

        X = df.drop(columns=[target_col])
        y = df[target_col]

        importance_scores = {}

        try:
            if task_type == "classification":
                # Chi-square test for categorical target
                scores, p_values = chi2(X, y)
            else:
                # F-test for regression
                scores, p_values = f_classif(X, y)

            for i, col in enumerate(X.columns):
                importance_scores[col] = {
                    "score": float(scores[i]),
                    "p_value": float(p_values[i]),
                    "significant": p_values[i] < 0.05
                }

        except Exception as e:
            print(f"Feature importance analysis failed: {e}")

        return importance_scores
