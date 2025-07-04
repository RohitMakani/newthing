import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class EnhancedMLPipeline:
    """Enhanced ML pipeline with advanced algorithms and hyperparameter tuning"""

    def __init__(self):
        self.charts_dir = "static/charts"
        self.models_dir = "outputs"
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Advanced model configurations
        self.classification_models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(random_state=42, eval_metric='logloss'),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.2],
                    "max_depth": [3, 6]
                }
            },
            "LightGBM": {
                "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.2],
                    "max_depth": [3, 6]
                }
            },
            "SVM": {
                "model": SVC(probability=True, random_state=42),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                }
            }
        }

        self.regression_models = {
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "Ridge Regression": {
                "model": Ridge(),
                "params": {
                    "alpha": [0.1, 1, 10]
                }
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None]
                }
            },
            "XGBoost": {
                "model": XGBRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.2],
                    "max_depth": [3, 6]
                }
            },
            "LightGBM": {
                "model": lgb.LGBMRegressor(random_state=42, verbose=-1),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.2],
                    "max_depth": [3, 6]
                }
            }
        }
        print("Yha ykk")
    async def train_and_evaluate(self, df: pd.DataFrame, task_type: str, target_col: str) -> Dict[str, Any]:
        """Enhanced training and evaluation pipeline"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
        )

        # Handle class imbalance for classification
        if task_type == "classification":
            X_train, y_train = self._handle_imbalance(X_train, y_train)

        # Train and evaluate models
        results = await self._train_models(X_train, X_test, y_train, y_test, task_type)

        # Create ensemble model
        ensemble_results = await self._create_ensemble(X_train, X_test, y_train, y_test, task_type, results)
        results.update(ensemble_results)

        # Generate comprehensive report
        report = self._generate_comprehensive_report(results, task_type, df.shape)

        return report

    def _handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE or other techniques"""
        class_counts = y_train.value_counts()
        min_class_count = class_counts.min()

        if len(class_counts) > 1 and class_counts.max() / min_class_count > 2:
            try:
                if min_class_count >= 6:
                    sampler = SMOTE(k_neighbors=5, random_state=42)
                elif min_class_count > 1:
                    sampler = SMOTE(k_neighbors=min_class_count - 1, random_state=42)
                else:
                    sampler = RandomOverSampler(random_state=42)

                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
            except Exception as e:
                print(f"Resampling failed: {e}, using original data")

        return X_train, y_train

    async def _train_models(self, X_train, X_test, y_train, y_test, task_type: str) -> Dict[str, Any]:
        """Train multiple models with hyperparameter tuning"""
        models_config = self.classification_models if task_type == "classification" else self.regression_models
        results = {}

        for name, config in models_config.items():
            try:
                print(f"Training {name}...")

                # Hyperparameter tuning if parameters are defined
                if config["params"]:
                    grid_search = GridSearchCV(
                        config["model"],
                        config["params"],
                        cv=3,
                        scoring='f1_macro' if task_type == "classification" else 'r2',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    best_model = config["model"]
                    best_model.fit(X_train, y_train)
                    best_params = {}

                # Make predictions
                y_pred = best_model.predict(X_test)

                # Calculate metrics
                if task_type == "classification":
                    metrics = self._calculate_classification_metrics(y_test, y_pred, best_model, X_test)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)

                # Generate plots
                plot_path = await self._generate_model_plots(y_test, y_pred, name, task_type)

                results[name] = {
                    "model": best_model,
                    "metrics": metrics,
                    "best_params": best_params,
                    "plot_path": plot_path
                }

            except Exception as e:
                print(f"Training {name} failed: {e}")
                results[name] = {"error": str(e)}

        return results

    def _calculate_classification_metrics(self, y_true, y_pred, model, X_test) -> Dict[str, float]:
        """Calculate comprehensive classification metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro')
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except:
                pass

        return metrics

    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        return {
            "r2_score": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

    async def _generate_model_plots(self, y_true, y_pred, model_name: str, task_type: str) -> str:
        """Generate visualization plots for model evaluation"""
        plt.figure(figsize=(12, 8))

        if task_type == "classification":
            # Confusion Matrix
            plt.subplot(2, 2, 1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Classification Report Heatmap
            plt.subplot(2, 2, 2)
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).iloc[:-1, :].T
            sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='RdYlBu')
            plt.title('Classification Report')

        else:
            # Actual vs Predicted
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.7)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted - {model_name}')

            # Residuals Plot
            plt.subplot(2, 2, 2)
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')

        plt.tight_layout()

        plot_path = f"{self.charts_dir}/{model_name.replace(' ', '_').lower()}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    async def _create_ensemble(self, X_train, X_test, y_train, y_test, task_type: str, results: Dict) -> Dict[str, Any]:
        """Create ensemble model from best performing models"""
        try:
            # Get top 3 models based on performance
            if task_type == "classification":
                valid_models = [(name, data) for name, data in results.items()
                                if "model" in data and "metrics" in data]
                valid_models.sort(key=lambda x: x[1]["metrics"]["f1_macro"], reverse=True)
            else:
                valid_models = [(name, data) for name, data in results.items()
                                if "model" in data and "metrics" in data]
                valid_models.sort(key=lambda x: x[1]["metrics"]["r2_score"], reverse=True)

            if len(valid_models) < 2:
                return {}

            # Create ensemble
            top_models = valid_models[:3]
            estimators = [(name, data["model"]) for name, data in top_models]

            if task_type == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=estimators)

            # Train ensemble
            ensemble.fit(X_train, y_train)
            y_pred_ensemble = ensemble.predict(X_test)

            # Calculate metrics
            if task_type == "classification":
                metrics = self._calculate_classification_metrics(y_test, y_pred_ensemble, ensemble, X_test)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred_ensemble)

            # Generate plots
            plot_path = await self._generate_model_plots(y_test, y_pred_ensemble, "Ensemble", task_type)

            return {
                "Ensemble": {
                    "model": ensemble,
                    "metrics": metrics,
                    "plot_path": plot_path,
                    "component_models": [name for name, _ in top_models]
                }
            }

        except Exception as e:
            print(f"Ensemble creation failed: {e}")
            return {}

    def _generate_comprehensive_report(self, results: Dict[str, Any], task_type: str, dataset_shape: Tuple) -> Dict[
        str, Any]:
        """Generate comprehensive ML report"""
        # Find best model
        if task_type == "classification":
            best_model_name = max(
                [name for name, data in results.items() if "metrics" in data],
                key=lambda x: results[x]["metrics"]["f1_macro"]
            )
            primary_metric = "f1_macro"
        else:
            best_model_name = max(
                [name for name, data in results.items() if "metrics" in data],
                key=lambda x: results[x]["metrics"]["r2_score"]
            )
            primary_metric = "r2_score"

        # Save best model
        best_model = results[best_model_name]["model"]
        model_path = f"{self.models_dir}/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # Create comparison table
        comparison_table = []
        for name, data in results.items():
            if "metrics" in data:
                row = {"Model": name}
                row.update(data["metrics"])
                if "plot_path" in data:
                    row["Visualization"] = data["plot_path"]
                comparison_table.append(row)

        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(len(best_model.feature_importances_))],
                best_model.feature_importances_
            ))

        return {
            "task_type": task_type,
            "dataset_shape": dataset_shape,
            "best_model": {
                "name": best_model_name,
                "metrics": results[best_model_name]["metrics"],
                "primary_score": results[best_model_name]["metrics"][primary_metric],
                "model_path": model_path
            },
            "comparison_table": comparison_table,
            "feature_importance": feature_importance,
            "total_models_trained": len([r for r in results.values() if "metrics" in r]),
            "training_summary": {
                "successful_models": len([r for r in results.values() if "metrics" in r]),
                "failed_models": len([r for r in results.values() if "error" in r])
            }
        }
