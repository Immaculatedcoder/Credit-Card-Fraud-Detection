import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detection for credit card fraud.
    
    How Isolation Forest Works:
    - Creates random decision trees by randomly selecting features and split values
    - Anomalies are easier to isolate (require fewer splits)
    - Normal points are harder to isolate (require more splits)
    - Anomaly score based on average path length across all trees
    """
    
    def __init__(self, data: pd.DataFrame, feature_columns: List[str]):
        """
        Initialize Isolation Forest detector.
        
        Args:
            data: DataFrame with transactions (must include 'Class' column for evaluation)
            feature_columns: List of feature column names to use for detection
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.X = None
        self.X_scaled = None
        self.y = data['Class'].values if 'Class' in data.columns else None
        
        # Model
        self.model = None
        self.scaler = StandardScaler()
        
        # Results
        self.anomaly_scores = None
        self.predictions = None
        self.risk_scores = None
        
        logger.info(f"Initialized IsolationForestDetector with {len(feature_columns)} features")
        logger.info(f"Dataset: {len(data)} transactions")
        if self.y is not None:
            logger.info(f"Fraud cases: {self.y.sum()} ({self.y.mean()*100:.2f}%)")
    
    def prepare_features(self):
        """Prepare and scale features for anomaly detection."""
        logger.info("Preparing features...")
        
        # Extract features
        self.X = self.data[self.feature_columns].values
        
        # Scale features (important for distance-based comparisons)
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        logger.info(f"Features prepared and scaled. Shape: {self.X_scaled.shape}")
    
    def train(
        self, 
        contamination: float = 'auto',
        n_estimators: int = 100,
        max_samples: int = 256,
        max_features: float = 1.0,
        random_state: int = 42
    ) -> Dict:
        """
        Train Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
                          - 'auto': Automatically determined
                          - float: Specific contamination rate (e.g., 0.001 for 0.1%)
            n_estimators: Number of isolation trees to build
            max_samples: Number of samples to draw from X to train each tree
                        - Lower values = faster training, might miss global patterns
                        - Higher values = slower, captures more global structure
            max_features: Number of features to draw to train each tree (1.0 = all features)
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with training results and evaluation metrics
        """
        logger.info("Training Isolation Forest...")
        
        if self.X_scaled is None:
            self.prepare_features()
        
        # If we know the fraud rate, use it as contamination estimate
        if contamination == 'auto' and self.y is not None:
            actual_fraud_rate = self.y.mean()
            contamination = max(actual_fraud_rate, 0.001)  # At least 0.1%
            logger.info(f"Using contamination rate based on actual fraud: {contamination:.4f} ({contamination*100:.2f}%)")
        elif contamination == 'auto':
            contamination = 0.1  # Default 10%
            logger.info(f"Using default contamination rate: {contamination}")
        
        # Initialize and train model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        logger.info(f"Training with {n_estimators} trees, max_samples={max_samples}...")
        self.model.fit(self.X_scaled)
        logger.info("Training complete!")
        
        # Get predictions (-1 for anomaly, 1 for normal)
        self.predictions = self.model.predict(self.X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        raw_scores = self.model.score_samples(self.X_scaled)
        
        # Convert to 0-1 scale where HIGHER = MORE ANOMALOUS (more intuitive)
        # Invert the scores first
        inverted_scores = -raw_scores
        
        # Normalize to 0-1 range
        self.anomaly_scores = (inverted_scores - inverted_scores.min()) / \
                             (inverted_scores.max() - inverted_scores.min())
        
        # Store in dataframe
        self.data['anomaly_score'] = self.anomaly_scores
        self.data['is_anomaly'] = (self.predictions == -1).astype(int)
        
        # Basic results
        n_anomalies = (self.predictions == -1).sum()
        anomaly_rate = (self.predictions == -1).mean()
        
        results = {
            'training_params': {
                'contamination': float(contamination),
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'max_features': max_features
            },
            'detection_summary': {
                'total_transactions': len(self.data),
                'anomalies_detected': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'anomaly_percentage': float(anomaly_rate * 100)
            },
            'score_statistics': {
                'min': float(self.anomaly_scores.min()),
                'max': float(self.anomaly_scores.max()),
                'mean': float(self.anomaly_scores.mean()),
                'median': float(np.median(self.anomaly_scores)),
                'std': float(self.anomaly_scores.std())
            }
        }
        
        # Evaluate if we have ground truth labels
        if self.y is not None:
            evaluation = self._evaluate_model()
            results['evaluation'] = evaluation
            
            logger.info(f"\nModel Performance:")
            logger.info(f"  ROC-AUC Score: {evaluation['metrics']['roc_auc']:.4f}")
            logger.info(f"  Precision: {evaluation['metrics']['precision']:.4f}")
            logger.info(f"  Recall: {evaluation['metrics']['recall']:.4f}")
            logger.info(f"  F1-Score: {evaluation['metrics']['f1_score']:.4f}")
            logger.info(f"  Frauds Detected: {evaluation['fraud_detection']['frauds_detected']}/{evaluation['fraud_detection']['total_frauds']}")
        
        logger.info(f"\nDetected {n_anomalies:,} anomalies ({anomaly_rate*100:.2f}% of transactions)")
        
        return results
    
    def _evaluate_model(self) -> Dict:
        """
        Evaluate model performance against true labels.
        
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if self.y is None:
            return {}
        
        # Binary predictions (1 = anomaly/fraud, 0 = normal)
        binary_preds = (self.predictions == -1).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y, binary_preds).ravel()
        
        # Classification report
        report = classification_report(self.y, binary_preds, output_dict=True, zero_division=0)
        
        # ROC-AUC using continuous scores
        roc_auc = roc_auc_score(self.y, self.anomaly_scores)
        
        # Average precision (better for imbalanced data than ROC-AUC)
        avg_precision = average_precision_score(self.y, self.anomaly_scores)
        
        # Find optimal threshold for best F1 score
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.anomaly_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
        best_f1 = f1_scores[best_f1_idx]
        
        return {
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'metrics': {
                'accuracy': float(report['accuracy']),
                'precision': float(report['1']['precision']) if '1' in report else 0.0,
                'recall': float(report['1']['recall']) if '1' in report else 0.0,
                'f1_score': float(report['1']['f1-score']) if '1' in report else 0.0,
                'roc_auc': float(roc_auc),
                'average_precision': float(avg_precision)
            },
            'fraud_detection': {
                'frauds_detected': int(tp),
                'frauds_missed': int(fn),
                'total_frauds': int(self.y.sum()),
                'detection_rate': float(tp / self.y.sum()) if self.y.sum() > 0 else 0,
                'false_alarms': int(fp),
                'false_alarm_rate': float(fp / (tn + fp)) if (tn + fp) > 0 else 0
            },
            'optimal_threshold': {
                'value': float(best_threshold),
                'f1_score': float(best_f1),
                'explanation': 'Threshold that maximizes F1 score'
            }
        }
    
    def create_risk_scores(self, risk_levels: int = 4) -> np.ndarray:
        """
        Create categorical risk scores from anomaly scores.
        
        Args:
            risk_levels: Number of risk categories (default: 4 = Low, Medium, High, Critical)
        
        Returns:
            Array of risk scores
        """
        if self.anomaly_scores is None:
            raise ValueError("No anomaly scores available. Train the model first.")
        
        logger.info(f"Creating {risk_levels}-level risk categories...")
        
        self.risk_scores = self.anomaly_scores  # Use anomaly scores as risk scores
        
        # Create categorical risk levels
        if risk_levels == 4:
            labels = ['Low', 'Medium', 'High', 'Critical']
            self.data['risk_category'] = pd.cut(
                self.risk_scores,
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=labels,
                include_lowest=True
            )
        elif risk_levels == 3:
            labels = ['Low', 'Medium', 'High']
            self.data['risk_category'] = pd.cut(
                self.risk_scores,
                bins=[0, 0.33, 0.67, 1.0],
                labels=labels,
                include_lowest=True
            )
        else:
            # Custom number of levels
            labels = [f'Level_{i}' for i in range(1, risk_levels + 1)]
            self.data['risk_category'] = pd.qcut(
                self.risk_scores,
                q=risk_levels,
                labels=labels
            )
        
        self.data['risk_score'] = self.risk_scores
        
        logger.info(f"Risk categories created: {', '.join(labels)}")
        
        return self.risk_scores
    
    def get_top_anomalies(self, n: int = 100) -> pd.DataFrame:
        """
        Get top N most anomalous transactions.
        
        Args:
            n: Number of top anomalies to return
        
        Returns:
            DataFrame with top anomalies, sorted by anomaly score (descending)
        """
        if self.anomaly_scores is None:
            raise ValueError("No anomaly scores available. Train the model first.")
        
        # Get indices of top anomalies
        top_indices = np.argsort(self.anomaly_scores)[-n:][::-1]
        
        # Create result dataframe
        result = self.data.iloc[top_indices].copy()
        result['rank'] = range(1, n + 1)
        
        # Reorder columns to show important info first
        priority_cols = ['rank', 'anomaly_score', 'is_anomaly', 'Class']
        other_cols = [col for col in result.columns if col not in priority_cols]
        result = result[priority_cols + other_cols]
        
        return result
    
    def analyze_anomalies(self, percentile_threshold: float = 99) -> Dict:
        """
        Analyze detected anomalies to find patterns.
        
        Args:
            percentile_threshold: Percentile to use for defining "anomaly" (default: 99 = top 1%)
        
        Returns:
            Dictionary with anomaly analysis
        """
        if self.anomaly_scores is None:
            raise ValueError("No anomaly scores available. Train the model first.")
        
        # Define anomaly threshold
        threshold = np.percentile(self.anomaly_scores, percentile_threshold)
        is_high_risk = self.anomaly_scores > threshold
        
        high_risk = self.data[is_high_risk]
        normal_risk = self.data[~is_high_risk]
        
        analysis = {
            'summary': {
                'threshold_percentile': percentile_threshold,
                'threshold_value': float(threshold),
                'high_risk_count': int(is_high_risk.sum()),
                'high_risk_percentage': float(is_high_risk.mean() * 100),
                'normal_risk_count': int((~is_high_risk).sum())
            }
        }
        
        # If we have true labels, analyze how well anomalies match fraud
        if self.y is not None:
            frauds_in_high_risk = high_risk['Class'].sum() if len(high_risk) > 0 else 0
            total_frauds = self.y.sum()
            
            analysis['fraud_overlap'] = {
                'frauds_in_high_risk': int(frauds_in_high_risk),
                'total_frauds': int(total_frauds),
                'percentage_frauds_caught': float(frauds_in_high_risk / total_frauds * 100) if total_frauds > 0 else 0,
                'precision_high_risk': float(frauds_in_high_risk / len(high_risk) * 100) if len(high_risk) > 0 else 0
            }
            
            # Fraud rate comparison
            analysis['fraud_rate_comparison'] = {
                'high_risk_fraud_rate': float(high_risk['Class'].mean() * 100) if len(high_risk) > 0 else 0,
                'normal_risk_fraud_rate': float(normal_risk['Class'].mean() * 100) if len(normal_risk) > 0 else 0,
                'overall_fraud_rate': float(self.y.mean() * 100)
            }
        
        # Analyze Amount distribution (if available)
        if 'Amount' in self.data.columns:
            analysis['amount_comparison'] = {
                'high_risk': {
                    'mean': float(high_risk['Amount'].mean()) if len(high_risk) > 0 else 0,
                    'median': float(high_risk['Amount'].median()) if len(high_risk) > 0 else 0,
                    'std': float(high_risk['Amount'].std()) if len(high_risk) > 0 else 0,
                    'min': float(high_risk['Amount'].min()) if len(high_risk) > 0 else 0,
                    'max': float(high_risk['Amount'].max()) if len(high_risk) > 0 else 0
                },
                'normal_risk': {
                    'mean': float(normal_risk['Amount'].mean()) if len(normal_risk) > 0 else 0,
                    'median': float(normal_risk['Amount'].median()) if len(normal_risk) > 0 else 0,
                    'std': float(normal_risk['Amount'].std()) if len(normal_risk) > 0 else 0,
                    'min': float(normal_risk['Amount'].min()) if len(normal_risk) > 0 else 0,
                    'max': float(normal_risk['Amount'].max()) if len(normal_risk) > 0 else 0
                }
            }
        
        # Risk category distribution
        if 'risk_category' in self.data.columns:
            risk_dist = self.data['risk_category'].value_counts().to_dict()
            analysis['risk_distribution'] = {str(k): int(v) for k, v in risk_dist.items()}
            
            # Fraud rate by risk category
            if self.y is not None:
                fraud_by_risk = {}
                for category in self.data['risk_category'].unique():
                    subset = self.data[self.data['risk_category'] == category]
                    fraud_by_risk[str(category)] = {
                        'count': int(len(subset)),
                        'fraud_count': int(subset['Class'].sum()),
                        'fraud_rate': float(subset['Class'].mean() * 100)
                    }
                analysis['fraud_by_risk_category'] = fraud_by_risk
        
        return analysis
    
    def plot_anomaly_scores(self, save_path: str = None):
        """Plot distribution of anomaly scores."""
        if self.anomaly_scores is None:
            logger.warning("No anomaly scores available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall distribution
        axes[0, 0].hist(self.anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.anomaly_scores.mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.anomaly_scores.mean():.3f}')
        axes[0, 0].axvline(np.percentile(self.anomaly_scores, 99), color='orange', 
                          linestyle='--', label='99th percentile')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Fraud vs Legitimate (if labels available)
        if self.y is not None:
            axes[0, 1].hist(self.anomaly_scores[self.y == 0], bins=50, alpha=0.5, 
                           label='Legitimate', density=True)
            axes[0, 1].hist(self.anomaly_scores[self.y == 1], bins=50, alpha=0.5, 
                           label='Fraud', density=True)
            axes[0, 1].set_xlabel('Anomaly Score')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Anomaly Score: Fraud vs Legitimate')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No labels available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Fraud vs Legitimate (N/A)')
        
        # 3. Box plot comparison
        if self.y is not None:
            data_to_plot = [self.anomaly_scores[self.y == 0], self.anomaly_scores[self.y == 1]]
            axes[1, 0].boxplot(data_to_plot, labels=['Legitimate', 'Fraud'])
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Anomaly Score Distribution by Class')
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No labels available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Box Plot (N/A)')
        
        # 4. Cumulative distribution
        sorted_scores = np.sort(self.anomaly_scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 1].plot(sorted_scores, cumulative, linewidth=2)
        axes[1, 1].axhline(0.99, color='red', linestyle='--', label='99th percentile')
        axes[1, 1].axvline(np.percentile(self.anomaly_scores, 99), color='red', linestyle='--')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Anomaly Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly score plots saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve for the model."""
        if self.y is None:
            logger.warning("No true labels available. Cannot plot ROC curve.")
            return
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(self.y, self.anomaly_scores)
        roc_auc = roc_auc_score(self.y, self.anomaly_scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'Isolation Forest (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curve - Isolation Forest Anomaly Detection', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """Plot Precision-Recall curve (better for imbalanced data)."""
        if self.y is None:
            logger.warning("No true labels available. Cannot plot PR curve.")
            return
        
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.anomaly_scores)
        avg_precision = average_precision_score(self.y, self.anomaly_scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, linewidth=2, 
                label=f'Isolation Forest (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Isolation Forest', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()