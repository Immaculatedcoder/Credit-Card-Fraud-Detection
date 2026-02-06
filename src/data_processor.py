# Importing all Dependencie

import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionDataProcessor:
    """
    This processes the Kaggle Credit card Fraud Detection Dataset 2023.
    It helps handle validation, cleaning, feature engineering.
    """
    # Class Attributes
    REQUIRED_COLUMNS = ['Time', 'Amount', 'Class']

    def __init__(self, csv_path: str):
        """
        :param self: Initialize the processor with a CSV file path.
        :param csv_path: Path to where my data was stored E.g "data/creditcard_2023.csv"
        :type csv_path: str
        """

        # Instance Attributes
        self.csv_path = csv_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.validation_report: Dict = {}

    
    def load_data(self) -> pd.DataFrame:
        """
        Instance Method1: We want to convert that .csv to dataframe
        """
        try: 
            self.raw_data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.raw_data)} transactions from {self.csv_path}")
            logger.info(f"Columns: {list(self.raw_data.columns)} ")
            return self.raw_data
        except FileNotFoundError:
            logger.error(f"File is not part of your directory: {self.csv_path}")
            raise
        except Exception as exactError:
            logger.error(f"Error loading data: {exactError}")
            raise
    
    def validate_data(self) -> Dict:
        """ 
        Validate the loaded data for required columns and data quality.

        Returns:
            Dictionary containing validation results
        """
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call .load_data() first")
        
        df = self.raw_data

        v_columns = [col for col in df.columns if col[0] == "V"]

        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'v_features_count': len(v_columns),
            'missing_columns': [],
            'missing_values_count': int(df.isna().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'fraud_count': int((df['Class'] == 1).sum()),
            'legitimate_count': int((df['Class'] == 0).sum()),
            'fraud_percentage': float((df['Class'] == 1).mean() * 100),
            'invalid_amounts': int((df['Amount'] < 0).sum()),
            'amount_range': {
                'min': float(df['Amount'].min()),
                'max': float(df['Amount'].max()),
                'mean': float(df['Amount'].mean()),
                'median': float(df['Amount'].median())
            },
            'time_range': {
                'min': float(df['Time'].min()) if 'Time' in df.columns else None,
                'max': float(df['Time'].max()) if 'Time' in df.columns else None,
                'duration_hours': float((df['Time'].max() - df['Time'].min())/3600) if 'Time' in df.columns else None,
            }
        }

        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        report['missing_columns'] = list(missing_cols)

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")

        self.validation_report = report
        logger.info(f"Validation complete:")
        logger.info(f" - Total transactions: {report['total_rows']:,}")
        logger.info(f" - Fraud cases: {report['fraud_count']:,} ({report['fraud_percentage']:.2f}%)")
        logger.info(f" - Legitimate cases: {report['legitimate_count']:,}")
        logger.info(f" - Missing values: {report['missing_values_count']}")
        logger.info(f" - Duplicate rows: {report['duplicate_rows']}")

        return report
    
    def clean_data(self) -> pd.DataFrame:
        """ 
        Clean the data by handling missing values, duplicates, and invalid entries

        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No date loaded. Call .load_data() first")
        
        df = self.raw_data.copy()
        initial_rows = len(df)

        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate transactions")

        if 'Amount' in df.columns:
            invalid_amounts = (df['Amount'] < 0).sum()
            df = df[df['Amount'] >= 0]

            if invalid_amounts > 0:
                logger.info(f"Removed {invalid_amounts} transactions with negative amounts")


        missing_before = df.isna().sum().sum()
        if missing_before > 0:
            for col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)
                        logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with median")

        logger.info(f"Cleaned data: {len(df)} transactions remaining (removed {initial_rows - len(df)} rows)")
        self.processed_data = df
        return df

    def get_fraud_analysis(self) -> Dict:
        """ 
        Specific analysis focusing on fraudulent transactions.

        Return:
            Dictinary containing fraud-specific insights
        """

        if self.processed_data is None:
            raise ValueError("No processed data, .load_data() .clean_data() ")
        
        df = self.processed_data
        fraud = df[df['Class'] == 1]
        legit = df[df['Class'] == 0]

        if len(fraud) == 0:
            return {'message': 'No fraudulent transactions found in dataset'}
        
        analysis = {
            'fraud_detection_metrics': {
                'total_fraud_cases': len(fraud),
                'fraud_percentage': float(len(fraud)/len(df)*100),
                'average_fraud_amount': float(fraud['Amount'].mean()),
                'median_fraud_amount': float(fraud['Amount'].median()),
                'total_fraud_loss': float(fraud['Amount'].sum())
            },
            'comparison': {
                'avg_amount_fraud_vs_legit': {
                    'fraud': float(fraud['Amount'].mean()),
                    'legitimate': float(legit['Amount'].mean()),
                    'ratio': float(fraud['Amount'].mean() / (legit['Amount'].mean() + 1e-6))
                },
                'median_amount_fraud_vs_legit': {
                    'fraud': float(fraud['Amount'].median()),
                    'legitimate': float(legit['Amount'].median())
                }
            }
        }
        return analysis
    
    def select_features_random_forest(
            self,
            n_estimators: int = 100,
            max_depth: int = 10,
            min_samples_split: int = 100,
            test_size: float = 0.3,
            random_state: int = 42
    ) -> Dict:
        """
        We choose random forest to determine the most important features for fraud detection.

        Args:
            n_estimators: Number of tress in the forest
            max_depth: Maximum depth of trees (preventing overfitting on imbalanced data)
            min_samples_split: Minimum samples required to split a node
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing feature importance results and selected features
        """

        if self.processed_data is None:
            raise ValueError("No processed data. implement .clean_data()")
        
        ## Step 1: Load the Dataset
        df = self.processed_data.copy()

        feature_columns = [col for col in df.columns if col[0] == "V"] + ['Amount']
        X = df[feature_columns]  # variables
        y = df['Class']

        logger.info(f"Training Random Forest with {len(feature_columns)} features")
        logger.info(f"Dataset: {len(X)} samples, {y.sum()} fraud cases ({y.mean()*100:.2f}%)")

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state, stratify=y)

        ## Step 2: Train a Random Forest Model (Before Feature Selection)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        logger.info("Training Random Forest classifier...")
        rf.fit(X_train, y_train)

        ## Perform Feature Selection using random forest
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)

        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")

        feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()

        # Different selection strategies
        top_10_features = feature_importance.head(10)['feature'].tolist()

        threshold_95 = feature_importance[feature_importance['cumulative_importance'] <= 0.95]['feature'].tolist()
        threshold_1pct = feature_importance[feature_importance['importance'] > 0.01]['feature'].tolist()

        results = {
            'model_performance' : {
                'train_accuracy': float(train_score),
                'test_accuracy': float(test_score),
            },
            'feature_importance': feature_importance.to_dict('records'),
            'feature_selection_strategies': {
                'top_10_features': top_10_features,
                'cumulative_95_percent': threshold_95,
                'importance_above_1_percent': threshold_1pct
            },
            'recommendations' : {
                'suggested_features': threshold_1pct
            }
        }

        self.rf_model = rf
        self.feature_importance = feature_importance
        self.selected_features = threshold_1pct

        logger.info(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info(f"\nFeature Selection Summary:")
        logger.info(f"  - Top 10: {len(top_10_features)} features")
        logger.info(f"  - Cumulative 95%: {len(threshold_95)} features")
        logger.info(f"  - Above 1% importance: {len(threshold_1pct)} features (RECOMMENDED)")

        return results
    
    def get_reduced_dataset(self, features: List[str] = None) -> pd.DataFrame:
        """
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        if features is None:
            if not hasattr(self, 'selected_features'):
                raise ValueError("No features selected. Run select_features_random_forest() first.")
            features = self.selected_features
        
        columns_to_keep = features + ['Class']
        reduced_df = self.processed_data[columns_to_keep].copy()

        logger.info(f"Reduced dataset shape: {reduced_df.shape}")
        logger.info(f"Features included: {features}")

        return reduced_df

    def process_pipeline(self) -> pd.DataFrame:
        """
        load -> validate -> clean -> 
        """
        self.load_data()
        self.validate_data()
        self.clean_data()
        self.get_fraud_analysis()
        logger.info("Processing pipeline complete")
        return self.processed_data

