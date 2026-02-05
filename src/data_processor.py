# Importing all Dependencie

import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
    
    def vaidate_data(self) -> Dict:
        """ 
        Validate the loaded data for required columns and data quality.

        Returns:
            Dictionary containing validation results
        """
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call .load_data() first")
        
        

