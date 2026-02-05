# Testing what I have done so far,

from src.data_processor import TransactionDataProcessor

def main():
    processor = TransactionDataProcessor("data/creditcard_2023.csv")
    
    try: 
        df = processor.load_data()
        print(df.head())
    except FileNotFoundError:
        print("Handled in main: file not found!")
    try: 
        df = processor.validate_data()
    except ValueError:
        print("Handle in main: Call .load_data() first!")
    

if __name__ == "__main__":
    main()