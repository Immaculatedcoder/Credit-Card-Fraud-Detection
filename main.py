# Testing what I have done so far,

from src.data_processor import TransactionDataProcessor
import json

def print_section(title):
    print("\n"+"="*70)
    print(f"{title}")
    print("="*70)

def main():
    print_section("INITIALIZING DATA PROCESSOR")
    processor = TransactionDataProcessor("data/creditcard_2023.csv")
    
    print_section("RUNNING PROCESSING PIPLINE")
    processed_data = processor.process_pipeline()

    print_section("VALIDATING REPORT")
    print(json.dumps(processor.validation_report, indent=2))

    print_section("FRAUD ANALYSIS")
    fraud_analysis = processor.get_fraud_analysis()
    print(json.dumps(fraud_analysis, indent=2))

    # Show sample of processed data
    print_section("SAMPLE PROCESSED DATA (First 5 Fraud Cases)")
    fraud_samples = processed_data[processed_data['Class'] == 1].head()
    print(fraud_samples[['id','Amount', 'Class']].to_string())

    print_section("SAMPLE PROCESSED DATA (First 5 Legit Cases)")
    legit_samples = processed_data[processed_data['Class'] == 0].head()
    print(legit_samples[['id','Amount', 'Class']].to_string())

    # save processing data
    print_section("SAVING PROCESSED DATA")
    output_path = 'data/processed_creditcard.csv'
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print(f"Shape: {processed_data.shape}")

    

    

    # try: 
    #     df = processor.load_data()
    #     print(df.head())
    # except FileNotFoundError:
    #     print("Handled in main: file not found!")
    # try: 
    #     reports = processor.validate_data()
    # except ValueError:
    #     print("Handle in main: Call .load_data() first!")
    # try: 
    #     df_processed = processor.clean_data()
    # except ValueError:
    #     print("Handle in main: Call .load_data() first!")
    

if __name__ == "__main__":
    main()