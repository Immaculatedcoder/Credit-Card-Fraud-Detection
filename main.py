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



    # Feature Selection using Random Forest
    print_section("FEATURE SELECTION - RANDOM FOREST")
    print("Training Random Forest to identify most important features...")
    print("This may take a few minutes with the full dataset...\n")

    feature_results = processor.select_features_random_forest(
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        test_size=0.3,
        random_state=42
    )

    # Print model performance
    print_section("MODEL PERFORMANCE")
    print(json.dumps(feature_results['model_performance'], indent=2))

    # Print top features
    print_section("TOP 15 MOST IMPORTANT FEATURES")
    top_15 = feature_results['feature_importance'][:15]
    for i, feat in enumerate(top_15, 1):
        print(f"{i:2d}. {feat['feature']:8s} - Importance: {feat['importance']:.4f} "
              f"(Cumulative: {feat['cumulative_importance']:.2%})")
    
    print_section("FEATURE SELECTION STRATEGIES")
    strategies = feature_results['feature_selection_strategies']
    print(f"Strategy 1 - Top 10 features: {len(strategies['top_10_features'])} features")
    print(f"  Features: {', '.join(strategies['top_10_features'])}\n")

    print(f"Strategy 2 - Cumulative 95% importance: {len(strategies['cumulative_95_percent'])} features")
    print(f"  Features: {', '.join(strategies['cumulative_95_percent'])}\n")

    print(f"Strategy 4 - Importance > 1%: {len(strategies['importance_above_1_percent'])} features (RECOMMENDED)")
    print(f"  Features: {', '.join(strategies['importance_above_1_percent'])}\n")

    # Get reduced dataset with selected features
    print_section("CREATING REDUCED DATASET")
    reduced_data = processor.get_reduced_dataset()
    print(f"Original dataset: {processor.processed_data.shape}")
    print(f"Reduced dataset: {reduced_data.shape}")
    print(f"Dimensionality reduction: {processor.processed_data.shape[1]} → {reduced_data.shape[1]} columns")
    print(f"Features removed: {processor.processed_data.shape[1] - reduced_data.shape[1]}")

    # Save reduced dataset
    output_path = 'data/reduced_creditcard.csv'
    reduced_data.to_csv(output_path, index=False)
    print(f"\nReduced dataset saved to: {output_path}")

    






    # # Show sample of processed data
    # print_section("SAMPLE PROCESSED DATA (First 5 Fraud Cases)")
    # fraud_samples = processed_data[processed_data['Class'] == 1].head()
    # print(fraud_samples[['id','Amount', 'Class']].to_string())

    # print_section("SAMPLE PROCESSED DATA (First 5 Legit Cases)")
    # legit_samples = processed_data[processed_data['Class'] == 0].head()
    # print(legit_samples[['id','Amount', 'Class']].to_string())

    # # save processing data
    # print_section("SAVING PROCESSED DATA")
    # output_path = 'data/processed_creditcard.csv'
    # processed_data.to_csv(output_path, index=False)
    # print(f"Processed data saved to: {output_path}")
    # print(f"Shape: {processed_data.shape}")

    

    

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