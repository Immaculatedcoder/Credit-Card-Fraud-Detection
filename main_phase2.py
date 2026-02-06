from src.data_processor import TransactionDataProcessor
from src.anomaly_detector import IsolationForestDetector
import json
import pandas as pd


def print_section(title):
    """Helper to print section headers"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    # Phase 1: Load and prepare data
    print_section("PHASE 1: DATA LOADING & FEATURE SELECTION")
    
    processor = TransactionDataProcessor("data/creditcard_2023.csv")
    processor.load_data()
    processor.validate_data()
    processor.clean_data()
    
    # Feature selection with Random Forest
    print("\nPerforming feature selection with Random Forest...")
    feature_results = processor.select_features_random_forest(
        n_estimators=1000,
        max_depth=100,
        random_state=42
    )
    
    print(f"\nRandom Forest Feature Selection Complete")
    # print(f"  ROC-AUC: {feature_results['model_performance']['roc_auc_score']:.4f}")
    print(f"  Test Accuracy: {feature_results['model_performance']['test_accuracy']:.4f}")
    
    # Get reduced dataset with selected features
    reduced_data = processor.get_reduced_dataset()
    selected_features = processor.selected_features
    
    print(f"\n✓ Selected {len(selected_features)} important features:")
    print(f"  {', '.join(selected_features[:10])}...")
    
    # Phase 2: Anomaly Detection with Isolation Forest
    print_section("PHASE 2: ISOLATION FOREST ANOMALY DETECTION")
    
    # Initialize detector
    print("\nInitializing Isolation Forest detector...")
    detector = IsolationForestDetector(reduced_data, selected_features)
    detector.prepare_features()
    
    # Train the model
    print("\nTraining Isolation Forest...")
    results = detector.train(
        contamination='auto',  # Will use actual fraud rate
        n_estimators=1000,
        max_samples=80000,
        random_state=42
    )
    
    # Display training results
    print_section("TRAINING RESULTS")
    print(json.dumps(results['training_params'], indent=2))
    print(json.dumps(results['detection_summary'], indent=2))
    
    # Display evaluation metrics
    if 'evaluation' in results:
        print_section("MODEL PERFORMANCE")
        eval_metrics = results['evaluation']
        
        print("\nConfusion Matrix:")
        cm = eval_metrics['confusion_matrix']
        print(f"  True Negatives:  {cm['true_negatives']:>8,}")
        print(f"  False Positives: {cm['false_positives']:>8,}")
        print(f"  False Negatives: {cm['false_negatives']:>8,}")
        print(f"  True Positives:  {cm['true_positives']:>8,}")
        
        print("\nPerformance Metrics:")
        metrics = eval_metrics['metrics']
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        
        print("\nFraud Detection Performance:")
        fraud_detect = eval_metrics['fraud_detection']
        print(f"  Frauds Detected:   {fraud_detect['frauds_detected']:>6,} / {fraud_detect['total_frauds']:>6,} "
              f"({fraud_detect['detection_rate']*100:.1f}%)")
        print(f"  Frauds Missed:     {fraud_detect['frauds_missed']:>6,}")
        print(f"  False Alarms:      {fraud_detect['false_alarms']:>6,} "
              f"({fraud_detect['false_alarm_rate']*100:.2f}%)")
        
        print("\nOptimal Threshold:")
        opt = eval_metrics['optimal_threshold']
        print(f"  Value:    {opt['value']:.4f}")
        print(f"  F1-Score: {opt['f1_score']:.4f}")
        print(f"  Note:     {opt['explanation']}")
    
    # Create risk scores
    print_section("CREATING RISK SCORES")
    detector.create_risk_scores(risk_levels=4)
    print("✓ Risk categories created: Low, Medium, High, Critical")
    
    # Analyze anomalies
    print_section("ANOMALY ANALYSIS")
    analysis = detector.analyze_anomalies(percentile_threshold=99)
    
    print("\nHigh-Risk Transaction Summary (Top 1%):")
    print(json.dumps(analysis['summary'], indent=2))
    
    if 'fraud_overlap' in analysis:
        print("\nFraud Detection in High-Risk Transactions:")
        print(json.dumps(analysis['fraud_overlap'], indent=2))
        
        print("\nFraud Rate Comparison:")
        print(json.dumps(analysis['fraud_rate_comparison'], indent=2))
    
    if 'amount_comparison' in analysis:
        print("\nTransaction Amount Analysis:")
        print(json.dumps(analysis['amount_comparison'], indent=2))
    
    # Get top anomalies
    print_section("TOP 20 MOST ANOMALOUS TRANSACTIONS")
    
    top_20 = detector.get_top_anomalies(n=20)
    
    # Display key columns
    display_cols = ['rank', 'anomaly_score', 'Class', 'Amount']
    # Add first few selected features
    display_cols += [f for f in selected_features[:3] if f in top_20.columns]
    display_cols = [col for col in display_cols if col in top_20.columns]
    
    print(top_20[display_cols].to_string(index=False))
    
    # Calculate statistics for top anomalies
    if 'Class' in top_20.columns:
        actual_frauds = top_20['Class'].sum()
        print(f"\n✓ Top 20 Anomalies Analysis:")
        print(f"  Actual frauds: {actual_frauds} ({actual_frauds/20*100:.1f}%)")
        print(f"  False alarms:  {20 - actual_frauds} ({(20-actual_frauds)/20*100:.1f}%)")
    
    # Risk category distribution
    print_section("RISK CATEGORY DISTRIBUTION")
    
    risk_dist = detector.data['risk_category'].value_counts().sort_index()
    
    print("\nTransactions by Risk Level:")
    print(f"{'Category':<12} {'Count':>12} {'Percentage':>12}")
    print("-" * 40)
    for category in ['Low', 'Medium', 'High', 'Critical']:
        if category in risk_dist.index:
            count = risk_dist[category]
            pct = count / len(detector.data) * 100
            print(f"{category:<12} {count:>12,} {pct:>11.2f}%")
    
    # Fraud rate by risk category
    if 'Class' in detector.data.columns:
        print("\nFraud Rate by Risk Category:")
        print(f"{'Category':<12} {'Transactions':>12} {'Frauds':>10} {'Fraud Rate':>12}")
        print("-" * 50)
        
        for category in ['Low', 'Medium', 'High', 'Critical']:
            subset = detector.data[detector.data['risk_category'] == category]
            if len(subset) > 0:
                fraud_count = subset['Class'].sum()
                fraud_rate = subset['Class'].mean() * 100
                print(f"{category:<12} {len(subset):>12,} {fraud_count:>10} {fraud_rate:>11.2f}%")
    
    # Generate visualizations
    print_section("GENERATING VISUALIZATIONS")
    
    try:
        print("1. Creating anomaly score distribution plots...")
        detector.plot_anomaly_scores(save_path='anomaly_scores_distribution.png')
        
        print("2. Creating ROC curve...")
        detector.plot_roc_curve(save_path='roc_curve.png')
        
        print("3. Creating Precision-Recall curve...")
        detector.plot_precision_recall_curve(save_path='precision_recall_curve.png')
        
        print("\nAll visualizations saved successfully")
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("(This is okay if running in a non-GUI environment)")
    
    # Save results
    print_section("SAVING RESULTS")
    
    # Save data with risk scores and anomaly scores
    output_data = detector.data.copy()
    output_data.to_csv('data/transactions_with_scores.csv', index=False)
    print(f"Saved all transactions with scores to 'data/transactions_with_scores.csv'")
    
    # Save top 1000 anomalies for review
    top_1000 = detector.get_top_anomalies(n=1000)
    top_1000.to_csv('data/top_1000_anomalies.csv', index=False)
    print(f"Saved top 1000 anomalies to 'data/top_1000_anomalies.csv'")
    
    # Save comprehensive report
    comprehensive_report = {
        'phase_1_feature_selection': {
            'algorithm': 'Random Forest',
            'features_selected': len(selected_features),
            'selected_features': selected_features,
            'rf_performance': feature_results['model_performance']
        },
        'phase_2_anomaly_detection': {
            'algorithm': 'Isolation Forest',
            'training_results': results,
            'anomaly_analysis': analysis
        }
    }
    
    with open('data/anomaly_detection_report.json', 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    print(f"Saved comprehensive report to 'data/anomaly_detection_report.json'")
    
  
    


if __name__ == "__main__":
    main()