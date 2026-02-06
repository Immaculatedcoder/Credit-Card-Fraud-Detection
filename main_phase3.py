from src.data_processor import TransactionDataProcessor
from src.anomaly_detector import IsolationForestDetector
from src.llm_analyzer import FraudAnalysisLLM
import json
import pandas as pd


def print_section(title):
    """Helper to print section headers"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def save_to_file(content: str, filename: str):
    """Save content to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Saved to {filename}")


def main():
    # Phase 1 & 2: Load data and run anomaly detection
    print_section("PHASE 1 & 2: RUNNING ANOMALY DETECTION")
    print("(This will take a few minutes...)\n")
    
    # Data loading
    processor = TransactionDataProcessor("data/creditcard_2023.csv")
    processor.load_data()
    processor.validate_data()
    processor.clean_data()
    
    # Get dataset info
    dataset_info = {
        'total_transactions': len(processor.processed_data),
        'fraud_count': int(processor.processed_data['Class'].sum()),
        'fraud_rate': float(processor.processed_data['Class'].mean())
    }
    
    print(f"Dataset: {dataset_info['total_transactions']:,} transactions")
    print(f"Frauds: {dataset_info['fraud_count']:,} ({dataset_info['fraud_rate']*100:.2f}%)")
    
    # Feature selection
    print("\nRunning feature selection...")
    feature_results = processor.select_features_random_forest(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    reduced_data = processor.get_reduced_dataset()
    selected_features = processor.selected_features
    print(f"✓ Selected {len(selected_features)} features")
    
    # Anomaly detection
    print("\nTraining Isolation Forest...")
    detector = IsolationForestDetector(reduced_data, selected_features)
    detector.prepare_features()
    
    results = detector.train(
        contamination=dataset_info['fraud_rate'],
        n_estimators=200,
        max_samples='auto',
        random_state=42
    )
    
    detector.create_risk_scores(risk_levels=4)
    risk_analysis = detector.analyze_anomalies(percentile_threshold=99)
    top_anomalies = detector.get_top_anomalies(n=100)
    
    print(f"✓ Anomaly detection complete")
    if 'evaluation' in results:
        print(f"  ROC-AUC: {results['evaluation']['metrics']['roc_auc']:.4f}")
        print(f"  Detection Rate: {results['evaluation']['fraud_detection']['detection_rate']*100:.1f}%")
    
    # Phase 3: LLM Analysis with OpenAI & LangChain
    print_section("PHASE 3: AI-POWERED ANALYSIS (OpenAI + LangChain)")
    
    # Initialize LLM
    print("\nInitializing OpenAI with LangChain...")
    try:
        llm = FraudAnalysisLLM(
            model="gpt-3.5-turbo",  # You can change to "gpt-4-turbo" or "gpt-3.5-turbo"
            temperature=0.3
        )
        print("✓ OpenAI initialized successfully with LangChain")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease set your OPENAI_API_KEY:")
        print("1. Create a .env file in your project directory")
        print("2. Add: OPENAI_API_KEY=your_key_here")
        print("3. Get your key from: https://platform.openai.com/api-keys")
        return
    
    # Load analysis context
    print("Loading analysis context into LLM...")
    llm.load_analysis_context(
        detection_results=results,
        top_anomalies=top_anomalies,
        risk_analysis=risk_analysis,
        dataset_info=dataset_info
    )
    print("✓ Context loaded")
    
    # 1. Generate Executive Summary
    print_section("1. EXECUTIVE SUMMARY")
    print("Generating executive summary with OpenAI...\n")
    
    executive_summary = llm.generate_executive_summary()
    print(executive_summary)
    print()
    save_to_file(executive_summary, 'reports/executive_summary.md')
    
    # 2. Generate Pattern Analysis
    print_section("2. FRAUD PATTERN ANALYSIS")
    print("Analyzing fraud patterns...\n")
    
    pattern_analysis = llm.generate_pattern_analysis()
    print(pattern_analysis)
    print()
    save_to_file(pattern_analysis, 'reports/pattern_analysis.md')
    
    # 3. Explain Top Anomalies
    print_section("3. TOP ANOMALY EXPLANATIONS")
    print("Generating explanations for top 5 anomalies...\n")
    
    anomaly_explanations = []
    
    for idx in range(min(5, len(top_anomalies))):
        transaction = top_anomalies.iloc[idx].to_dict()
        
        print(f"\n--- Anomaly Rank #{idx + 1} ---")
        print(f"Anomaly Score: {transaction['anomaly_score']:.4f}")
        print(f"Amount: ${transaction.get('Amount', 0):.2f}")
        print(f"Actual Status: {'FRAUD' if transaction.get('Class', 0) == 1 else 'Legitimate'}")
        print(f"Risk Category: {transaction.get('risk_category', 'N/A')}\n")
        
        explanation = llm.explain_anomaly(transaction)
        print(explanation)
        print()
        
        anomaly_explanations.append({
            'rank': idx + 1,
            'transaction': transaction,
            'explanation': explanation
        })
    
    # Save explanations
    explanations_text = "\n\n".join([
        f"# Anomaly Rank {item['rank']}\n\n"
        f"**Anomaly Score:** {item['transaction']['anomaly_score']:.4f}\n"
        f"**Amount:** ${item['transaction'].get('Amount', 0):.2f}\n"
        f"**Actual Status:** {'FRAUD' if item['transaction'].get('Class', 0) == 1 else 'Legitimate'}\n\n"
        f"{item['explanation']}"
        for item in anomaly_explanations
    ])
    save_to_file(explanations_text, 'reports/anomaly_explanations.md')
    
    # 4. Risk Category Reports
    print_section("4. RISK CATEGORY REPORTS")
    
    risk_reports = {}
    for category in ['Critical', 'High', 'Medium', 'Low']:
        print(f"\nGenerating report for {category} risk category...")
        report = llm.generate_risk_report(category)
        risk_reports[category] = report
        
        print(f"\n--- {category} Risk Category ---")
        print(report)
        print()
    
    # Save risk reports
    all_risk_reports = "\n\n---\n\n".join([
        f"# {category} Risk Category\n\n{report}"
        for category, report in risk_reports.items()
    ])
    save_to_file(all_risk_reports, 'reports/risk_category_reports.md')
    
    # 5. Interactive Q&A
    print_section("5. INTERACTIVE Q&A SESSION")
    print("\nYou can now ask questions about the fraud detection analysis.")
    print("LangChain will maintain conversation context across questions.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    # Example questions
    example_questions = [
        "What are the main patterns that distinguish fraudulent transactions?",
        "How reliable is this fraud detection system?",
        "What should be our top priority for improving fraud detection?",
        "Why is the fraud rate so high even in the Low risk category?",
        "How can we reduce false positives?"
    ]
    
    print("Example questions you can ask:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "-" * 80)
    
    # Interactive loop
    while True:
        try:
            question = input("\nYour question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nEnding Q&A session...")
                break
            
            if not question:
                continue
            
            print("\n🤔 Thinking...\n")
            answer = llm.chat(question)
            print(answer)
            
        except KeyboardInterrupt:
            print("\n\nQ&A session interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Save Q&A history
    qa_history = llm.get_conversation_history()
    if qa_history:
        qa_text = "\n\n---\n\n".join([
            f"**Q: {item['question']}**\n\n{item['answer']}"
            for item in qa_history
        ])
        save_to_file(qa_text, 'reports/qa_session.md')
    
    # Final Summary
    print_section("PHASE 3 COMPLETE!")
    
    print("\n✓ Generated executive summary")
    print("✓ Analyzed fraud patterns")
    print("✓ Explained top 5 anomalies")
    print("✓ Created risk category reports")
    print(f"✓ Answered {len(qa_history)} questions with conversation memory")
    
    print("\nAll reports saved to 'reports/' directory:")
    print("  - executive_summary.md")
    print("  - pattern_analysis.md")
    print("  - anomaly_explanations.md")
    print("  - risk_category_reports.md")
    if qa_history:
        print("  - qa_session.md")
    
    print("\n" + "=" * 80)
    print(" PROJECT COMPLETE - ALL PHASES FINISHED!")
    print("=" * 80)
    print("\nYou've successfully built a GenAI fraud detection assistant that:")
    print("  1. ✓ Processes transaction data")
    print("  2. ✓ Selects important features with Random Forest")
    print("  3. ✓ Detects anomalies with Isolation Forest")
    print("  4. ✓ Generates natural language summaries with OpenAI")
    print("  5. ✓ Answers questions interactively with LangChain memory")
    print("\nCongratulations! 🎉")
    print("\nTechnologies mastered:")
    print("  • Python data processing (pandas, numpy)")
    print("  • Machine Learning (scikit-learn)")
    print("  • LLM Integration (OpenAI API)")
    print("  • LangChain framework")
    print("  • Prompt engineering")
    print("  • Real-world AI application development")


if __name__ == "__main__":
    # Create reports directory if it doesn't exist
    import os
    os.makedirs('reports', exist_ok=True)
    
    main()