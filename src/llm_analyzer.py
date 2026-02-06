import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging
import time
from openai import RateLimitError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class FraudAnalysisLLM:
    """
    Uses OpenAI (via LangChain) to generate natural language summaries
    and answer questions about credit card fraud detection results.
    """

    def __init__(
           self,
           api_key: Optional[str] = None,
           model: str = "gpt-4o",
           temperature: float = 0.3
    ):
        """
        Docstring for __init__
        
        :param self: Description
        :param api_key: Description
        :type api_key: Optional[str]
        :param model: Description
        :type model: str
        :param temperature: Description
        :type temperature: float
        """

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model,
            temperature=temperature
        )
        
        self.chat_history_store = {}
        self.session_id = "fraud_analysis_session"

        self.detection_results = None
        self.top_anomalies = None
        self.risk_analysis = None
        self.dataset_info = None
        self.context_summary = None

        logger.info(f"LLM Analyzer initialized with {model}")

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create chat history for a session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Chat message history
        """
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = InMemoryChatMessageHistory()
        return self.chat_history_store[session_id]

    def load_analysis_context(
        self,
        detection_results: Dict,
        top_anomalies: pd.DataFrame,
        risk_analysis: Dict,
        dataset_info: Dict = None
    ):
        """
        Load the fraud detection results for analysis.
        
        Args:
            detection_results: Results from Isolation Forest training
            top_anomalies: DataFrame with top anomalous transactions
            risk_analysis: Risk category analysis
            dataset_info: Optional dataset metadata
        """
        self.detection_results = detection_results
        self.top_anomalies = top_anomalies
        self.risk_analysis = risk_analysis
        self.dataset_info = dataset_info or {}
        
        # Prepare context summary once
        self.context_summary = self._prepare_context_summary()
        
        logger.info("Analysis context loaded successfully")
    
    def _prepare_context_summary(self) -> str:
        """
        Prepare a structured summary of the analysis for the LLM.
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Dataset overview
        if self.dataset_info:
            context_parts.append("## Dataset Overview")
            context_parts.append(f"- Total Transactions: {self.dataset_info.get('total_transactions', 'N/A'):,}")
            context_parts.append(f"- Fraud Cases: {self.dataset_info.get('fraud_count', 'N/A'):,}")
            context_parts.append(f"- Fraud Rate: {self.dataset_info.get('fraud_rate', 0)*100:.2f}%")
            context_parts.append("")
        
        # Model performance
        if self.detection_results and 'evaluation' in self.detection_results:
            eval_data = self.detection_results['evaluation']
            context_parts.append("## Model Performance")
            context_parts.append(f"- Algorithm: Isolation Forest (unsupervised anomaly detection)")
            context_parts.append(f"- ROC-AUC Score: {eval_data['metrics']['roc_auc']:.4f}")
            context_parts.append(f"- Precision: {eval_data['metrics']['precision']:.4f}")
            context_parts.append(f"- Recall: {eval_data['metrics']['recall']:.4f}")
            context_parts.append(f"- F1-Score: {eval_data['metrics']['f1_score']:.4f}")
            context_parts.append(f"- Frauds Detected: {eval_data['fraud_detection']['frauds_detected']:,} / {eval_data['fraud_detection']['total_frauds']:,}")
            context_parts.append(f"- Detection Rate: {eval_data['fraud_detection']['detection_rate']*100:.1f}%")
            context_parts.append(f"- False Alarms: {eval_data['fraud_detection']['false_alarms']:,}")
            context_parts.append("")
        
        # Risk categories
        if self.risk_analysis and 'fraud_by_risk_category' in self.risk_analysis:
            context_parts.append("## Risk Category Analysis")
            for category in ['Critical', 'High', 'Medium', 'Low']:
                if category in self.risk_analysis['fraud_by_risk_category']:
                    stats = self.risk_analysis['fraud_by_risk_category'][category]
                    context_parts.append(
                        f"- {category}: {stats['count']:,} transactions, "
                        f"{stats['fraud_count']:,} frauds ({stats['fraud_rate']:.2f}% fraud rate)"
                    )
            context_parts.append("")
        
        # Top anomalies sample
        if self.top_anomalies is not None and len(self.top_anomalies) > 0:
            context_parts.append("## Top 10 Anomalies Sample")
            top_10 = self.top_anomalies.head(10)
            
            for idx, row in top_10.iterrows():
                fraud_status = "FRAUD" if row.get('Class', 0) == 1 else "Legitimate"
                amount = row.get('Amount', 0)
                score = row.get('anomaly_score', 0)
                context_parts.append(
                    f"- Rank {row.get('rank', '?')}: Score={score:.3f}, "
                    f"Amount=${amount:.2f}, Actual={fraud_status}"
                )
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_executive_summary(
        self,
        focus_areas: List[str] = None,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate an executive summary using LangChain.
        
        Args:
            focus_areas: Optional list of specific areas to focus on
            max_tokens: Maximum length of summary
        
        Returns:
            Executive summary as a string
        """
        logger.info("Generating executive summary...")
        
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\n\nFocus particularly on: {', '.join(focus_areas)}"
        
        # Create prompt template
        system_template = """You are a fraud detection analyst preparing an executive summary for senior management.
You specialize in explaining technical fraud detection results in clear, business-friendly language."""

        human_template = """Based on the following fraud detection analysis results, create a clear, concise executive summary.

{context}

Please create an executive summary that includes:

1. **Overview**: Brief description of what was analyzed
2. **Key Findings**: Most important discoveries (3-5 bullet points)
3. **Model Performance**: How well the detection system works
4. **Risk Levels**: Explain the risk category distribution and what it means
5. **Actionable Recommendations**: 3-4 specific actions management should take

Write in clear, business-friendly language. Avoid technical jargon where possible.
Use specific numbers from the data to support your points.
Be concise but thorough.{focus_instruction}

Format the summary with clear headers and bullet points for easy reading."""

        # Create messages
        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template.format(
                context=self.context_summary,
                focus_instruction=focus_instruction
            ))
        ]
        
        try:
            # Use LangChain's invoke method
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(messages)
                    summary = response.content
                    logger.info("Executive summary generated successfully")
                    return summary
                except Exception as e:
                    if "insufficient_quota" in str(e) or "rate_limit" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise
            
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def explain_anomaly(
        self,
        transaction_data: Dict,
        max_tokens: int = 800
    ) -> str:
        """
        Explain why a specific transaction was flagged as anomalous.
        
        Args:
            transaction_data: Dictionary with transaction details
            max_tokens: Maximum length of explanation
        
        Returns:
            Plain English explanation
        """
        logger.info("Explaining anomaly...")
        
        # Format transaction data
        trans_info = []
        for key, value in transaction_data.items():
            if key == 'anomaly_score':
                trans_info.append(f"- Anomaly Score: {value:.4f} (0=normal, 1=highly anomalous)")
            elif key == 'Amount':
                trans_info.append(f"- Transaction Amount: ${value:.2f}")
            elif key == 'Class':
                status = "Confirmed Fraud" if value == 1 else "Legitimate"
                trans_info.append(f"- Actual Status: {status}")
            elif key == 'risk_category':
                trans_info.append(f"- Risk Category: {value}")
            elif key.startswith('V'):
                trans_info.append(f"- {key}: {value:.4f}")
            elif key not in ['rank', 'is_anomaly']:
                trans_info.append(f"- {key}: {value}")
        
        transaction_str = "\n".join(trans_info)
        
        # Create prompt
        system_template = """You are a fraud analyst explaining why transactions are flagged as suspicious.
You translate technical anomaly detection results into clear, actionable insights."""

        human_template = """Transaction Details:
{transaction_details}

Dataset Context:
- This is credit card transaction data
- V1-V28 are anonymized features from PCA transformation (to protect user privacy)
- Higher anomaly scores indicate more unusual patterns compared to normal transactions
- The Isolation Forest algorithm identifies transactions that are easy to isolate (outliers)

Please explain in plain English:
1. Why this transaction was flagged (based on the anomaly score and features)
2. What patterns make it suspicious
3. Whether it matches known fraud characteristics (based on the actual status if available)

Keep the explanation:
- Clear and non-technical
- Focused on what matters for fraud investigators
- Actionable
- 2-3 paragraphs maximum"""

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template.format(transaction_details=transaction_str))
        ]
        
        try:
            response = self.llm.invoke(messages)
            explanation = response.content
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining anomaly: {e}")
            raise
    
    def answer_question(
        self,
        question: str,
        use_memory: bool = True
    ) -> str:
        """
        Answer a question about the fraud detection analysis using LangChain.
        
        Args:
            question: User's question
            use_memory: Whether to use conversation memory
        
        Returns:
            Answer as a string
        """
        logger.info(f"Answering question: {question[:50]}...")
        
        if use_memory:
            # Create conversational chain with memory (new LangChain pattern)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a fraud detection analyst answering questions about a credit card fraud detection analysis.
You provide clear, accurate answers based on the data provided.
If the data doesn't contain enough information to answer fully, you say so and explain what you can determine.
You use specific numbers from the analysis when relevant.

Analysis Context:
{context}"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            # Create chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Wrap with message history
            chain_with_history = RunnableWithMessageHistory(
                chain,
                self._get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )
            
            try:
                # Invoke with session config
                response = chain_with_history.invoke(
                    {
                        "context": self.context_summary,
                        "question": question
                    },
                    config={"configurable": {"session_id": self.session_id}}
                )
                
                logger.info("Question answered successfully")
                return response
                
            except Exception as e:
                logger.error(f"Error answering question: {e}")
                raise
        else:
            # Simple question without history
            messages = [
                SystemMessage(content=f"""You are a fraud detection analyst answering questions about a credit card fraud detection analysis.

Analysis Context:
{self.context_summary}"""),
                HumanMessage(content=question)
            ]
            
            try:
                response = self.llm.invoke(messages)
                return response.content
                
            except Exception as e:
                logger.error(f"Error answering question: {e}")
                raise
    
    def generate_risk_report(
        self,
        risk_category: str,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a detailed report for a specific risk category.
        
        Args:
            risk_category: One of 'Low', 'Medium', 'High', 'Critical'
            max_tokens: Maximum length of report
        
        Returns:
            Risk category report as a string
        """
        logger.info(f"Generating risk report for {risk_category} category...")
        
        if not self.risk_analysis or 'fraud_by_risk_category' not in self.risk_analysis:
            return "Risk analysis data not available."
        
        risk_data = self.risk_analysis['fraud_by_risk_category'].get(risk_category, {})
        
        if not risk_data:
            return f"No data available for {risk_category} risk category."
        
        system_template = """You are a fraud analyst creating detailed reports for different risk categories.
You provide actionable insights for fraud prevention teams."""

        human_template = """Risk Category: {risk_category}

Statistics:
- Total Transactions: {count:,}
- Fraud Cases: {fraud_count:,}
- Fraud Rate: {fraud_rate:.2f}%

Overall Dataset Fraud Rate: {baseline_fraud_rate:.2f}%

Please create a report that includes:

1. **Category Overview**: What this risk level means
2. **Fraud Rate Analysis**: How this category compares to the baseline (is it higher/lower and by how much)
3. **Volume Assessment**: Whether the number of transactions is concerning
4. **Recommended Actions**: Specific steps for handling transactions in this category
5. **Monitoring Priorities**: What to watch for

Keep it practical and actionable for fraud prevention teams.
Use the specific numbers provided to support your analysis."""

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template.format(
                risk_category=risk_category,
                count=risk_data.get('count', 0),
                fraud_count=risk_data.get('fraud_count', 0),
                fraud_rate=risk_data.get('fraud_rate', 0),
                baseline_fraud_rate=self.dataset_info.get('fraud_rate', 0) * 100
            ))
        ]
        
        try:
            response = self.llm.invoke(messages)
            report = response.content
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise
    
    def generate_pattern_analysis(self) -> str:
        """
        Generate analysis of fraud patterns discovered.
        
        Returns:
            Pattern analysis as a string
        """
        logger.info("Generating pattern analysis...")
        
        # Analyze feature distributions if available
        feature_analysis = ""
        if self.top_anomalies is not None and len(self.top_anomalies) > 0:
            fraud_anomalies = self.top_anomalies[self.top_anomalies['Class'] == 1]
            legit_anomalies = self.top_anomalies[self.top_anomalies['Class'] == 0]
            
            feature_analysis = f"""
## Anomaly Composition
- Top 100 anomalies contain: {len(fraud_anomalies)} frauds, {len(legit_anomalies)} legitimate transactions
- Fraud precision in top anomalies: {len(fraud_anomalies)/len(self.top_anomalies)*100:.1f}%
"""
        
        system_template = """You are a fraud detection analyst identifying patterns in fraudulent transactions.
You look for actionable insights that can improve fraud prevention."""

        human_template = """Based on this fraud detection analysis, identify key patterns:

{context}
{feature_analysis}

Please analyze:

1. **Fraud Distribution Patterns**: How fraud is distributed across risk categories
2. **Detection Effectiveness**: What the model does well and where it struggles
3. **False Positives/Negatives**: Patterns in misclassifications
4. **Actionable Insights**: Specific patterns fraud teams should watch for
5. **Model Limitations**: What the current system might be missing

Focus on practical, actionable insights."""

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=human_template.format(
                context=self.context_summary,
                feature_analysis=feature_analysis
            ))
        ]
        
        try:
            response = self.llm.invoke(messages)
            analysis = response.content
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating pattern analysis: {e}")
            raise
    
    def chat(self, message: str) -> str:
        """
        Interactive chat interface for exploring the analysis.
        
        Args:
            message: User's message
        
        Returns:
            Assistant's response
        """
        return self.answer_question(message, use_memory=True)
    
    def clear_conversation_history(self):
        """Clear the conversation memory."""
        if self.session_id in self.chat_history_store:
            self.chat_history_store[self.session_id].clear()
            logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation exchanges
        """
        if self.session_id not in self.chat_history_store:
            return []
        
        history = self.chat_history_store[self.session_id].messages
        
        formatted_history = []
        i = 0
        while i < len(history):
            # Look for human message followed by AI message
            if isinstance(history[i], HumanMessage):
                human_msg = history[i].content
                ai_msg = history[i + 1].content if i + 1 < len(history) and isinstance(history[i + 1], AIMessage) else ""
                
                formatted_history.append({
                    'question': human_msg,
                    'answer': ai_msg
                })
                i += 2
            else:
                i += 1
        
        return formatted_history
    