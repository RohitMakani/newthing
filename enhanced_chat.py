import json
import requests
from typing import Dict, Any, List, Optional
import re
from datetime import datetime
import numpy as np
import pandas as pd


class IntelligentChatEngine:
    """Enhanced chat engine with context awareness and memory"""

    def __init__(self, api_key: str, api_url: str, model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.conversation_memory = []
        self.context_templates = {
            "chart_analysis": """
            You are an expert data analyst specializing in chart interpretation and data visualization insights.

            Chart Type: {chart_type}
            Extracted Data: {extracted_data}

            Context: You are analyzing a {chart_type} chart. The extracted data shows: {data_summary}

            Please provide:
            1. Key insights from the chart
            2. Notable patterns or trends
            3. Potential implications or recommendations
            4. Questions for further investigation

            Be specific, actionable, and focus on data-driven insights.
            """,

            "eda_analysis": """
            You are an expert statistician and data scientist analyzing exploratory data analysis results.

            Dataset Info: {dataset_info}
            Statistical Summary: {statistics}
            Task Type: {task_type}
            Target Variable: {target_column}

            Please provide:
            1. Key findings from the EDA
            2. Data quality assessment
            3. Feature relationships and correlations
            4. Recommendations for modeling
            5. Potential data issues to address

            Focus on actionable insights that will improve model performance.
            """,

            "model_performance": """
            You are a machine learning expert analyzing model performance results.

            Task: {task_type}
            Best Model: {best_model}
            Performance Metrics: {metrics}
            Comparison Results: {comparison}

            Please provide:
            1. Model performance interpretation
            2. Strengths and limitations of the chosen model
            3. Recommendations for improvement
            4. Business implications of the results
            5. Next steps for deployment

            Explain technical concepts in business-friendly terms when possible.
            """,

            "general_inquiry": """
            You are an AI assistant specializing in data science, machine learning, and business intelligence.

            Context: {context}
            Previous conversation: {conversation_history}

            User Question: {question}

            Please provide a helpful, accurate, and detailed response. If the question relates to the analysis context,
            reference specific findings and data points. If it's a general question, provide educational value while
            staying focused on data science topics.
            """
        }

    async def generate_response(self, message: str, context: Dict[str, Any], context_type: str = "general") -> str:
        """Generate intelligent response based on context and conversation history"""
        try:
            # Build prompt based on context type
            if context_type == "chart_analysis":
                prompt = self._build_chart_analysis_prompt(message, context)
            elif context_type == "eda_analysis":
                prompt = self._build_eda_analysis_prompt(message, context)
            elif context_type == "model_performance":
                prompt = self._build_model_performance_prompt(message, context)
            else:
                prompt = self._build_general_prompt(message, context)

            # Add conversation memory
            if self.conversation_memory:
                conversation_context = "\
".join([
                    f"User: {item['message']}\
AI: {item['response']}"
                    for item in self.conversation_memory[-3:]  # Last 3 exchanges
                ])
                prompt = f"Previous conversation:\
{conversation_context}\
\
{prompt}"

            # Make API call
            response = await self._call_groq_api(prompt)

            # Store in memory
            self.conversation_memory.append({
                "message": message,
                "response": response,
                "context_type": context_type,
                "timestamp": datetime.now().isoformat()
            })

            # Keep memory manageable
            if len(self.conversation_memory) > 10:
                self.conversation_memory = self.conversation_memory[-10:]

            return response

        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    async def generate_chart_insights(self, chart_type: str, extracted_data: Dict[str, Any],
                                      dataset: Optional[pd.DataFrame] = None) -> List[str]:
        """Generate comprehensive insights from chart analysis"""
        try:
            data_summary = self._summarize_chart_data(chart_type, extracted_data)

            prompt = f"""
            Analyze this {chart_type} chart with the following extracted data:
            {json.dumps(extracted_data, indent=2)}

            Data Summary: {data_summary}

            Provide 3-5 key insights in bullet point format. Focus on:
            - Patterns and trends visible in the data
            - Notable outliers or anomalies  
            - Business implications
            - Recommended actions

            Keep each insight concise but actionable.
            """

            response = await self._call_groq_api(prompt)

            # Parse response into list of insights
            insights = [line.strip().lstrip('•-*').strip()
                        for line in response.split('\
')
                        if line.strip() and not line.strip().startswith('Here')]

            return insights[:5]  # Limit to 5 insights

        except Exception as e:
            return [f"Error generating insights: {str(e)}"]

    async def generate_summary_insights(self, chart_data: List[Dict], dataset: pd.DataFrame) -> List[str]:
        """Generate summary insights from multiple charts"""
        try:
            summary = f"""
            Analyzed {len(chart_data)} charts from the document.
            Chart types found: {', '.join(set(chart['type'] for chart in chart_data))}
            Dataset shape: {dataset.shape}
            Dataset columns: {', '.join(dataset.columns.tolist())}
            """

            prompt = f"""
            You've analyzed multiple charts from a business intelligence report. Here's the summary:

            {summary}

            Individual chart insights:
            {json.dumps([chart.get('insights', []) for chart in chart_data], indent=2)}

            Provide 3-5 high-level strategic insights that synthesize findings across all charts:
            - Overall business performance trends
            - Key relationships between different metrics
            - Strategic recommendations
            - Areas requiring attention

            Focus on executive-level insights that connect the data to business value.
            """

            response = await self._call_groq_api(prompt)

            insights = [line.strip().lstrip('•-*').strip()
                        for line in response.split('\
')
                        if line.strip() and not line.strip().startswith('Here')]

            return insights[:5]

        except Exception as e:
            return [f"Error generating summary insights: {str(e)}"]

    def _build_chart_analysis_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build prompt for chart analysis context"""
        chart_type = context.get('chart_type', 'unknown')
        extracted_data = context.get('extracted_data', {})
        data_summary = self._summarize_chart_data(chart_type, extracted_data)

        return self.context_templates["chart_analysis"].format(
            chart_type=chart_type,
            extracted_data=json.dumps(extracted_data, indent=2),
            data_summary=data_summary
        ) + f"\
\
User Question: {message}"

    def _build_eda_analysis_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build prompt for EDA analysis context"""
        return self.context_templates["eda_analysis"].format(
            dataset_info=json.dumps(context.get('dataset_info', {}), indent=2),
            statistics=json.dumps(context.get('statistics', {}), indent=2),
            task_type=context.get('task_type', 'unknown'),
            target_column=context.get('target_column', 'unknown')
        ) + f"\
\
User Question: {message}"

    def _build_model_performance_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build prompt for model performance context"""
        return self.context_templates["model_performance"].format(
            task_type=context.get('task_type', 'unknown'),
            best_model=json.dumps(context.get('best_model', {}), indent=2),
            metrics=json.dumps(context.get('metrics', {}), indent=2),
            comparison=json.dumps(context.get('comparison_table', []), indent=2)
        ) + f"\
\
User Question: {message}"

    def _build_general_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build prompt for general inquiries"""
        conversation_history = "\
".join([
            f"Q: {item['message']}\
A: {item['response']}"
            for item in self.conversation_memory[-2:]
        ]) if self.conversation_memory else "None"

        return self.context_templates["general_inquiry"].format(
            context=json.dumps(context, indent=2) if context else "No specific context",
            conversation_history=conversation_history,
            question=message
        )

    def _summarize_chart_data(self, chart_type: str, extracted_data: Dict[str, Any]) -> str:
        """Create human-readable summary of chart data"""
        try:
            if chart_type == "bar":
                bars = extracted_data.get('visual_data', {}).get('bars', [])
                if bars:
                    return f"Bar chart with {len(bars)} bars, heights ranging from {min(bar['height'] for bar in bars)} to {max(bar['height'] for bar in bars)}"

            elif chart_type == "line":
                lines = extracted_data.get('visual_data', {}).get('lines', [])
                if lines:
                    return f"Line chart with {len(lines)} line segments detected"

            elif chart_type == "pie":
                pie_data = extracted_data.get('visual_data', {})
                segments = pie_data.get('estimated_segments', 0)
                if segments:
                    return f"Pie chart with approximately {segments} segments"

            elif chart_type == "scatter":
                points = extracted_data.get('visual_data', {}).get('points', [])
                if points:
                    return f"Scatter plot with {len(points)} data points"

            # Fallback to extracted values
            values = extracted_data.get('extracted_values', [])
            if values:
                return f"Extracted {len(values)} numeric values: {values[:5]}{'...' if len(values) > 5 else ''}"

            return "Chart data extracted, awaiting detailed analysis"

        except Exception:
            return "Chart data summary unavailable"

    async def _call_groq_api(self, prompt: str) -> str:
        """Make API call to Groq"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert data scientist and business intelligence analyst. Provide clear, actionable insights based on data analysis. Use bullet points for lists and be concise but comprehensive."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Failed to get AI response: {str(e)}"

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_memory.copy()
