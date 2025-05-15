import pandas as pd
import numpy as np
import random
import traceback
import re
import streamlit as st
import google.generativeai as genai

# --- Configure Gemini API ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API Configured successfully.")
    _gemini_api_configured = True
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    _gemini_api_configured = False

def calculate_metric_impact(metric, merchant_value, avg_value):
    """Calculate the impact score for a metric."""
    if pd.isna(merchant_value) or pd.isna(avg_value):
        return 0, 0
    
    # Calculate percentage difference
    if metric in ['avg_txn_value', 'daily_txn_count']:
        pct_diff = ((merchant_value - avg_value) / avg_value) * 100
        is_better = merchant_value > avg_value
    else:  # refund_rate, rent_pct_revenue
        pct_diff = ((avg_value - merchant_value) / avg_value) * 100
        is_better = merchant_value < avg_value
    
    return pct_diff, is_better

def format_metric_comparison(metric, merchant_value, avg_value, pct_diff, is_better):
    """Format metric comparison with proper units and formatting."""
    if metric == 'avg_txn_value':
        return f"₹{merchant_value:.2f} vs ₹{avg_value:.2f} ({pct_diff:+.1f}%)"
    elif metric == 'daily_txn_count':
        return f"{merchant_value:.0f} vs {avg_value:.0f} ({pct_diff:+.1f}%)"
    elif metric in ['refund_rate', 'rent_pct_revenue']:
        return f"{merchant_value*100:.1f}% vs {avg_value*100:.1f}% ({pct_diff:+.1f}%)"
    return f"{merchant_value:.2f} vs {avg_value:.2f} ({pct_diff:+.1f}%)"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def call_gemini_api(prompt, model_name="gemini-1.5-flash"):
    """Calls the configured Google Generative AI model (Gemini)."""
    if not _gemini_api_configured:
        return "### AI Analysis Error:\nGoogle API Key not configured or configuration failed."

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if response.parts:
            return response.text
        else:
            return "### AI Analysis Error:\nNo response generated from the AI model."
            
    except Exception as e:
        return f"### AI Analysis Error:\nAn error occurred: {str(e)}"

def generate_impact_visualization(merchant_row, comparison_local, comparison_cluster, response=None):
    """Generate data for impact visualization."""
    if comparison_local is None:
        return None

    impact_data = []
    
    # Split insights into individual blocks
    insights = response.strip().split('\n\n')
    
    for i, insight_block in enumerate(insights):
        if not insight_block.strip():
            continue
            
        lines = insight_block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Extract metric and impact percentage
        metric = None
        impact_pct = None
        
        # Determine metric from the first line using comprehensive detection
        insight_text = lines[0].lower()
        if 'transaction value' in insight_text or 'avg' in insight_text or 'value' in insight_text:
            metric = 'Avg Txn Value'
        elif 'daily count' in insight_text:
            metric = 'Daily Txn Count (Count)'
        elif 'daily txns' in insight_text:
            metric = 'Daily Txn Count (Txns)'
        elif 'daily' in insight_text or 'count' in insight_text or 'transaction' in insight_text or 'customers' in insight_text:
            metric = 'Daily Txn Count'
        elif 'refund' in insight_text or 'rate' in insight_text:
            metric = 'Refund Rate'
        elif 'income' in insight_text or 'level' in insight_text:
            metric = 'Income Level'
            
        # Extract impact percentage from the last line
        if '📈 IMPACT:' in lines[2]:
            match = re.search(r'(\d+)%', lines[2])
            if match:
                impact_pct = float(match.group(1)) / 100
                
        if metric and impact_pct is not None:
            # Get current and local average values
            local_row = comparison_local[comparison_local['Metric'] == 'Daily Txn Count']  # Use base metric for data
            if not local_row.empty:
                current_value = local_row.iloc[0]['Merchant Value']
                local_avg = local_row.iloc[0]['Local Avg']
                
                # Get cluster average if available
                cluster_avg = None
                if comparison_cluster is not None:
                    cluster_row = comparison_cluster[comparison_cluster['Metric'] == 'Daily Txn Count']
                    if not cluster_row.empty:
                        cluster_avg = cluster_row.iloc[0]['Cluster Avg']
                
                # Calculate expected value based on impact
                if metric == 'Income Level':
                    expected_value = current_value
                elif metric in ['Avg Txn Value', 'Daily Txn Count', 'Daily Txn Count (Count)', 'Daily Txn Count (Txns)']:
                    expected_value = current_value * (1 + impact_pct)
                else:  # Refund Rate
                    expected_value = current_value * (1 - impact_pct)
                
                impact_data.append({
                    'metric': metric,
                    'current': current_value,
                    'expected': expected_value,
                    'local_avg': local_avg,
                    'cluster_avg': cluster_avg,
                    'impact_pct': impact_pct,
                    'insight_index': i  # Add index for unique identification
                })
    
    return impact_data

def generate_crisp_insights(merchant_row, comparison_local, comparison_cluster, cluster_peers, cluster_averages):
    """Generate concise, actionable insights using Gemini AI"""
    
    # Format the data for the prompt with better aesthetics
    merchant_profile = f"""
    🏪 Business Profile:
    ──────────────────────────────
    • Industry: {merchant_row.get('industry', 'N/A')}
    • Store Type: {merchant_row.get('store_type', 'N/A')}
    • Location: {merchant_row.get('city', 'N/A')}
    • Area Income Level: ₹{merchant_row.get('income_level', 0):.2f} per month
    ──────────────────────────────
    """

    # Format performance metrics with better aesthetics
    performance_metrics = []
    if comparison_local is not None:
        for _, row in comparison_local.iterrows():
            metric = row['Metric']
            if 'income' in metric.lower():
                continue
                
            merchant_value = row['Merchant Value']
            local_avg = row['Local Avg']
            performance = row['Performance']
            gap = ((merchant_value - local_avg) / local_avg) * 100
            
            # Add context about whether the metric is good or bad
            metric_context = ""
            if metric == 'Avg Txn Value':
                if gap > 0:
                    metric_context = " (Good: Higher is better)"
                else:
                    metric_context = " (Issue: Lower than average)"
            elif metric == 'Daily Txn Count':
                if gap > 0:
                    metric_context = " (Good: Higher is better)"
                else:
                    metric_context = " (Issue: Lower than average)"
            elif metric == 'Refund Rate':
                if gap < 0:
                    metric_context = " (Good: Lower is better)"
                else:
                    metric_context = " (Issue: Higher than average)"
            
            performance_metrics.append(f"""
            📊 {metric}{metric_context}
            ──────────────────────────────
            • Your Value: {merchant_value:.2f}
            • Local Average: {local_avg:.2f}
            • Performance: {performance}
            • Gap: {gap:+.1f}%
            ──────────────────────────────
            """)

    # Create a concise prompt for Gemini with better formatting
    prompt = f"""As a retail business consultant, provide <= 3 key, actionable insights for this merchant. Keep each insight to 3 lines maximum:

{merchant_profile}

📈 Current Performance:
{''.join(performance_metrics)}

Format each insight using **exactly** this structure, with a **blank line between each insight**:

💪 STRENGTH: High avg transaction
💡 ACTION: Offer combo meals
📈 IMPACT: Boost by 5%

OR

🎯 OPPORTUNITY: Low daily count
💡 ACTION: Run lunch specials
📈 IMPACT: Increase by 15%

Rules:
- Keep each line under 40 characters
- Use specific numbers
- Always give specific expected percentage change in impact
- Focus on immediate actions
- Avoid complex solutions
- Use simple language
- Be direct and clear
- For income level, suggest how to better serve the local demographic
- DO NOT compare income levels between merchants
- Focus on how to better serve the local market
- Focus on transaction value, customer count, and refund rates
- For high avg transaction value, suggest how to leverage it
- For low avg transaction value, suggest how to increase it
- For high refund rate, suggest how to reduce it
- For low daily transactions, suggest how to increase footfall
- Always give specific expected percentage change in impact
- IMPORTANT: Keep the emoji at the start of each line
- IMPORTANT: Keep the format exactly as shown in the examples
- IMPORTANT: Add a blank line between different insights
- IMPORTANT: Keep each insight group together without extra lines

Use these emojis:
💪 for strengths
🎯 for opportunities
💡 for actions
📈 for impact
💰 for costs
✨ for wins
🏆 for achievements
⚠️ for warnings

Make it super concise and actionable and personalized for each merchant's demographic and industry!"""

    try:
        # Call Gemini API
        raw_response = call_gemini_api(prompt)
        print(raw_response)
        response = reformat_insights_multiline(raw_response)
        
        # Generate impact visualization data with the extracted insights
        impact_data = generate_impact_visualization(merchant_row, comparison_local, comparison_cluster, response)
        
        return response, impact_data
    except Exception as e:
        return f"Error generating insights: {str(e)}", None

def format_impact_visualization(impact_data):
    """Format impact visualization data for display."""
    if not impact_data:
        return ""
        
    formatted = "### Expected Impact of Actions\n\n"
    for data in impact_data:
        metric = data['metric']
        current = data['current']
        expected = data['expected']
        change_pct = ((expected - current) / current) * 100
        
        formatted += f"**{metric}:**\n"
        formatted += f"- Current: {current:.2f}\n"
        formatted += f"- Expected: {expected:.2f}\n"
        formatted += f"- Change: {change_pct:+.1f}%\n\n"
    
    return formatted

def generate_quick_insights(merchant_row, comparison_local, comparison_cluster, cluster_peers, cluster_averages):
    """Generate quick, number-focused insights for immediate merchant value"""
    
    prompt = f"""As a payment analytics expert, provide a concise, number-focused analysis for this merchant:

**Key Metrics:**
- Merchant ID: {merchant_row.get('merchant_id', 'N/A')}
- Industry: {merchant_row.get('industry', 'N/A')}
- Location: {merchant_row.get('city', 'N/A')}

**Current Performance:**
- Average Transaction Value: ₹{merchant_row.get('avg_txn_value', 'N/A'):.2f}
- Daily Transactions: {merchant_row.get('daily_txn_count', 'N/A')}
- Refund Rate: {merchant_row.get('refund_rate', 'N/A')*100:.1f}%

**Local Comparison:**
{format_quick_comparison(comparison_local, 'Local')}

**Cluster Comparison:**
{format_quick_comparison(comparison_cluster, 'Cluster')}

**Request:**
Provide a concise analysis with:
1. Top 3 key metrics that need attention (with specific numbers)
2. One immediate action item for each metric
3. One key strength (with numbers)
4. One quick win opportunity (with numbers)

Format the response as:
- Use emojis for visual appeal
- Keep each point to one line
- Include specific numbers in each point
- Make it scannable and actionable
- Focus on the most impactful metrics only"""

    return call_gemini_api(prompt)

def generate_advanced_ai_insights(merchant_row, comparison_local, comparison_cluster, cluster_peers, cluster_averages):
    """Generate detailed analysis when requested"""
    
    prompt = f"""As a payment analytics expert, provide a detailed analysis for this merchant:

**Merchant Profile:**
- Merchant ID: {merchant_row.get('merchant_id', 'N/A')}
- Industry: {merchant_row.get('industry', 'N/A')}
- Location: {merchant_row.get('city', 'N/A')}

**Detailed Metrics:**
- Average Transaction Value: ₹{merchant_row.get('avg_txn_value', 'N/A'):.2f}
- Daily Transactions: {merchant_row.get('daily_txn_count', 'N/A')}
- Refund Rate: {merchant_row.get('refund_rate', 'N/A')*100:.1f}%

**Local Market Analysis:**
{format_detailed_comparison(comparison_local, 'Local')}

**Cluster Analysis:**
{format_detailed_comparison(comparison_cluster, 'Cluster')}

**Request:**
Provide a comprehensive analysis including:
1. Detailed performance analysis with specific numbers
2. Root cause analysis for any issues
3. Strategic recommendations with implementation steps
4. Market positioning analysis
5. Risk assessment
6. Growth opportunities

Structure the response with clear sections and subsections.
Include specific numbers and percentages throughout.
Provide detailed, actionable recommendations."""

    return call_gemini_api(prompt)

def format_quick_comparison(comparison_df, comp_type):
    """Format comparison data for quick insights"""
    if comparison_df is None or comparison_df.empty:
        return f"No {comp_type} comparison data available."
    
    formatted = f"**{comp_type} Average Comparison:**\n"
    for _, row in comparison_df.iterrows():
        if pd.notna(row['Performance']):
            formatted += f"- {row['Metric']}: {row['Merchant Value']:.2f} vs {row[f'{comp_type} Avg']:.2f}\n"
    return formatted

def format_detailed_comparison(comparison_df, comp_type):
    """Format comparison data for detailed analysis"""
    if comparison_df is None or comparison_df.empty:
        return f"No {comp_type} comparison data available."
    
    formatted = f"**{comp_type} Market Analysis:**\n"
    for _, row in comparison_df.iterrows():
        if pd.notna(row['Performance']):
            formatted += f"- {row['Metric']}:\n"
            formatted += f"  * Current: {row['Merchant Value']:.2f}\n"
            formatted += f"  * {comp_type} Average: {row[f'{comp_type} Avg']:.2f}\n"
            formatted += f"  * Performance: {row['Performance']}\n"
            formatted += f"  * Difference: {((row['Merchant Value'] - row[f'{comp_type} Avg']) / row[f'{comp_type} Avg'] * 100):.1f}%\n"
    return formatted

def extract_percentage_from_insight(insight_text):
    """Extracts the first percentage value from the AI insight text."""
    if not insight_text:
        return None
    match = re.search(r'(\d+)%', insight_text)
    if match:
        return float(match.group(1)) / 100
    return None

def parse_insights_for_impact(response):
    """Parse AI response to extract impact percentages for each metric."""
    insights_dict = {}
    current_metric = None
    
    for line in response.split('\n'):
        line = line.strip()
        
        if '💪 STRENGTH:' in line or '🎯 OPPORTUNITY:' in line:
            if 'value' in line.lower():
                current_metric = 'Avg Txn Value'
            elif 'count' in line.lower():
                current_metric = 'Daily Txn Count'
            elif 'rate' in line.lower():
                current_metric = 'Refund Rate'
            elif 'level' in line.lower():
                current_metric = 'Income Level'
        
        if '📈 IMPACT:' in line and current_metric:
            match = re.search(r'(\d+)%', line)
            if match:
                # Store the full impact line
                insights_dict[current_metric] = line.strip()
    
    return insights_dict


def reformat_insights_multiline(insight_text):
    """
    Reformat single-line Gemini insights into multiline blocks with spacing.
    """
    lines = insight_text.split('\n')
    formatted = []
    block = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(emoji in line for emoji in ['💪', '🎯', '⚠️']):
            if block:
                formatted.append('\n'.join(block) + '\n')
                block = []
        block.append(line)

    if block:
        formatted.append('\n'.join(block) + '\n')

    return '\n'.join(formatted)

def format_insights_for_display(insights_text, impact_data=None):
    """Format insights with proper spacing and line breaks for Streamlit display."""
    # Split into individual insights
    insights = insights_text.strip().split('\n\n')
    formatted_insights = []
    
    for i, insight in enumerate(insights):
        if not insight.strip():
            continue
            
        # Split the insight into lines
        lines = insight.strip().split('\n')
        if len(lines) >= 3:  # Ensure we have all three parts
            # Get the metric from the insight
            metric = None
            insight_text = lines[0].lower()
            
            # More comprehensive metric detection with unique identifiers
            if 'transaction value' in insight_text or 'avg' in insight_text or 'value' in insight_text:
                metric = 'Avg Txn Value'
            elif 'daily count' in insight_text:
                metric = 'Daily Txn Count (Count)'
            elif 'daily txns' in insight_text:
                metric = 'Daily Txn Count (Txns)'
            elif 'daily' in insight_text or 'count' in insight_text or 'transaction' in insight_text or 'customers' in insight_text:
                metric = 'Daily Txn Count'
            elif 'refund' in insight_text or 'rate' in insight_text:
                metric = 'Refund Rate'
            elif 'income' in insight_text or 'level' in insight_text:
                metric = 'Income Level'
            
            # Only add to formatted insights if we found a metric
            if metric:
                formatted_insight = f"""
                <div style='margin: 10px 0; padding: 15px; border-radius: 8px; border: 1px solid #4dabf7; background: transparent;'>
                    <p style='margin: 5px 0; color: #e0e0e0;'>{lines[0]}</p>
                    <p style='margin: 5px 0; color: #e0e0e0;'>{lines[1]}</p>
                    <p style='margin: 5px 0; color: #e0e0e0;'>{lines[2]}</p>
                </div>
                """
                formatted_insights.append((formatted_insight, metric, i))  # Add index for unique key
    
    return formatted_insights
