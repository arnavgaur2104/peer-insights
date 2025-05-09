import pandas as pd
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

def generate_insights(comparison_df):
    """Generate simple rule-based suggestions based on the performance comparison."""
    insights = []
    if comparison_df is None or comparison_df.empty:
        print("Warning: generate_insights called with None or empty comparison_df.")
        return ["Comparison data is missing or empty."]

    expected_cols = ['Metric', 'Merchant Value', 'Performance']
    if len(comparison_df.columns) > 2:
        avg_col_name = comparison_df.columns[2]
        expected_cols.append(avg_col_name)
    else:
        print("Warning: Comparison DataFrame has fewer than 3 columns in generate_insights.")
        avg_col_name = None

    if not all(col in comparison_df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in comparison_df.columns]
        print(f"Warning: Missing expected columns in comparison_df for generate_insights: {missing}")
        return [f"Comparison data is incomplete (missing: {missing})."]

    for idx, row in comparison_df.iterrows():
        try:
            metric_label = row['Metric']
            merchant_value = row['Merchant Value']
            competitor_avg = row.get(avg_col_name, 'N/A') if avg_col_name else 'N/A'
            performance = row['Performance']

            if pd.isna(performance) or '✅' in str(performance) or 'N/A' in str(performance):
                continue

            metric_key = metric_label.lower().replace(' ', '_')
            comp_avg_str = 'N/A'
            if pd.notna(competitor_avg) and isinstance(competitor_avg, (int, float)):
                if metric_key in ['refund_rate', 'rent_pct_revenue']:
                    comp_avg_str = f"{competitor_avg*100:.1f}%"
                else:
                    comp_avg_str = f"{competitor_avg:.2f}"

            if metric_key == 'avg_txn_value':
                insights.append(f"💡 **Avg Transaction Value:** Consider bundling products or offering small upsells to increase value (Competitor avg: {comp_avg_str}).")
            elif metric_key == 'daily_txn_count':
                insights.append(f"💡 **Daily Transactions:** Explore promotions, loyalty programs, or better signage to increase customer visits (Competitor avg: {comp_avg_str}).")
            elif metric_key == 'refund_rate':
                insights.append(f"⚠️ **Refund Rate:** High refund rate detected. Review product quality or return policy clarity (Competitor avg: {comp_avg_str}).")
            elif metric_key == 'rent_pct_revenue':
                insights.append(f"⚠️ **Rent Cost:** High rent percentage detected. Consider operational efficiencies or renegotiating rent (Competitor avg: {comp_avg_str} of revenue).")

        except Exception as e:
            print(f"Error processing row {idx} in generate_insights: {e}")
            continue

    if not insights:
        insights.append("✅ Great! No major performance issues detected compared to local competitors based on basic rules.")
    return insights

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

Format the response EXACTLY as follows:
- Use emojis for visual appeal
- Include specific numbers in each point
- Make it scannable and actionable
- Focus on the most impactful metrics only
- ALWAYS format each issue and action pair on its own line
- Use proper line breaks between sections
- Use markdown formatting for better readability

Example format:

**Key Issues & Actions:**
🔍 Low avg transaction value (₹500 vs ₹800) → 💡 Bundle products to increase ticket size

🔍 High refund rate (5% vs 2%) → 💡 Review product quality and return policy

🔍 Low daily transactions (20 vs 35) → 💡 Launch loyalty program

**Strengths:**
✅ Key Strength: Excellent rent efficiency (15% vs 25% of revenue)

**Quick Win:**
🚀 Implement upselling (potential +₹200 per transaction)"""

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
