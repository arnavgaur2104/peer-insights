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
    if metric in ['avg_txn_value', 'daily_txn_count', 'income_level']:
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
    
    # Initialize lists for different types of insights
    critical_issues = []
    opportunities = []
    quick_wins = []
    
    # Process local comparisons
    if comparison_local is not None and not comparison_local.empty:
        for _, row in comparison_local.iterrows():
            metric = row['Metric'].lower().replace(' ', '_')
            merchant_value = row['Merchant Value']
            local_avg = row['Local Avg']
            
            if pd.isna(merchant_value) or pd.isna(local_avg):
                continue
            
            pct_diff, is_better = calculate_metric_impact(metric, merchant_value, local_avg)
            
            # Critical issues (significant negative impact)
            if not is_better and abs(pct_diff) > 10:
                if metric == 'avg_txn_value':
                    critical_issues.append({
                        'title': 'Low Transaction Value',
                        'comparison': format_metric_comparison(metric, merchant_value, local_avg, pct_diff, is_better),
                        'action': f"Increase by ₹{abs(merchant_value - local_avg):.2f} through bundling",
                        'impact': f"Potential revenue increase: ₹{abs(merchant_value - local_avg) * merchant_row.get('daily_txn_count', 0):.2f} daily"
                    })
                elif metric == 'daily_txn_count':
                    critical_issues.append({
                        'title': 'Low Customer Footfall',
                        'comparison': format_metric_comparison(metric, merchant_value, local_avg, pct_diff, is_better),
                        'action': f"Add {abs(merchant_value - local_avg):.0f} customers through promotions",
                        'impact': f"Potential revenue increase: ₹{abs(merchant_value - local_avg) * merchant_row.get('avg_txn_value', 0):.2f} daily"
                    })
                elif metric == 'refund_rate':
                    critical_issues.append({
                        'title': 'High Refund Rate',
                        'comparison': format_metric_comparison(metric, merchant_value, local_avg, pct_diff, is_better),
                        'action': f"Reduce by {abs(merchant_value - local_avg)*100:.1f}% through quality control",
                        'impact': f"Potential savings: ₹{abs(merchant_value - local_avg) * merchant_row.get('avg_txn_value', 0) * merchant_row.get('daily_txn_count', 0):.2f} daily"
                    })
            
            # Opportunities (significant positive impact)
            elif is_better and abs(pct_diff) > 10:
                if metric == 'avg_txn_value':
                    opportunities.append({
                        'title': 'Strong Transaction Value',
                        'comparison': format_metric_comparison(metric, merchant_value, local_avg, pct_diff, is_better),
                        'action': "Leverage to cross-sell premium products",
                        'impact': f"Current advantage: ₹{abs(merchant_value - local_avg):.2f} per transaction"
                    })
                elif metric == 'daily_txn_count':
                    opportunities.append({
                        'title': 'High Customer Footfall',
                        'comparison': format_metric_comparison(metric, merchant_value, local_avg, pct_diff, is_better),
                        'action': "Increase basket size through upselling",
                        'impact': f"Potential increase: ₹{merchant_value * merchant_row.get('avg_txn_value', 0) * 0.1:.2f} daily"
                    })
    
    # Generate insights text
    insights = []
    
    # Add merchant context
    insights.append(f"## 📊 Performance Analysis for {merchant_row.get('industry', 'Business')}")
    insights.append(f"Location: {merchant_row.get('city', 'N/A')} | Store Type: {merchant_row.get('store_type', 'N/A')}\n")
    
    # Add critical issues
    if critical_issues:
        insights.append("### 🔍 Critical Issues")
        for issue in critical_issues[:3]:  # Top 3 issues
            insights.append(f"**{issue['title']}**")
            insights.append(f"- Current vs Average: {issue['comparison']}")
            insights.append(f"- Action: {issue['action']}")
            insights.append(f"- Impact: {issue['impact']}\n")
    else:
        insights.append("### ✅ No Critical Issues Detected\n")
    
    # Add opportunities
    if opportunities:
        insights.append("### 💡 Growth Opportunities")
        for opp in opportunities[:2]:  # Top 2 opportunities
            insights.append(f"**{opp['title']}**")
            insights.append(f"- Current vs Average: {opp['comparison']}")
            insights.append(f"- Action: {opp['action']}")
            insights.append(f"- Impact: {opp['impact']}\n")
    
    # Add quick wins
    insights.append("### 🚀 Quick Wins")
    
    # Quick win 1: Upselling opportunity
    if merchant_row.get('avg_txn_value', 0) < merchant_row.get('income_level', 0) * 0.1:
        target_value = merchant_row.get('income_level', 0) * 0.1
        current_value = merchant_row.get('avg_txn_value', 0)
        insights.append("1. **Upselling Opportunity**")
        insights.append(f"   - Current: ₹{current_value:.2f}")
        insights.append(f"   - Target: ₹{target_value:.2f}")
        insights.append(f"   - Action: Train staff on premium product features")
        insights.append(f"   - Expected Impact: +₹{target_value - current_value:.2f} per transaction\n")
    
    # Quick win 2: Footfall boost
    if merchant_row.get('daily_txn_count', 0) < 50:
        current_count = merchant_row.get('daily_txn_count', 0)
        target_count = int(current_count * 1.2)  # 20% increase
        insights.append("2. **Footfall Boost**")
        insights.append(f"   - Current: {current_count} customers")
        insights.append(f"   - Target: {target_count} customers")
        insights.append("   - Action: Launch 2-week promotion with 10% discount")
        insights.append(f"   - Expected Impact: +{target_count - current_count} customers daily\n")
    
    # Add cluster insights
    if cluster_peers is not None and not cluster_peers.empty:
        insights.append("### 🔄 Cluster Insights")
        top_performers = cluster_peers.nlargest(3, 'avg_txn_value')
        if not top_performers.empty:
            insights.append("**Top Performers in Your Cluster:**")
            for _, peer in top_performers.iterrows():
                insights.append(f"- {peer['store_type']} in {peer['city']}: ₹{peer['avg_txn_value']:.2f} avg. transaction")
    
    return "\n".join(insights)

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
