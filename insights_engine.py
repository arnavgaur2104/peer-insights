import pandas as pd
import numpy as np
import random
import traceback
import re
import streamlit as st
import google.generativeai as genai
import os

# --- Configure Gemini API ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def call_gemini_api(prompt, model_name="gemini-1.5-flash"):
    """Calls the configured Google Generative AI model (Gemini)."""
    
    # Configure API fresh each time
    try:
        # Try environment variable first
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            # Fallback to Streamlit secrets if available
            try:
                if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
        
        if not api_key or not api_key.strip():
            return "### AI Analysis Error:\nGoogle API Key not configured. Please set GOOGLE_API_KEY environment variable."
        
        # Configure the API
        genai.configure(api_key=api_key.strip())
        
        # Create model and generate response
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if response.parts:
            return response.text
        else:
            return "### AI Analysis Error:\nNo response generated from the AI model."
            
    except Exception as e:
        return f"### AI Analysis Error:\nAn error occurred: {str(e)}"

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

def generate_impact_visualization(merchant_row, comparison_local, comparison_cluster, response=None):
    """Generate data for impact visualization."""
    print(f"\n\n=== IMPACT VISUALIZATION DEBUG START ===")
    print(f"DEBUG: generate_impact_visualization called")
    print(f"DEBUG: response=\n{response}")
    print(f"DEBUG: response type: {type(response)}")
    print(f"DEBUG: comparison_cluster is None: {comparison_cluster is None}")
    print(f"=== IMPACT VISUALIZATION DEBUG START ===\n")
    
    if comparison_cluster is None:
        print(f"DEBUG: comparison_cluster is None, returning None")
        return None

    impact_data = []
    
    # Split insights into individual blocks using the same logic as format_insights_for_display
    raw_insights = response.strip().split('\n\n')
    print(f"DEBUG: Found {len(raw_insights)} raw insights after splitting")
    
    for i, raw_insight in enumerate(raw_insights):
        print(f"DEBUG: Raw insight {i}: '{raw_insight}'")
    
    # Filter insights the same way as format_insights_for_display
    cleaned_insights = []
    for insight in raw_insights:
        insight = insight.strip()
        if not insight:  # Skip empty
            continue
        if len(insight) <= 3 and insight in ['💪', '🎯', '⚠️']:  # Skip standalone emojis
            continue
        if len(insight.split('\n')) >= 2:  # Only include multi-line insights
            cleaned_insights.append(insight)
    
    print(f"DEBUG: After filtering, {len(cleaned_insights)} cleaned insights")
    for i, insight in enumerate(cleaned_insights):
        print(f"DEBUG: Cleaned insight {i}: '{insight}'")
    
    # Process each cleaned insight
    for insight_index, insight_block in enumerate(cleaned_insights):
        print(f"DEBUG: Processing insight {insight_index}")
        lines = insight_block.strip().split('\n')
        if len(lines) < 2:  # Need at least 2 lines
            print(f"DEBUG: Skipping insight {insight_index} - not enough lines")
            continue
            
        # Extract metric and impact percentage
        metric = None
        impact_pct = None
        
        # Detect which metric this insight is about
        insight_text = insight_block.lower()
        print(f"DEBUG: Full insight_text for metric detection: '{insight_text}'")
        
        if 'avg' in insight_text and ('transaction' in insight_text or 'txn' in insight_text):
            metric = 'Avg Txn Value'
            print(f"DEBUG: Matched Avg Txn Value - found 'avg' and 'transaction/txn'")
        elif 'daily' in insight_text and ('count' in insight_text or 'txn' in insight_text or 'transaction' in insight_text):
            metric = 'Daily Txn Count'
            print(f"DEBUG: Matched Daily Txn Count - found 'daily' and 'count/txn/transaction'")
        elif 'repeat' in insight_text and ('customer' in insight_text or 'rate' in insight_text):
            metric = 'Repeat Customer Rate'
            print(f"DEBUG: Matched Repeat Customer Rate - found 'repeat' and 'customer/rate'")
        elif 'refund' in insight_text and 'rate' in insight_text:
            metric = 'Refund Rate'
            print(f"DEBUG: Matched Refund Rate - found 'refund' and 'rate'")
        else:
            print(f"DEBUG: No metric match found")
            print(f"DEBUG: Contains 'avg': {'avg' in insight_text}")
            print(f"DEBUG: Contains 'transaction': {'transaction' in insight_text}")
            print(f"DEBUG: Contains 'txn': {'txn' in insight_text}")
            print(f"DEBUG: Contains 'daily': {'daily' in insight_text}")
            print(f"DEBUG: Contains 'count': {'count' in insight_text}")
            print(f"DEBUG: Contains 'repeat': {'repeat' in insight_text}")
            print(f"DEBUG: Contains 'customer': {'customer' in insight_text}")
            print(f"DEBUG: Contains 'rate': {'rate' in insight_text}")
        
        print(f"DEBUG: Detected metric for insight {insight_index}: {metric}")
        
        if not metric:
            print(f"DEBUG: No metric detected for insight {insight_index}")
            continue
        
        # Extract impact percentage from any line that contains 'IMPACT:'
        for line in lines:
            if '📈 IMPACT:' in line:
                print(f"DEBUG: Found impact line: {line}")
                # Extract percentage using regex
                import re
                percentage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                if percentage_match:
                    impact_pct = float(percentage_match.group(1)) / 100.0
                    print(f"DEBUG: Extracted impact percentage: {impact_pct}")
                break
        
        if impact_pct is None:
            print(f"DEBUG: No impact percentage found for insight {insight_index}")
            continue
        
        # Find corresponding metric in cluster comparison data
        metric_row = comparison_cluster[comparison_cluster['Metric'] == metric]
        if metric_row.empty:
            print(f"DEBUG: No metric row found for {metric} in comparison_cluster")
            continue
        
        metric_data = metric_row.iloc[0]
        current_value = metric_data['Merchant Raw']
        cluster_avg = metric_data['Cluster Raw']
        
        print(f"DEBUG: Found metric data - current: {current_value}, cluster_avg: {cluster_avg}")
        
        # Calculate expected value after improvement
        if metric == 'Refund Rate':
            # For refund rate, improvement means reduction
            expected_value = current_value * (1 - impact_pct)
        else:
            # For other metrics, improvement means increase
            expected_value = current_value * (1 + impact_pct)
        
        print(f"DEBUG: Calculated expected value: {expected_value}")
        
        impact_data.append({
            'metric': metric,
            'insight_index': insight_index,
            'current': current_value,
            'expected': expected_value,
            'cluster_avg': cluster_avg,
            'impact_pct': impact_pct
        })
        
        print(f"DEBUG: Added impact data for {metric}")
    
    print(f"DEBUG: Final impact_data has {len(impact_data)} items")
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

    # Format performance metrics with better aesthetics (use cluster data only)
    performance_metrics = []
    comparison_data = comparison_cluster if comparison_cluster is not None else comparison_local
    
    if comparison_data is not None:
        for _, row in comparison_data.iterrows():
            metric = row['Metric']
            if 'income' in metric.lower():
                continue
                
            merchant_value = row['Merchant Value']  # Display value
            
            # Use cluster average if available, otherwise skip
            if 'Cluster Avg' in row:
                cluster_avg = row['Cluster Avg']  # Display value
                performance = row['Performance']
                
                # Use raw values for gap calculation
                merchant_raw = row.get('Merchant Raw', 0)
                cluster_raw = row.get('Cluster Raw', 0)
                gap = ((merchant_raw - cluster_raw) / cluster_raw) * 100 if cluster_raw != 0 else 0
                
                # Add context about whether the metric is good or bad
                metric_context = ""
                if metric == 'Avg Txn Value':
                    if gap > 0:
                        metric_context = " (Good: Higher than similar businesses)"
                    else:
                        metric_context = " (Issue: Lower than similar businesses)"
                elif metric == 'Daily Txn Count':
                    if gap > 0:
                        metric_context = " (Good: Higher than similar businesses)"
                    else:
                        metric_context = " (Issue: Lower than similar businesses)"
                elif metric == 'Repeat Customer Rate':
                    if gap > 0:
                        metric_context = " (Good: Higher than similar businesses)"
                    else:
                        metric_context = " (Issue: Lower than similar businesses)"
                elif metric == 'Refund Rate':
                    if gap < 0:
                        metric_context = " (Good: Lower than similar businesses)"
                    else:
                        metric_context = " (Issue: Higher than similar businesses)"
                
                performance_metrics.append(f"""
                📊 {metric}{metric_context}
                ──────────────────────────────
                • Your Value: {merchant_value}
                • Similar Business Average: {cluster_avg}
                • Performance: {performance}
                • Gap: {gap:+.1f}%
                ──────────────────────────────
                """)

    # Create a concise prompt for Gemini with better formatting
    prompt = f"""As a retail business consultant, provide <= 3 key, actionable insights for this merchant. Keep each insight to 3 lines maximum:

{merchant_profile}

📈 Current Performance vs Similar Businesses:
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
- For high repeat customer rate, suggest how to leverage loyalty
- For low repeat customer rate, suggest how to improve retention
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
        print(f"DEBUG: About to call Gemini API")
        raw_response = call_gemini_api(prompt)
        print(f"DEBUG: Got raw_response from Gemini")
        response = reformat_insights_multiline(raw_response)
        print(f"DEBUG: Reformatted response")
        
        # Generate impact visualization data using cluster comparison
        print(f"DEBUG: About to call generate_impact_visualization")
        print(f"DEBUG: comparison_cluster is None: {comparison_cluster is None}")
        impact_data = generate_impact_visualization(merchant_row, comparison_local, comparison_cluster, response)
        print(f"DEBUG: generate_impact_visualization returned: {impact_data}")
        
        # Enhance insights with actionable links AFTER generating impact data
        print(f"DEBUG: About to enhance insights with links")
        enhanced_response = enhance_insights_with_links(
            response, 
            merchant_row.get('industry', ''), 
            merchant_row.get('store_type', '')
        )
        print(f"DEBUG: Enhanced insights completed")
        
        return enhanced_response, impact_data
    except Exception as e:
        print(f"DEBUG: Exception in generate_crisp_insights: {e}")
        print(f"DEBUG: Exception traceback: {traceback.format_exc()}")
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
            # Use the formatted display values directly
            formatted += f"- {row['Metric']}: {row['Merchant Value']} vs {row[f'{comp_type} Avg']}\n"
    return formatted

def format_detailed_comparison(comparison_df, comp_type):
    """Format comparison data for detailed analysis"""
    if comparison_df is None or comparison_df.empty:
        return f"No {comp_type} comparison data available."
    
    formatted = f"**{comp_type} Market Analysis:**\n"
    for _, row in comparison_df.iterrows():
        if pd.notna(row['Performance']):
            # Calculate percentage difference using raw values
            merchant_raw = row.get('Merchant Raw', 0)
            avg_raw = row.get(f'{comp_type} Raw', 0)
            
            if avg_raw != 0:
                diff_pct = ((merchant_raw - avg_raw) / avg_raw * 100)
            else:
                diff_pct = 0
                
            formatted += f"- {row['Metric']}:\n"
            formatted += f"  * Current: {row['Merchant Value']}\n"
            formatted += f"  * {comp_type} Average: {row[f'{comp_type} Avg']}\n"
            formatted += f"  * Performance: {row['Performance']}\n"
            formatted += f"  * Difference: {diff_pct:.1f}%\n"
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
    if not insights_text:
        return []
    
    # Split insights using double newlines first, then clean up
    raw_insights = insights_text.split('\n\n')
    
    # Filter and clean insights
    cleaned_insights = []
    for insight in raw_insights:
        insight = insight.strip()
        if not insight:  # Skip empty
            continue
        if len(insight) <= 3 and insight in ['💪', '🎯', '⚠️']:  # Skip standalone emojis
            continue
        if len(insight.split('\n')) >= 2:  # Only include multi-line insights
            cleaned_insights.append(insight)
    
    # Format each insight for HTML display with proper styling
    formatted_insights = []
    
    for insight_index, insight in enumerate(cleaned_insights):
        # Split insight into lines for processing
        lines = insight.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Check if line contains HTML tags (links)
            if '<a href=' in line or '<span style=' in line:
                # Preserve HTML lines as-is
                formatted_lines.append(line)
            else:
                # Wrap regular text lines in paragraph tags
                formatted_lines.append(f"<p style='margin: 5px 0; color: #e0e0e0;'>{line}</p>")
        
        # Join all lines and wrap in styled container
        insight_content = '\n'.join(formatted_lines)
        
        # Wrap the entire insight in a styled container
        formatted_insight = f"""
        <div style='margin: 10px 0; padding: 15px; border-radius: 8px; border: 1px solid #4dabf7; background: transparent;'>
            {insight_content}
        </div>
        """
        
        # Extract metric and insight index for impact data matching
        metric = None
        insight_lower = insight.lower()
        print(f"DEBUG: format_insights_for_display - Processing insight {insight_index}: '{insight[:50]}...'")
        print(f"DEBUG: insight_lower for metric detection: '{insight_lower[:100]}...'")
        
        if 'avg' in insight_lower and ('transaction' in insight_lower or 'txn' in insight_lower):
            metric = 'Avg Txn Value'
            print(f"DEBUG: Matched Avg Txn Value")
        elif 'daily' in insight_lower and ('transaction' in insight_lower or 'txn' in insight_lower or 'count' in insight_lower):
            metric = 'Daily Txn Count'
            print(f"DEBUG: Matched Daily Txn Count")
        elif 'repeat' in insight_lower and ('customer' in insight_lower or 'rate' in insight_lower):
            metric = 'Repeat Customer Rate'
            print(f"DEBUG: Matched Repeat Customer Rate")
        elif 'refund' in insight_lower and 'rate' in insight_lower:
            metric = 'Refund Rate'
            print(f"DEBUG: Matched Refund Rate")
        else:
            print(f"DEBUG: No metric matched for insight")
            
        print(f"DEBUG: Final metric for insight {insight_index}: {metric}")
        
        formatted_insights.append((formatted_insight, metric, insight_index))
    
    return formatted_insights

def get_link_relevance_explanation(action_text, links_provided, industry, store_type):
    """
    Explain why the provided links are relevant to the specific action.
    """
    action_lower = action_text.lower()
    explanations = []
    
    # Explain relevance based on action type
    if any(keyword in action_lower for keyword in ['pos', 'payment', 'digital', 'qr']):
        explanations.append("💳 These payment solutions help streamline transactions and reduce checkout friction")
    
    if any(keyword in action_lower for keyword in ['online', 'delivery', 'ordering']):
        explanations.append("🚀 Online platforms expand your reach beyond physical footfall")
        if industry == 'Restaurant':
            explanations.append("🍽️ Food delivery platforms can increase daily transaction count by 40-60%")
    
    if any(keyword in action_lower for keyword in ['marketing', 'promote', 'advertise']):
        explanations.append("📈 Digital marketing tools help attract new customers and increase visibility")
        explanations.append("🎯 Google My Business alone can increase foot traffic by 20-30%")
    
    if any(keyword in action_lower for keyword in ['loyalty', 'reward', 'retention']):
        explanations.append("🔄 Customer retention tools can increase repeat purchase rate by 25-50%")
        explanations.append("💰 Acquiring new customers costs 5x more than retaining existing ones")
    
    if any(keyword in action_lower for keyword in ['combo', 'bundle', 'package']):
        explanations.append("📦 Bundling strategies can increase average transaction value by 15-25%")
    
    if any(keyword in action_lower for keyword in ['inventory', 'analytics', 'system']):
        explanations.append("📊 Business analytics help identify patterns and optimize operations")
    
    if any(keyword in action_lower for keyword in ['credit', 'loan', 'capital']):
        explanations.append("💵 Access to capital helps implement growth strategies faster")
        explanations.append("⚡ Razorpay Capital offers instant business loans based on your transaction history")
    
    # Store type specific relevance
    if store_type == 'Mall':
        explanations.append("🏬 Mall-specific tools help leverage high footfall and premium positioning")
    elif store_type == 'Street Front':
        explanations.append("🛣️ Local business tools help establish neighborhood presence and community relationships")
    elif store_type == 'Standalone':
        explanations.append("🏪 Independent business tools help compete with chains through digital presence")
    
    return explanations

def get_actionable_links(action_text, industry, store_type):
    """
    Map specific action items to directly relevant business links.
    Returns enhanced action text with embedded links that directly support the suggested action.
    """
    print(f"DEBUG: get_actionable_links called with action_text='{action_text}', industry={industry}, store_type={store_type}")
    
    action_lower = action_text.lower()
    print(f"DEBUG: action_lower='{action_lower}'")
    
    # Direct action-to-solution mapping
    action_links = {}
    
    # COMBO MEALS & BUNDLING ACTIONS
    if any(phrase in action_lower for phrase in ['combo', 'bundle', 'package', 'meal', 'deal', 'offer']):
        print(f"DEBUG: Matched combo/bundling action")
        if industry == 'Restaurant':
            action_links['Menu Design Tools'] = {
                'url': 'https://www.canva.com/create/menus/',
                'benefit': 'Create attractive combo menu displays that make customers order 30% more combo meals'
            }
            action_links['Food Photography'] = {
                'url': 'https://www.zomato.com/business/photo-shoot',
                'benefit': 'Professional combo photos increase online orders by 60% - customers order what looks appetizing'
            }
            action_links['POS Systems'] = {
                'url': 'https://www.petpooja.com/',
                'benefit': 'Auto-suggest combos at checkout - staff can upsell 40% more effectively'
            }
        elif industry == 'Retail':
            action_links['Product Bundling Guide'] = {
                'url': 'https://blog.shopify.com/product-bundling',
                'benefit': 'Learn to create "Buy 2 Get 1 Free" deals that increase basket size by 35%'
            }
            action_links['POS Bundle Setup'] = {
                'url': 'https://razorpay.com/pos/',
                'benefit': 'Set up automatic bundle pricing - "3 soaps for ₹150" instead of individual pricing'
            }
            action_links['Inventory Management'] = {
                'url': 'https://www.zoho.com/inventory/',
                'benefit': 'Track which product combinations sell best to optimize your deals'
            }
        elif industry == 'Fashion':
            action_links['Outfit Styling'] = {
                'url': 'https://www.canva.com/create/fashion-lookbooks/',
                'benefit': 'Create "Complete Look" displays - shirt + pants + accessories for 40% higher sales'
            }
            action_links['Fashion Bundles'] = {
                'url': 'https://www.shopify.com/blog/fashion-ecommerce',
                'benefit': 'Learn to bundle "Formal Office Wear" or "Weekend Casual" sets effectively'
            }
    
    # PROMOTION & DISCOUNT ACTIONS (including happy hour)
    if any(phrase in action_lower for phrase in ['happy hour', 'discount', 'sale', 'promotion', 'special', 'offer']):
        print(f"DEBUG: Matched promotion action")
        if industry == 'Restaurant':
            action_links['Happy Hour Setup'] = {
                'url': 'https://pos.toasttab.com/blog/restaurant-happy-hour',
                'benefit': 'Design happy hour menus that bring 50% more customers during slow hours'
            }
            action_links['Social Media Promotion'] = {
                'url': 'https://business.facebook.com/',
                'benefit': 'Post "Happy Hour 4-7 PM: 30% off appetizers!" to bring immediate crowds'
            }
        elif industry == 'Retail':
            action_links['Flash Sale Tools'] = {
                'url': 'https://www.shopify.com/blog/flash-sales',
                'benefit': 'Run "2-hour flash sales" that create urgency and boost daily sales by 60%'
            }
            action_links['WhatsApp Promotions'] = {
                'url': 'https://business.whatsapp.com/',
                'benefit': 'Send "Today only: 20% off cleaning supplies!" to your customer list'
            }
        elif industry == 'Fashion':
            action_links['Fashion Sales Strategy'] = {
                'url': 'https://www.shopify.com/retail/fashion-retail-strategies',
                'benefit': 'Learn to run "End of Season" sales that clear inventory while maintaining margins'
            }
    
    # LOYALTY PROGRAM ACTIONS
    if any(phrase in action_lower for phrase in ['loyalty', 'repeat', 'retention', 'customer', 'return']):
        print(f"DEBUG: Matched loyalty action")
        if industry == 'Restaurant':
            action_links['WhatsApp Business'] = {
                'url': 'https://business.whatsapp.com/',
                'benefit': 'Send "Miss you!" messages to inactive customers - brings back 30% within a week'
            }
            action_links['Customer Database'] = {
                'url': 'https://www.zoho.com/crm/',
                'benefit': 'Remember customer preferences so you can send personalized offers like "Your favorite kurta style is back in stock!"'
            }
            action_links['Restaurant Loyalty'] = {
                'url': 'https://www.toast.com/products/loyalty',
                'benefit': 'Track "Buy 5 meals, get 6th free" automatically - customers visit 40% more often'
            }
        elif industry == 'Fashion':
            action_links['WhatsApp Business'] = {
                'url': 'https://business.whatsapp.com/',
                'benefit': 'Send personalized style updates - "Your size in the dress you liked is back!" doubles return visits'
            }
            action_links['Customer Database'] = {
                'url': 'https://www.zoho.com/crm/',
                'benefit': 'Track style preferences - "Customers who bought kurtas also love these dupattas" increases sales by 35%'
            }
            action_links['Fashion Loyalty'] = {
                'url': 'https://www.limeroad.com/business',
                'benefit': 'Points for every purchase - "You are 200 points away from 20% off!" keeps customers engaged'
            }
            action_links['Style Rewards'] = {
                'url': 'https://razorpay.com/magic-checkout/',
                'benefit': 'Create smooth, fast checkout experience that makes loyal customers want to return'
            }
        elif industry == 'Retail':
            action_links['WhatsApp Business'] = {
                'url': 'https://business.whatsapp.com/',
                'benefit': 'Remind about expiring products - "Your usual cooking oil is running low?" increases frequency by 25%'
            }
            action_links['Customer Database'] = {
                'url': 'https://www.zoho.com/crm/',
                'benefit': 'Track buying patterns - "Monthly grocery reminder" brings customers back on schedule'
            }
            action_links['Retail Loyalty'] = {
                'url': 'https://www.loyverse.com/',
                'benefit': 'Automatic discounts for bulk buyers - "Buy 5 items, get 10% off" increases basket size by 45%'
            }
    
    # DELIVERY & ONLINE ACTIONS
    if any(phrase in action_lower for phrase in ['delivery', 'online', 'zomato', 'swiggy', 'digital']):
        print(f"DEBUG: Matched delivery action")
        if industry == 'Restaurant':
            action_links['Zomato Partner'] = {
                'url': 'https://www.zomato.com/business',
                'benefit': 'Get 50+ daily orders within first month - delivery customers order 20% more than walk-ins'
            }
            action_links['Swiggy Partner'] = {
                'url': 'https://partner.swiggy.com/',
                'benefit': 'Dual platform presence increases delivery orders by 80% - more hungry customers find you'
            }
            action_links['Payment Links'] = {
                'url': 'https://razorpay.com/payment-links/',
                'benefit': 'Send payment links for pre-orders - "Pay now, pick up in 20 mins" reduces wait time complaints'
            }
    
    # MARKETING & PROMOTION ACTIONS
    if any(phrase in action_lower for phrase in ['marketing', 'promotion', 'advertise', 'social', 'google']):
        print(f"DEBUG: Matched marketing action")
        action_links['Google My Business'] = {
            'url': 'https://business.google.com/',
            'benefit': 'Free Google listing brings 40% more walk-ins - customers search "restaurants near me" and find you first'
        }
        action_links['Facebook Business'] = {
            'url': 'https://business.facebook.com/',
            'benefit': 'Post daily specials to 500+ local followers - "Today only: Biryani + Raita for ₹199!" increases sales by 25%'
        }
        action_links['Instagram Business'] = {
            'url': 'https://business.instagram.com/',
            'benefit': 'Food photos get 60% more engagement - customers tag friends and bring groups, increasing table size'
        }
    
    # STAFF & TRAINING ACTIONS
    if any(phrase in action_lower for phrase in ['staff', 'training', 'service', 'upsell']):
        print(f"DEBUG: Matched staff training action")
        action_links['Staff Training Videos'] = {
            'url': 'https://www.youtube.com/playlist?list=PLrAXtmRdnEQdvF9OGAXZSGsN_bG6Mk8PD',
            'benefit': 'Trained staff suggest "Would you like fries with that?" - increases average order value by 15-20%'
        }
        action_links['Upselling Techniques'] = {
            'url': 'https://blog.hubspot.com/service/upselling-techniques',
            'benefit': 'Learn 5 phrases that double dessert sales - "Save room for our signature chocolate cake!"'
        }
    
    print(f"DEBUG: Found {len(action_links)} action links: {list(action_links.keys())}")
    
    # If no links found, return original text
    if not action_links:
        print(f"DEBUG: No links found, returning original text")
        return action_text
    
    # Create enhanced action text with embedded links as proper HTML list
    enhanced_text = action_text + "<br><span style='color: #e67700; font-weight: 600;'>📎 Quick Setup:</span><ul style='margin: 5px 0; padding-left: 20px;'>"
    
    for link_name, link_info in action_links.items():
        enhanced_text += f"<li style='margin: 3px 0;'><a href='{link_info['url']}' target='_blank' style='color: #4dabf7; text-decoration: none; font-weight: 600;'>{link_name}</a> <span style='color: #adb5bd; font-size: 0.85em; font-style: italic;'>({link_info['benefit']})</span></li>"
    
    enhanced_text += "</ul>"
    
    print(f"DEBUG: Enhanced text created: {enhanced_text}")
    return enhanced_text

def enhance_insights_with_links(insights_text, industry, store_type):
    """
    Process the insights text to add actionable links to action items.
    """
    print(f"DEBUG: enhance_insights_with_links called with industry={industry}, store_type={store_type}")
    print(f"DEBUG: insights_text=\n{insights_text}")
    
    lines = insights_text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        if line.strip().startswith('💡 ACTION:'):
            print(f"DEBUG: Found action line: {line}")
            # Extract the action text
            action_text = line.strip()
            # Get enhanced version with links
            enhanced_action = get_actionable_links(action_text, industry, store_type)
            print(f"DEBUG: Enhanced action: {enhanced_action}")
            enhanced_lines.append(enhanced_action)
        else:
            enhanced_lines.append(line)
    
    result = '\n'.join(enhanced_lines)
    print(f"DEBUG: Final enhanced result=\n{result}")
    return result

def get_link_documentation():
    """
    Comprehensive documentation of all links provided in the system with purpose and use cases.
    This helps merchants understand exactly what each tool does and how it helps their business.
    """
    
    link_documentation = {
        
        # ==================== COMBO MEALS & BUNDLING ACTIONS ====================
        "combo_bundling": {
            "Menu Design Tools (Canva)": {
                "purpose": "Create professional, attractive menus that highlight combo deals and bundles",
                "use_case": "Restaurant wants to promote 'Lunch Combo: Main + Drink + Dessert for ₹299'. Use Canva to design eye-catching menu boards showing the combo prominently with appetizing visuals and clear savings message.",
                "benefit": "Well-designed menus can increase combo sales by 30-40%"
            },
            "Food Photography (Zomato)": {
                "purpose": "Get professional photos of your combo meals that make them look irresistible",
                "use_case": "Your 'Family Feast' combo looks amazing in person but photos on delivery apps are poor quality. Zomato's photo service creates high-quality images that make customers order more combos online.",
                "benefit": "Professional food photos can increase online orders by 50-70%"
            },
            "POS Menu Setup (Razorpay)": {
                "purpose": "Configure your point-of-sale system to easily ring up combo deals and track their performance",
                "use_case": "You want to offer 'Buy 2 Shirts + Get 1 Tie Free' but staff struggle with pricing. Razorpay POS lets you set up combo buttons that automatically calculate discounts and track which combos sell best.",
                "benefit": "Automated combo pricing reduces errors and speeds up checkout by 25%"
            },
            "Outfit Styling (Canva Lookbooks)": {
                "purpose": "Create visual guides showing how different clothing items work together as complete outfits",
                "use_case": "Fashion store wants to sell 'Complete Office Look: Shirt + Pants + Belt + Shoes'. Create lookbook showing the full outfit on models, helping customers visualize the complete style.",
                "benefit": "Outfit displays can increase average purchase value by 40-60%"
            },
            "Product Bundling (Shopify Blog)": {
                "purpose": "Learn proven strategies for creating attractive product bundles that customers want to buy",
                "use_case": "You sell electronics but customers buy items separately. Learn how to bundle 'Phone + Case + Screen Protector + Charger' as 'Complete Phone Protection Kit' with attractive pricing.",
                "benefit": "Strategic bundling can increase transaction value by 20-35%"
            }
        },

        # ==================== LOYALTY & CUSTOMER RETENTION ACTIONS ====================
        "loyalty_retention": {
            "WhatsApp Business": {
                "purpose": "Send personalized loyalty updates, special offers, and rewards directly to customers' phones",
                "use_case": "Regular customer hasn't visited in 2 weeks. Send WhatsApp message: 'Hi Priya! Miss your usual masala chai ☕ Come back this week and get 20% off your favorite snacks!' with a direct link to order.",
                "benefit": "Direct messaging can bring back 25-35% of inactive customers"
            },
            "Customer Database (Zoho CRM)": {
                "purpose": "Track every customer's purchase history, preferences, and behavior to create personalized rewards",
                "use_case": "Mr. Sharma always buys formal shirts on Fridays. Your system automatically sends him an SMS every Thursday: 'New formal collection arrived! First look for our VIP customer - 15% off tomorrow only.'",
                "benefit": "Personalized offers have 3x higher conversion rates than generic promotions"
            },
            "Restaurant Loyalty (Toast)": {
                "purpose": "Set up point-based loyalty system specifically designed for restaurants",
                "use_case": "Customer orders biryani 3 times = gets 4th biryani free. System tracks automatically, sends progress updates: 'You're 1 biryani away from your free meal!' Creates excitement and repeat visits.",
                "benefit": "Loyalty programs can increase repeat visits by 40-50%"
            },
            "Style Rewards (Razorpay Magic Checkout)": {
                "purpose": "Create smooth, fast checkout experience that makes loyal customers want to return",
                "use_case": "VIP fashion customer gets one-click checkout for new arrivals. No need to enter details repeatedly - just click and buy. Makes luxury shopping feel effortless.",
                "benefit": "Smooth checkout experience increases repeat purchases by 30%"
            }
        },

        # ==================== DELIVERY & ONLINE ORDERING ACTIONS ====================
        "delivery_online": {
            "Zomato Partner": {
                "purpose": "Join India's largest food delivery platform to reach millions of hungry customers",
                "use_case": "Small restaurant has only 20 daily customers. After joining Zomato, gets 50+ delivery orders daily from customers within 5km radius who discovered them online.",
                "benefit": "Delivery platforms can increase customer reach by 200-300%"
            },
            "Swiggy Partner": {
                "purpose": "Access Swiggy's delivery network and customer base for restaurant growth",
                "use_case": "Office area restaurant slow on weekends. Swiggy connects them to nearby residential customers who order family meals, doubling weekend revenue.",
                "benefit": "Multi-platform presence can increase total orders by 60-80%"
            },
            "Payment Links (Razorpay)": {
                "purpose": "Create simple links customers can click to order and pay instantly, even without an app",
                "use_case": "Customer calls asking 'Can I order biryani for pickup?' You instantly send WhatsApp payment link. They click, pay ₹350, order confirmed. No cash handling needed.",
                "benefit": "Payment links can reduce order abandonment by 40%"
            },
            "WhatsApp Catalog": {
                "purpose": "Display your full product range with photos and prices directly in WhatsApp for easy ordering",
                "use_case": "Electronics store creates WhatsApp catalog with phone cases, chargers, headphones. Customers browse, choose, and order directly through chat - no separate app needed.",
                "benefit": "WhatsApp ordering can increase sales by 50% among existing customers"
            }
        },

        # ==================== MARKETING & PROMOTION ACTIONS ====================
        "marketing_promotion": {
            "Google My Business": {
                "purpose": "Make your business discoverable when people search for services near them",
                "use_case": "Someone searches 'best biryani near me' at 8 PM. Your restaurant appears first with photos, timing, menu, and 4.5-star reviews. They call and order immediately.",
                "benefit": "Optimized Google listing can increase foot traffic by 25-40%"
            },
            "Facebook Business": {
                "purpose": "Create targeted ads to reach potential customers in your area with specific interests",
                "use_case": "Fashion store targets ads to 'Women aged 25-40 within 5km who like fashion brands'. Ad shows new collection with 'First 50 customers get 30% off' - brings 20 new customers in one week.",
                "benefit": "Targeted social media ads can generate 10x return on ad spend"
            },
            "Mall Advertising": {
                "purpose": "Learn how to advertise effectively to mall shoppers with higher purchasing power",
                "use_case": "Jewelry store in mall creates ads targeting 'People who visited malls in last 30 days' for wedding season. Reaches engaged couples planning to shop for wedding jewelry.",
                "benefit": "Mall-specific advertising can increase premium sales by 60%"
            },
            "Local Ads (Google)": {
                "purpose": "Show ads only to people in your immediate neighborhood who are most likely to visit",
                "use_case": "Street-front pharmacy shows ads for 'Medicine delivery in 30 minutes' only to people within 2km radius during evening hours when they're most likely to need it.",
                "benefit": "Hyper-local ads have 5x higher conversion rates than broad targeting"
            }
        },

        # ==================== STAFF TRAINING & SERVICE ACTIONS ====================
        "staff_training": {
            "Customer Service Training (Udemy)": {
                "purpose": "Train staff to handle customers professionally, increasing satisfaction and repeat business",
                "use_case": "Restaurant staff often argue with customers about order mistakes. After training, they learn to say 'I apologize, let me fix this immediately' and offer compensation. Customer complaints drop 80%.",
                "benefit": "Trained staff can increase customer retention by 45%"
            },
            "Staff Management (Razorpay Payroll)": {
                "purpose": "Automate salary payments and manage staff efficiently to reduce administrative burden",
                "use_case": "Store owner spends 2 hours monthly calculating salaries, bonuses, deductions. Razorpay Payroll automates everything - salaries paid on time, staff happier, owner focuses on business growth.",
                "benefit": "Automated payroll saves 10-15 hours monthly and reduces errors"
            },
            "Food Service Training": {
                "purpose": "Specialized training for restaurant staff on food safety, service speed, and customer interaction",
                "use_case": "Waiters don't know how to describe dishes to customers. Training teaches them to say 'Our butter chicken is creamy, mildly spiced, perfect with naan' instead of just 'it's good'. Orders increase 25%.",
                "benefit": "Specialized training can increase average order value by 20-30%"
            }
        },

        # ==================== PAYMENT & CHECKOUT ACTIONS ====================
        "payment_checkout": {
            "Razorpay POS": {
                "purpose": "Accept all types of payments (cards, UPI, wallets) in one device, making checkout faster",
                "use_case": "Customer wants to pay ₹1,247 but only has ₹1,000 cash. With Razorpay POS, they pay ₹1,000 cash + ₹247 via UPI. Sale completed, customer happy.",
                "benefit": "Multiple payment options can reduce lost sales by 30%"
            },
            "Payment Gateway": {
                "purpose": "Accept online payments securely for e-commerce, bookings, or advance orders",
                "use_case": "Cake shop takes advance orders for birthdays. Customers pay 50% online while ordering, 50% on delivery. No more cancelled orders due to payment issues.",
                "benefit": "Online payments can reduce order cancellations by 50%"
            },
            "QR Code Setup": {
                "purpose": "Generate simple QR codes customers can scan to pay instantly without cash",
                "use_case": "Street food vendor places QR code on cart. Customers scan, pay ₹50 for vada pav, payment confirmed instantly. No cash handling, no change issues, faster service.",
                "benefit": "QR payments can speed up transactions by 40% and improve cash flow"
            }
        },

        # ==================== ANALYTICS & INSIGHTS ACTIONS ====================
        "analytics_insights": {
            "Business Analytics (Razorpay Dashboard)": {
                "purpose": "See which products sell best, when customers buy most, and track business growth patterns",
                "use_case": "Clothing store discovers 70% sales happen between 6-9 PM on weekdays. Adjusts staff schedule and runs 'Evening Special' discounts during peak hours, increasing sales 25%.",
                "benefit": "Data-driven decisions can improve profitability by 30-40%"
            },
            "Sales Reports": {
                "purpose": "Get detailed breakdowns of daily, weekly, monthly sales to identify trends and opportunities",
                "use_case": "Restaurant notices biryani sales drop every Tuesday. Investigation reveals competitor's Tuesday biryani offer. Introduces 'Tuesday Tandoori Special' to compete, recovers lost revenue.",
                "benefit": "Regular reporting helps identify and fix revenue leaks worth 15-25%"
            },
            "Performance Tracking (Google Analytics)": {
                "purpose": "Track how customers find and interact with your business online",
                "use_case": "Jewelry store sees most website visitors come from 'gold price' searches but don't buy. Creates 'Gold Price Protection Plan' offer for these visitors, converting 20% to customers.",
                "benefit": "Website optimization can increase online inquiries by 50-100%"
            }
        },

        # ==================== CUSTOMER EXPERIENCE ACTIONS ====================
        "customer_experience": {
            "Feedback System (Google Reviews)": {
                "purpose": "Collect and respond to customer reviews to build trust and improve service",
                "use_case": "Restaurant gets review: 'Food great but service slow'. Owner responds publicly: 'Thank you for feedback! We've added 2 servers for faster service. Please visit again!' Shows responsiveness.",
                "benefit": "Responding to reviews can increase positive ratings by 30% and attract more customers"
            },
            "Customer Support (WhatsApp)": {
                "purpose": "Provide instant customer support through the app most Indians use daily",
                "use_case": "Customer messages at 10 PM: 'Is my saree alteration ready?' Quick reply: 'Yes! Ready for pickup tomorrow after 11 AM. Here's the photo.' Customer feels valued and cared for.",
                "benefit": "Instant support can increase customer satisfaction scores by 40%"
            },
            "Table Booking (OpenTable)": {
                "purpose": "Allow customers to book restaurant tables online, reducing wait times and managing capacity",
                "use_case": "Family wants dinner on Saturday 8 PM. Books table online, arrives to reserved seating. No waiting, no arguments. Restaurant manages capacity better, serves more customers.",
                "benefit": "Online booking can increase table utilization by 25% and reduce no-shows"
            }
        }
    }
    
    return link_documentation

def display_link_guide():
    """
    Display a very concise, text-based guide of business tools suitable for sidebar.
    """
    
    # Minimal text-based guide for sidebar
    guide_text = """
**📚 Business Tools Quick Reference**

When you see tool links in your insights, here's what they help with:

**🍽️ Combo & Bundling:** Canva (menu design), POS systems, product bundling
**🔄 Customer Loyalty:** WhatsApp Business, CRM, loyalty programs  
**🚚 Delivery & Online:** Zomato/Swiggy, payment links, WhatsApp catalog
**📢 Marketing:** Google My Business, Facebook ads, local advertising
**👥 Staff Training:** Udemy courses, payroll systems, service training
**💳 Payments:** Razorpay POS, payment gateways, QR codes
**📊 Analytics:** Business dashboards, Google Analytics, sales reports
**⭐ Customer Experience:** Google Reviews, WhatsApp support, booking systems

**💡 Tools are chosen based on:**
• Your industry (Restaurant/Retail/Fashion)
• Store type (Mall/Street/Standalone)
• Specific performance gaps
• Business size & customer patterns

**🎯 Quick Start:**
• **Low Sales?** → Use Marketing tools
• **Low Repeat Customers?** → Focus on Loyalty systems  
• **Low Order Value?** → Try Bundling strategies
"""
    
    return guide_text

def display_full_link_guide():
    """
    Display the complete detailed guide of all available links and their purposes.
    This is the original comprehensive version for separate pages/tabs.
    """
    
    documentation = get_link_documentation()
    
    guide_html = """
    <div style='max-width: 1200px; margin: 0 auto; padding: 20px; color: #e0e0e0;'>
        <h1 style='color: #4dabf7; text-align: center; margin-bottom: 30px;'>📚 Business Tools Guide</h1>
        <p style='text-align: center; font-size: 1.1em; margin-bottom: 40px; color: #adb5bd;'>
            Comprehensive guide to all business tools and links provided in your AI insights
        </p>
    """
    
    # Category headers with emojis and descriptions
    category_info = {
        "combo_bundling": {
            "title": "🍽️ Combo Meals & Product Bundling",
            "description": "Tools to create attractive combos and bundles that increase your average transaction value"
        },
        "loyalty_retention": {
            "title": "🔄 Customer Loyalty & Retention", 
            "description": "Systems to keep customers coming back and increase repeat purchases"
        },
        "delivery_online": {
            "title": "🚚 Delivery & Online Ordering",
            "description": "Platforms to expand your reach beyond physical store location"
        },
        "marketing_promotion": {
            "title": "📢 Marketing & Promotion",
            "description": "Tools to attract new customers and increase your business visibility"
        },
        "staff_training": {
            "title": "👥 Staff Training & Management",
            "description": "Resources to improve your team's skills and efficiency"
        },
        "payment_checkout": {
            "title": "💳 Payment & Checkout Solutions",
            "description": "Modern payment systems to make transactions smooth and fast"
        },
        "analytics_insights": {
            "title": "📊 Analytics & Business Insights",
            "description": "Data tools to understand your business performance and make better decisions"
        },
        "customer_experience": {
            "title": "⭐ Customer Experience Enhancement",
            "description": "Tools to improve how customers interact with your business"
        }
    }
    
    for category_key, category_data in documentation.items():
        if category_key in category_info:
            cat_info = category_info[category_key]
            
            guide_html += f"""
            <div style='margin: 40px 0; padding: 25px; border-radius: 12px; border-left: 4px solid #4dabf7; background: rgba(77, 171, 247, 0.1);'>
                <h2 style='color: #4dabf7; margin-bottom: 10px;'>{cat_info['title']}</h2>
                <p style='color: #adb5bd; margin-bottom: 25px; font-style: italic;'>{cat_info['description']}</p>
            """
            
            for tool_name, tool_info in category_data.items():
                guide_html += f"""
                <div style='margin: 20px 0; padding: 20px; border-radius: 8px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);'>
                    <h3 style='color: #e67700; margin-bottom: 12px; font-size: 1.1em;'>{tool_name}</h3>
                    
                    <div style='margin: 12px 0;'>
                        <strong style='color: #4dabf7;'>Purpose:</strong>
                        <span style='color: #e0e0e0;'> {tool_info['purpose']}</span>
                    </div>
                    
                    <div style='margin: 12px 0;'>
                        <strong style='color: #2b8a3e;'>Real Example:</strong>
                        <span style='color: #e0e0e0; font-style: italic;'> {tool_info['use_case']}</span>
                    </div>
                    
                    <div style='margin: 12px 0; padding: 10px; background: rgba(43, 138, 62, 0.2); border-radius: 6px; border-left: 3px solid #2b8a3e;'>
                        <strong style='color: #2b8a3e;'>💰 Expected Benefit:</strong>
                        <span style='color: #e0e0e0;'> {tool_info['benefit']}</span>
                    </div>
                </div>
                """
            
            guide_html += "</div>"
    
    guide_html += """
        <div style='margin-top: 50px; padding: 30px; text-align: center; background: rgba(77, 171, 247, 0.1); border-radius: 12px; border: 2px solid #4dabf7;'>
            <h2 style='color: #4dabf7; margin-bottom: 15px;'>🚀 Getting Started</h2>
            <p style='color: #e0e0e0; font-size: 1.1em; margin-bottom: 20px;'>
                Choose tools based on your current business challenges:
            </p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;'>
                <div style='padding: 15px; background: rgba(230, 119, 0, 0.2); border-radius: 8px;'>
                    <strong style='color: #e67700;'>📈 Need More Sales?</strong><br>
                    <span style='color: #e0e0e0; font-size: 0.9em;'>Start with Marketing & Promotion tools</span>
                </div>
                <div style='padding: 15px; background: rgba(43, 138, 62, 0.2); border-radius: 8px;'>
                    <strong style='color: #2b8a3e;'>🔄 Want Repeat Customers?</strong><br>
                    <span style='color: #e0e0e0; font-size: 0.9em;'>Focus on Loyalty & Retention systems</span>
                </div>
                <div style='padding: 15px; background: rgba(77, 171, 247, 0.2); border-radius: 8px;'>
                    <strong style='color: #4dabf7;'>💰 Increase Order Value?</strong><br>
                    <span style='color: #e0e0e0; font-size: 0.9em;'>Use Combo & Bundling strategies</span>
                </div>
            </div>
        </div>
    </div>
    """
    
    return guide_html
