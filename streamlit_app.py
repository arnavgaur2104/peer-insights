# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy
import traceback # Import traceback for error printing
import plotly.graph_objects as go

# Use the new comparison function and insight engine
# Ensure these imports don't implicitly call Streamlit functions before set_page_config
# Make sure compare_merchants.py and insights_engine.py are in the same directory or accessible
try:
    # Assuming compare_merchants.py has the corrected get_comparison_data function
    from compare_merchants import get_comparison_data
    # Update insights_engine imports to only include what we need
    from insights_engine import (
        generate_crisp_insights,
        format_insights_for_display
    )
except ImportError as import_err:
    # Display error in the app if imports fail
    st.set_page_config(page_title="Import Error", layout="centered")
    st.error(f"Error importing necessary functions: {import_err}. Make sure compare_merchants.py and insights_engine.py are present and correct.")
    st.stop()


# --- Set Page Config FIRST ---
st.set_page_config(
    page_title="Merchant AI Insights v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'detailed_insights' not in st.session_state:
    st.session_state.detailed_insights = None
if 'show_detailed_analysis' not in st.session_state:
    st.session_state.show_detailed_analysis = False
if 'show_visual_insights' not in st.session_state:
    st.session_state.show_visual_insights = False
if 'crisp_insights' not in st.session_state:
    st.session_state.crisp_insights = None

# Add custom CSS for better styling
st.markdown("""
    <style>
    /* Make sidebar background match main background */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }

    /* Remove grey background from search box */
    section[data-testid="stSidebar"] input[type="text"] {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040 !important;
        border-radius: 5px !important;
    }
    section[data-testid="stSidebar"] input[type="text"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px #4dabf7 !important;
    }

    /* Remove grey background from selectbox (dropdown) */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040 !important;
        border-radius: 5px !important;
    }
    /* Dropdown menu and options */
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div[aria-selected="true"] {
        background-color: #4dabf7 !important;
        color: #1a1a1a !important;
    }

    /* Remove grey backgrounds from all insights and cards */
    .insight-card, .insight-section, .highlight, .metric-card, .performance-section, .insight-recommendation, .performance-details {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    /* Remove background from any divs inside insights */
    .insight-section div, .insight-card div, .performance-section div, .highlight div {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Tooltip styles */
    [title] {
        position: relative;
        cursor: help;
    }
    [title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        background-color: #2b2b2b;
        color: #e0e0e0;
        border-radius: 4px;
        font-size: 0.8em;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #404040;
        max-width: 300px;
        white-space: normal;
    }
    [title]:hover::before {
        content: '';
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 5px;
        border-style: solid;
        border-color: #404040 transparent transparent transparent;
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data (Define Function AFTER set_page_config) ---
@st.cache_data # Cache data loading
def load_data():
    """Loads merchant and competitor data from CSV files."""
    try:
        merchants = pd.read_csv('data/merchants.csv')
        competitors = pd.read_csv('data/competitors.csv')
        # Basic validation
        if merchants.empty:
             st.warning("Merchant data file (merchants.csv) is empty. Please run generate_data.py.")
             # Return None if merchants is empty as it's crucial
             return None, competitors
        if competitors.empty:
              st.warning("Competitor data file (competitors.csv) is empty. Comparisons may be limited.")
              # Allow continuing but with a warning
        return merchants, competitors
    except FileNotFoundError:
        # Use st.error ONLY after set_page_config has been called
        st.error("Error: `data/merchants.csv` or `data/competitors.csv` not found. Please run `generate_data.py` first.")
        return None, None # Return None to indicate failure
    except pd.errors.EmptyDataError as e:
         st.error(f"Error: A data file is empty or invalid ({e}). Please check data/merchants.csv and data/competitors.csv.")
         return None, None
    except Exception as e:
         st.error(f"An unexpected error occurred during data loading: {e}")
         st.text(traceback.format_exc())
         return None, None


# --- Call Load Data Function ---
# Call the function after defining it and after set_page_config
merchants, competitors = load_data()

# --- Start Building the UI ---
st.title("🧠 AI-Powered Merchant Insights")

# Add a welcome message
st.markdown("""
    Welcome to your personalized merchant insights dashboard! 
    Select a merchant from the sidebar to get started with AI-powered analysis and recommendations.
""")

# Check if data loaded successfully
if merchants is None or merchants.empty:
    st.error("No merchant data available. Please check your data files and try again.")
    st.stop()

# Initialize merchant list
merchant_id_list = merchants['merchant_id'].unique().tolist()

# --- Sidebar ---
with st.sidebar:
    st.header("🔍 Select Merchant")
    
    # Add filters section
    st.subheader("Filters")
    
    # Industry filter
    industries = sorted(merchants['industry'].unique().tolist())
    selected_industry = st.selectbox(
        "Industry",
        ["All"] + industries,
        index=0
    )
    
    # Store Type filter
    store_types = sorted(merchants['store_type'].unique().tolist())
    selected_store_type = st.selectbox(
        "Store Type",
        ["All"] + store_types,
        index=0
    )
    
    # City filter
    cities = sorted(merchants['city'].unique().tolist())
    selected_city = st.selectbox(
        "City",
        ["All"] + cities,
        index=0
    )
    
    st.divider()
    
    # Add a search box for merchant IDs
    merchant_search = st.text_input("Search Merchant ID", "")
    
    # Filter merchants using pandas (much more efficient)
    filtered_df = merchants.copy()
    
    # Apply filters
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
    
    if selected_store_type != "All":
        filtered_df = filtered_df[filtered_df['store_type'] == selected_store_type]
    
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df['city'] == selected_city]
    
    # Apply search filter
    if merchant_search:
        filtered_df = filtered_df[filtered_df['merchant_id'].str.lower().str.contains(merchant_search.lower())]
    
    # Get filtered merchant IDs and sort them
    filtered_merchants = sorted(filtered_df['merchant_id'].tolist())
    
    # Display merchant count
    st.markdown(f"**Found {len(filtered_merchants)} merchants**")
    
    if filtered_merchants:
        # Simple selectbox with just merchant IDs for now
        merchant_id = st.selectbox(
            "Select from Results",
            filtered_merchants,
            index=0
        )
        
        # Reset session state when merchant changes
        if 'last_merchant' not in st.session_state:
            st.session_state.last_merchant = None
            
        if merchant_id != st.session_state.last_merchant:
            st.session_state.crisp_insights = None
            st.session_state.detailed_insights = None
            st.session_state.show_detailed_analysis = False
            st.session_state.show_visual_insights = False
            st.session_state.last_merchant = merchant_id
            
        # Display selected merchant info
        selected_merchant_info = merchants[merchants['merchant_id'] == merchant_id].iloc[0]
        st.markdown(f"**Selected:** {merchant_id}")
        st.markdown(f"**Industry:** {selected_merchant_info['industry']}")
        st.markdown(f"**City:** {selected_merchant_info['city']}")
        st.markdown(f"**Store Type:** {selected_merchant_info['store_type']}")
    else:
        st.warning("No merchants found matching your filters.")
        merchant_id = None

    st.divider()
    
    # Add clustering explanation
    with st.expander("ℹ️ Understanding Your Analysis", expanded=False):
        st.markdown("""
        ### How We Analyze Your Business

        We use two powerful methods to understand your business performance:

        #### 1. Local Competitors 👥
        - Businesses in your immediate area (same pincode)
        - Same industry as yours
        - Direct competition for customers
        - Helps you understand your local market position

        #### 2. Similar Businesses (Clusters) 🔄
        - Businesses with similar performance patterns
        - May be in different locations
        - Share similar characteristics:
            - Transaction patterns
            - Customer behavior
            - Operational efficiency
            - Business size and scale

        #### Why This Matters
        - Compare with immediate competition
        - Learn from similar businesses
        - Identify unique opportunities
        - Make data-driven decisions
        """)

# --- Main Area ---
if merchant_id:
    # Get merchant data
    merchant_row, comparison_df_local, comparison_df_cluster, local_competitors, cluster_peers, cluster_averages = get_comparison_data(merchant_id, merchants, competitors)

    if merchant_row is not None:
        # Create a header with merchant info
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.markdown(f"### {merchant_row.get('industry', 'Business')} Analysis")
            st.markdown(f"**Location:** {merchant_row.get('city', 'N/A')} | **Store Type:** {merchant_row.get('store_type', 'N/A')}")
        
        # Quick stats in cards
        with col2:
            st.metric(
                "Average Transaction",
                f"₹{merchant_row.get('avg_txn_value', 0):.2f}",
                f"{((merchant_row.get('avg_txn_value', 0) - comparison_df_local['Local Avg'].iloc[0]) / comparison_df_local['Local Avg'].iloc[0] * 100):.1f}%" if comparison_df_local is not None else None
            )
        
        with col3:
            st.metric(
                "Daily Customers",
                f"{merchant_row.get('daily_txn_count', 0)}",
                f"{((merchant_row.get('daily_txn_count', 0) - comparison_df_local['Local Avg'].iloc[1]) / comparison_df_local['Local Avg'].iloc[1] * 100):.1f}%" if comparison_df_local is not None else None
            )

        # Main content in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Key Insights", "📈 Performance", "📋 Details"])

        with tab1:
            # Quick Insights
            try:
                st.markdown("### 📊 Key Insights")
                
                # Initialize session state for insights if not exists
                if 'crisp_insights' not in st.session_state:
                    st.session_state.crisp_insights = None
                if 'impact_data' not in st.session_state:
                    st.session_state.impact_data = None
                
                # Only generate insights if they don't exist in session state
                if st.session_state.crisp_insights is None:
                    with st.spinner("Generating insights..."):
                        st.session_state.crisp_insights, st.session_state.impact_data = generate_crisp_insights(
                            merchant_row,
                            comparison_df_local,
                            comparison_df_cluster,
                            cluster_peers,
                            cluster_averages
                        )
                
                # Format insights for display
                formatted_insights = format_insights_for_display(st.session_state.crisp_insights)
                
                # Display each insight with its corresponding graph
                for i, (insight_html, metric, insight_index) in enumerate(formatted_insights):
                    # Display the insight
                    st.markdown(insight_html, unsafe_allow_html=True)
                    
                    # Find and display the corresponding impact data
                    if st.session_state.impact_data:
                        for data in st.session_state.impact_data:
                            if data['metric'] == metric and data['insight_index'] == insight_index:
                                # Create bar chart
                                fig = go.Figure()
                                
                                # Add bars for current, expected, local average, and cluster average
                                x_labels = ['Current', 'Local Avg']
                                y_values = [data['current'], data['local_avg']]
                                colors = ['#4dabf7', '#adb5bd']
                                
                                # Add cluster average if available
                                if data['cluster_avg'] is not None:
                                    x_labels.append('Cluster Avg')
                                    y_values.append(data['cluster_avg'])
                                    colors.append('#e67700')
                                
                                # Only add expected value for non-income metrics
                                if data['metric'] != 'Income Level':
                                    x_labels.insert(1, 'Expected')
                                    y_values.insert(1, data['expected'])
                                    colors.insert(1, '#2b8a3e')
                                
                                fig.add_trace(go.Bar(
                                    x=x_labels,
                                    y=y_values,
                                    text=[f'{v:.2f}' for v in y_values],
                                    textposition='auto',
                                    marker_color=colors,
                                    name=metric
                                ))
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"{metric} Analysis",
                                    xaxis_title="",
                                    yaxis_title="Value",
                                    showlegend=False,
                                    height=300,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#e0e0e0'),
                                    bargap=0.3
                                )
                                
                                # Add annotations for percentage changes only for non-income metrics
                                if data['metric'] != 'Income Level':
                                    fig.add_annotation(
                                        x='Expected',
                                        y=data['expected'],
                                        text=f'{data["impact_pct"]*100:+.1f}%',
                                        showarrow=True,
                                        arrowhead=1,
                                        ax=0,
                                        ay=-40,
                                        font=dict(color='#2b8a3e', size=12)
                                    )
                                
                                # Customize axes
                                fig.update_yaxes(
                                    gridcolor='rgba(128, 128, 128, 0.2)',
                                    zerolinecolor='rgba(128, 128, 128, 0.2)',
                                    showgrid=True
                                )
                                fig.update_xaxes(showgrid=False)
                                
                                # Display the chart with a unique key
                                st.plotly_chart(fig, use_container_width=True, key=f"impact_chart_{insight_index}")
                                
                                # Add metric details with cluster average
                                details_html = f"""
                                <div style='margin-top: -20px; margin-bottom: 30px;'>
                                    <p style='color: #e0e0e0; font-size: 0.9em;'>
                                        <span style='color: #4dabf7;' title='Your current performance value'>●</span> Current: {data['current']:.2f} | 
                                """
                                if data['metric'] != 'Income Level':
                                    details_html += f"<span style='color: #2b8a3e;' title='Expected value after implementing the suggested action'>●</span> Expected: {data['expected']:.2f} | "
                                details_html += f"<span style='color: #adb5bd;' title='Average of businesses in your immediate area (same pincode and industry)'>●</span> Local Average: {data['local_avg']:.2f}"
                                if data['cluster_avg'] is not None:
                                    details_html += f" | <span style='color: #e67700;' title='Average of businesses with similar performance patterns, regardless of location'>●</span> Cluster Average: {data['cluster_avg']:.2f}"
                                details_html += "</p></div>"
                                
                                st.markdown(details_html, unsafe_allow_html=True)
                                break
            except Exception as insight_err:
                st.error(f"Error generating insights: {insight_err}")

        with tab2:
            # Performance Metrics
            st.markdown("""
                <style>
                .performance-section {
                    margin: 2rem 0;
                    padding: 1rem;
                    border-left: 4px solid #4dabf7;
                }
                .performance-title {
                    color: #e0e0e0;
                    font-size: 1.4rem;
                    margin-bottom: 0.5rem;
                }
                .performance-metrics {
                    display: flex;
                    gap: 2rem;
                    margin: 1rem 0;
                }
                .performance-metric {
                    flex: 1;
                }
                .performance-value {
                    color: #4dabf7;
                    font-weight: 600;
                }
                .performance-average {
                    color: #adb5bd;
                }
                .performance-status {
                    margin-top: 0.5rem;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-weight: 500;
                }
                .status-good {
                    color: #2b8a3e;
                }
                .status-warning {
                    color: #e67700;
                }
                .status-bad {
                    color: #e03131;
                }
                .performance-details {
                    padding: 1rem;
                    border-radius: 4px;
                    margin-top: 1rem;
                }
                .performance-detail-item {
                    margin: 0.5rem 0;
                    color: #b0b0b0;
                }
                .performance-detail-value {
                    color: #e0e0e0;
                    font-weight: 500;
                }
                </style>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Local Market Position")
                if comparison_df_local is not None:
                    for _, row in comparison_df_local.iterrows():
                        status_class = "status-good" if "✅" in str(row['Performance']) else "status-warning" if "⚠️" in str(row['Performance']) else "status-bad"
                        st.markdown(f"""
                        <div class="performance-section">
                            <div class="performance-title">{row['Metric']}</div>
                            <div class="performance-metrics">
                                <div class="performance-metric">
                                    <p>Your Value: <span class="performance-value">{row['Merchant Value']}</span></p>
                                    <p>Local Average: <span class="performance-average">{row['Local Avg']}</span></p>
                                </div>
                                <div class="performance-metric">
                                    <p class="performance-status {status_class}">{row['Performance']}</p>
                                </div>
                            </div>
                            <div class="performance-details">
                                <div class="performance-detail-item">
                                    <span class="performance-detail-value">Impact:</span> {((row['Merchant Value'] - row['Local Avg']) / row['Local Avg'] * 100):.1f}% vs local average
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col2:
                st.markdown("### Cluster Comparison")
                if comparison_df_cluster is not None:
                    for _, row in comparison_df_cluster.iterrows():
                        status_class = "status-good" if "✅" in str(row['Performance']) else "status-warning" if "⚠️" in str(row['Performance']) else "status-bad"
                        st.markdown(f"""
                        <div class="performance-section">
                            <div class="performance-title">{row['Metric']}</div>
                            <div class="performance-metrics">
                                <div class="performance-metric">
                                    <p>Your Value: <span class="performance-value">{row['Merchant Value']}</span></p>
                                    <p>Cluster Average: <span class="performance-average">{row['Cluster Avg']}</span></p>
                                </div>
                                <div class="performance-metric">
                                    <p class="performance-status {status_class}">{row['Performance']}</p>
                                </div>
                            </div>
                            <div class="performance-details">
                                <div class="performance-detail-item">
                                    <span class="performance-detail-value">Impact:</span> {((row['Merchant Value'] - row['Cluster Avg']) / row['Cluster Avg'] * 100):.1f}% vs cluster average
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab3:
            # Detailed Information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Business Profile")
                profile_data = {
                    "Industry": merchant_row.get('industry', 'N/A'),
                    "Store Type": merchant_row.get('store_type', 'N/A'),
                    "Location": merchant_row.get('city', 'N/A'),
                    "Income Level": f"₹{merchant_row.get('income_level', 0):.2f}"
                }
                for key, value in profile_data.items():
                    st.markdown(f"**{key}:** {value}")

            with col2:
                st.subheader("Peer Comparison")
                if cluster_peers is not None and not cluster_peers.empty:
                    st.markdown(f"**Similar Businesses in Your Cluster:** {len(cluster_peers)}")
                    st.dataframe(
                        cluster_peers[['merchant_id', 'city', 'store_type']].head(),
                        use_container_width=True,
                        hide_index=True
                    )

        # Download section at the bottom
        st.divider()
        st.subheader("⬇️ Download Reports")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        if comparison_df_local is not None:
            col_dl1.download_button(
                "📊 Local Comparison",
                comparison_df_local.to_csv(index=False).encode('utf-8'),
                f"{merchant_id}_local_comparison.csv",
                "text/csv"
            )
        
        if comparison_df_cluster is not None:
            col_dl2.download_button(
                "📈 Cluster Comparison",
                comparison_df_cluster.to_csv(index=False).encode('utf-8'),
                f"{merchant_id}_cluster_comparison.csv",
                "text/csv"
            )
        
        if st.session_state.detailed_insights:
            col_dl3.download_button(
                "📝 AI Insights",
                st.session_state.detailed_insights,
                f"{merchant_id}_insights.txt",
                "text/plain"
            )

else:
    # Welcome screen when no merchant is selected
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Welcome to Merchant Insights</h2>
            <p>Select a merchant from the sidebar to get started with AI-powered analysis.</p>
        </div>
    """, unsafe_allow_html=True)

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