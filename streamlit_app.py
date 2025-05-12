# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy
import traceback # Import traceback for error printing

# Use the new comparison function and insight engine
# Ensure these imports don't implicitly call Streamlit functions before set_page_config
# Make sure compare_merchants.py and insights_engine.py are in the same directory or accessible
try:
    # Assuming compare_merchants.py has the corrected get_comparison_data function
    from compare_merchants import get_comparison_data
    # Assuming insights_engine.py has the corrected functions
    from insights_engine import generate_advanced_ai_insights, generate_insights, generate_quick_insights, format_quick_comparison
except ImportError as import_err:
     # Display error in the app if imports fail
     st.set_page_config(page_title="Import Error", layout="centered") # Minimal config for error
     st.error(f"Error importing necessary functions: {import_err}. Make sure compare_merchants.py and insights_engine.py are present and correct.")
     st.stop() # Stop execution if imports fail


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

# Initialize merchant list
merchant_id_list = []
if not merchants.empty:
    merchant_id_list = merchants['merchant_id'].unique().tolist()

# --- Sidebar ---
with st.sidebar:
    st.header("🔍 Select Merchant")
    
    # Add a search box for merchant IDs
    merchant_search = st.text_input("Search Merchant ID", "")
    
    # Filter merchant IDs based on search
    if merchant_search:
        filtered_merchants = [m for m in merchant_id_list if merchant_search.lower() in m.lower()]
    else:
        filtered_merchants = merchant_id_list
    
    if filtered_merchants:
        merchant_id = st.selectbox(
            "Select from Results",
            sorted(filtered_merchants),
            index=0
        )
    else:
        st.warning("No merchants found matching your search.")
        merchant_id = None

    st.divider()
    
    # Add configuration options in an expander
    with st.expander("⚙️ Settings", expanded=False):
        show_simple_insights = st.checkbox("Show Simple Rule-Based Insights", value=False)
        st.info("Toggle additional insights and analysis options here.")

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
        tab1, tab2, tab3 = st.tabs(["📊 Quick Insights", "📈 Performance", "📋 Details"])

        with tab1:
            # Quick Insights
            try:
                quick_insights = generate_quick_insights(
                    merchant_row,
                    comparison_df_local,
                    comparison_df_cluster,
                    cluster_peers,
                    cluster_averages
                )
                
                # Display quick insights
                st.markdown(quick_insights, unsafe_allow_html=True)
                
                # Add space and center the toggle button
                st.markdown("<br><br>", unsafe_allow_html=True)  # Add vertical space
                col1, col2, col3 = st.columns([1,2,1])  # Create 3 columns, middle one is wider
                with col2:  # Use middle column for the button
                    if st.button("📊 Toggle Detailed Analysis", type="secondary", use_container_width=True):
                        st.session_state.show_detailed_analysis = not st.session_state.show_detailed_analysis
                
                # Add more space before detailed analysis
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                if st.session_state.show_detailed_analysis:
                    st.divider()
                    st.subheader("📊 Detailed Analysis")
                    if st.session_state.detailed_insights is None:
                        with st.spinner("Generating detailed analysis..."):
                            st.session_state.detailed_insights = generate_advanced_ai_insights(
                                merchant_row,
                                comparison_df_local,
                                comparison_df_cluster,
                                cluster_peers,
                                cluster_averages
                            )
                    st.markdown(st.session_state.detailed_insights)
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
                    "Store Size": f"{merchant_row.get('store_size_sqft', 0)} sq ft",
                    "Income Level": f"₹{merchant_row.get('income_level', 0):.2f}",
                    "Rent % of Revenue": f"{merchant_row.get('rent_pct_revenue', 0)*100:.1f}%"
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