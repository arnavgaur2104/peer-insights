# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy
import traceback # Import traceback for error printing
import plotly.graph_objects as go
import re

# Use the new comparison function and insight engine
# Ensure these imports don't implicitly call Streamlit functions before set_page_config
# Make sure compare_merchants.py and insights_engine.py are in the same directory or accessible
try:
    # Assuming compare_merchants.py has the corrected get_comparison_data function
    from compare_merchants import get_comparison_data
    # Update insights_engine imports to only include what we need
    from insights_engine import (
        generate_crisp_insights,
        format_insights_for_display,
        display_link_guide,
        display_full_link_guide
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

    /* Performance insight styles */
    .performance-insight {
        margin-top: 10px;
        padding: 10px;
        border-left: 3px solid #e67700;
        background-color: rgba(230, 119, 0, 0.1);
        border-radius: 0 4px 4px 0;
    }
    .insight-title {
        color: #e67700;
        font-weight: 700;
        font-size: 0.9em;
        margin-bottom: 8px;
    }
    .insight-content {
        color: #e0e0e0;
        font-size: 0.85em;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data (Define Function AFTER set_page_config) ---
@st.cache_data # Cache data loading
def load_data():
    """Loads merchant data from CSV files."""
    try:
        merchants = pd.read_csv('data/merchants.csv')
        # Basic validation
        if merchants.empty:
             st.warning("Merchant data file (merchants.csv) is empty. Please run generate_data.py.")
             # Return None if merchants is empty as it's crucial
             return None
        return merchants
    except FileNotFoundError:
        # Use st.error ONLY after set_page_config has been called
        st.error("Error: `data/merchants.csv` not found. Please run `generate_data.py` first.")
        return None # Return None to indicate failure
    except pd.errors.EmptyDataError as e:
         st.error(f"Error: A data file is empty or invalid ({e}). Please check data/merchants.csv.")
         return None
    except Exception as e:
         st.error(f"An unexpected error occurred during data loading: {e}")
         st.text(traceback.format_exc())
         return None


# --- Call Load Data Function ---
# Call the function after defining it and after set_page_config
merchants = load_data()

def get_performance_insights(metric, merchant_value, avg_value, performance_status, industry, store_type):
    """Generate actionable insights for below-average performance metrics."""
    insights = []
    
    # Only provide insights for below-average performance
    if "❌" not in performance_status:
        return insights
    
    # Calculate performance gap for contextualized advice
    gap_pct = abs((merchant_value - avg_value) / avg_value * 100)
    is_large_gap = gap_pct > 20  # Significant underperformance
    is_small_gap = gap_pct < 10  # Minor underperformance
    
    metric_lower = metric.lower()
    
    if "avg txn value" in metric_lower:
        # Store type specific recommendations for transaction value
        if store_type == "Mall":
            if industry == "Restaurant":
                insights = [
                    "💡 Leverage Mall Footfall for Higher Orders",
                    "• Create mall-exclusive combo meals and family portions",
                    "• Partner with cinema/shops for meal + entertainment packages", 
                    "• Use digital menu boards to showcase premium options",
                    "• Target families with shareable platters and desserts"
                ]
            elif industry == "Retail":
                insights = [
                    "💡 Maximize Mall Shopping Experience",
                    "• Create attractive window displays with premium products",
                    "• Bundle trending items for mall shoppers",
                    "• Offer 'mall exclusive' product collections",
                    "• Use mall events to showcase higher-value items"
                ]
            elif industry == "Fashion":
                insights = [
                    "💡 Position as Premium Mall Destination",
                    "• Curate exclusive collections for mall demographics",
                    "• Create complete outfit displays in windows",
                    "• Offer personal styling services for mall shoppers",
                    "• Partner with other mall stores for cross-promotions"
                ]
        
        elif store_type == "Street Front":
            if industry == "Restaurant":
                insights = [
                    "💡 Build Neighborhood Value Perception",
                    "• Introduce 'chef's special' higher-value dishes",
                    "• Create loyalty programs for repeat customers",
                    "• Offer home delivery with minimum order values",
                    "• Add beverages and sides to increase average order"
                ]
            elif industry == "Retail":
                insights = [
                    "💡 Become the Go-To Local Store",
                    "• Stock premium local and organic products",
                    "• Create convenience bundles (breakfast, dinner kits)",
                    "• Offer credit facilities for regular customers",
                    "• Focus on quality over quantity positioning"
                ]
            elif industry == "Fashion":
                insights = [
                    "💡 Establish Local Fashion Authority",
                    "• Curate trending styles for local demographics",
                    "• Offer alteration services to justify higher prices",
                    "• Create seasonal collections for local events",
                    "• Build relationships with local influencers"
                ]
        
        elif store_type == "Standalone":
            if industry == "Restaurant":
                insights = [
                    "💡 Create Destination Dining Experience",
                    "• Develop signature dishes to justify premium pricing",
                    "• Create ambiance that supports higher ticket sizes",
                    "• Offer special occasion packages and catering",
                    "• Build reputation through food quality and service"
                ]
            elif industry == "Retail":
                insights = [
                    "💡 Differentiate Through Specialization",
                    "• Focus on niche, high-quality product categories",
                    "• Offer expert consultation and recommendations",
                    "• Create bulk buying options for families",
                    "• Position as premium alternative to chain stores"
                ]
            elif industry == "Fashion":
                insights = [
                    "💡 Build Boutique Brand Identity",
                    "• Curate unique, hard-to-find fashion pieces",
                    "• Offer personalized styling and fitting services",
                    "• Create exclusive collections or designer partnerships",
                    "• Focus on quality and craftsmanship messaging"
                ]
                
    elif "daily txn count" in metric_lower:
        # Store type specific recommendations for customer count
        if store_type == "Mall":
            if industry == "Restaurant":
                insights = [
                    "🎯 Capture Mall Traffic Effectively",
                    "• Position staff at entrance during peak mall hours",
                    "• Create quick-service options for busy shoppers",
                    "• Offer mall walker discounts during off-peak hours",
                    "• Partner with movie theaters for pre/post show meals"
                ]
            elif industry == "Retail":
                insights = [
                    "🎯 Maximize Mall Visibility",
                    "• Create eye-catching storefront displays",
                    "• Participate in mall-wide sales and events",
                    "• Offer exclusive mall shopper discounts",
                    "• Use digital signage to attract passing shoppers"
                ]
            elif industry == "Fashion":
                insights = [
                    "🎯 Attract Mall Fashion Shoppers",
                    "• Create seasonal window displays with trending styles",
                    "• Offer styling sessions during peak mall hours",
                    "• Partner with beauty salons for complete makeovers",
                    "• Host mini fashion shows during mall events"
                ]
                
        elif store_type == "Street Front":
            if industry == "Restaurant":
                insights = [
                    "🎯 Become the Neighborhood Favorite",
                    "• Offer breakfast and tea service for morning commuters",
                    "• Create loyalty programs for office workers nearby",
                    "• Extend operating hours to capture dinner crowd",
                    "• Use social media to announce daily specials"
                ]
            elif industry == "Retail":
                insights = [
                    "🎯 Increase Local Foot Traffic",
                    "• Improve street-facing signage and visibility",
                    "• Stock daily essentials to encourage frequent visits",
                    "• Create seasonal promotional displays",
                    "• Build relationships with nearby office/residential areas"
                ]
            elif industry == "Fashion":
                insights = [
                    "🎯 Build Local Fashion Community",
                    "• Host neighborhood fashion events and trunk shows",
                    "• Create referral programs for existing customers",
                    "• Offer home delivery for busy local customers",
                    "• Partner with local gyms, salons for cross-promotion"
                ]
                
        elif store_type == "Standalone":
            if industry == "Restaurant":
                insights = [
                    "🎯 Drive Destination Traffic",
                    "• Invest in online presence and food delivery apps",
                    "• Create social media buzz with food photography",
                    "• Offer catering services to local businesses/events",
                    "• Build word-of-mouth through exceptional service"
                ]
            elif industry == "Retail":
                insights = [
                    "🎯 Expand Customer Reach",
                    "• Develop online presence for product discovery",
                    "• Offer home delivery for bulk purchases",
                    "• Create customer referral incentive programs",
                    "• Partner with local businesses for B2B sales"
                ]
            elif industry == "Fashion":
                insights = [
                    "🎯 Build Fashion Destination Appeal",
                    "• Develop strong social media presence with styling tips",
                    "• Offer appointment-based personal shopping",
                    "• Create email newsletters with fashion trends",
                    "• Host exclusive preview events for new collections"
                ]
            
    elif "refund rate" in metric_lower:
        # Store type specific recommendations for refund rate
        if store_type == "Mall":
            insights = [
                "⚠️ Maintain Mall Standards",
                "• Implement quality checks before displaying products",
                "• Train staff on product knowledge for better recommendations",
                "• Create clear size/fit guides for customer education",
                "• Offer exchange policies instead of full refunds when possible"
            ]
        elif store_type == "Street Front":
            insights = [
                "⚠️ Build Local Trust and Reliability",
                "• Focus on product quality over variety",
                "• Offer trial periods for regular customers",
                "• Provide detailed product demonstrations",
                "• Build personal relationships to understand customer needs"
            ]
        elif store_type == "Standalone":
            insights = [
                "⚠️ Ensure Premium Quality Standards",
                "• Implement strict quality control processes",
                "• Offer detailed consultations before purchase",
                "• Provide comprehensive after-sales support",
                "• Create customer feedback systems for continuous improvement"
            ]
    
    elif "repeat customer rate" in metric_lower or "repeat" in metric_lower:
        # Store type specific recommendations for customer retention
        if store_type == "Mall":
            if industry == "Restaurant":
                insights = [
                    "🔄 Build Mall Dining Loyalty",
                    "• Create loyalty cards with mall shopping integration",
                    "• Offer special discounts for movie + meal combos",
                    "• Remember regular customers' preferences and orders",
                    "• Create family-friendly loyalty programs for weekend visits"
                ]
            elif industry == "Retail":
                insights = [
                    "🔄 Encourage Mall Return Visits",
                    "• Create membership programs with exclusive mall benefits",
                    "• Send SMS alerts for new arrivals and sales",
                    "• Offer personal shopping services for repeat customers",
                    "• Partner with other mall stores for cross-loyalty rewards"
                ]
            elif industry == "Fashion":
                insights = [
                    "🔄 Develop Fashion Loyalty Community",
                    "• Create VIP styling sessions for repeat customers",
                    "• Send personalized style recommendations via SMS",
                    "• Offer early access to new collections",
                    "• Create seasonal fashion events for loyal customers"
                ]
        
        elif store_type == "Street Front":
            if industry == "Restaurant":
                insights = [
                    "🔄 Become the Neighborhood Regular Spot",
                    "• Remember customers' names and usual orders",
                    "• Create punch cards for free meals after X visits",
                    "• Offer home delivery loyalty discounts",
                    "• Send SMS updates about daily specials to regulars"
                ]
            elif industry == "Retail":
                insights = [
                    "🔄 Build Local Customer Relationships",
                    "• Maintain customer purchase history for recommendations",
                    "• Offer credit facilities for trusted repeat customers",
                    "• Send SMS reminders for regular purchase items",
                    "• Create neighborhood customer referral programs"
                ]
            elif industry == "Fashion":
                insights = [
                    "🔄 Establish Local Fashion Authority",
                    "• Keep customer size and style preferences on file",
                    "• Offer alteration services for repeat customers",
                    "• Send style trend updates to loyal customers",
                    "• Create local fashion influencer community programs"
                ]
        
        elif store_type == "Standalone":
            if industry == "Restaurant":
                insights = [
                    "🔄 Create Destination Dining Loyalty",
                    "• Implement digital loyalty program with mobile tracking",
                    "• Create exclusive chef's table experiences for VIP customers",
                    "• Offer catering discounts for repeat customers",
                    "• Send personalized meal recommendations based on history"
                ]
            elif industry == "Retail":
                insights = [
                    "🔄 Build Specialized Customer Base",
                    "• Create expert consultation relationships with repeat buyers",
                    "• Offer bulk purchase loyalty discounts",
                    "• Maintain detailed customer preference databases",
                    "• Provide exclusive access to premium products for loyal customers"
                ]
            elif industry == "Fashion":
                insights = [
                    "🔄 Develop Boutique Customer Relationships",
                    "• Offer personal styling consultations for repeat customers",
                    "• Create exclusive fashion preview events",
                    "• Maintain detailed style profiles for each customer",
                    "• Provide wardrobe consultation services for loyal clients"
                ]
    
    # Add performance gap specific advice
    if is_large_gap:
        if store_type == "Mall":
            insights.append("• Consider immediate staff training and process review")
        elif store_type == "Street Front":
            insights.append("• Focus on building stronger customer relationships")
        else:  # Standalone
            insights.append("• Reassess positioning and target customer strategy")
    elif is_small_gap:
        insights.append("• Small adjustments can close this gap quickly")
    
    return insights

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

        We provide **two types of comparisons** to give you complete market insights:

        #### 1. Local Market Analysis 📍
        - Compares you with businesses in **your exact area** (same pincode)
        - **Same industry** (Restaurant vs Restaurant, Retail vs Retail)
        - Shows how you perform against **geographic neighbors**
        - Helps understand **local market dynamics**
        - **Example**: Your restaurant vs other restaurants in the same pincode

        #### 2. Industry Cluster Analysis 🔄
        - Uses **machine learning clustering** within your industry
        - Groups businesses with **similar performance patterns** across India
        - Compares against businesses that operate **similarly to you**
        - **Location-independent**: Finds similar performers anywhere
        - **Performance-based**: Groups by transaction patterns, customer behavior, efficiency

        #### Why Both Matter
        - **Local Market**: Shows your position in your immediate market
            - "Am I competitive in my neighborhood?"
            - Local pricing, local customer expectations
            - Direct geographic competition
        
        - **Industry Cluster**: Shows your potential and best practices
            - "What can I achieve with businesses like mine?"
            - Learn from similar performers across India
            - Industry-specific growth strategies

        #### Example Comparison
        - **Local**: Your café vs 3 other cafés in Bandra (400050)
        - **Cluster**: Your café vs 47 similar-performing cafés across Mumbai, Delhi, Bangalore with comparable customer patterns

        #### Analysis Benefits
        - **Immediate**: Fix local competitive issues
        - **Strategic**: Learn from industry best practices
        - **Actionable**: Get location-specific + performance-based insights
        - **Comprehensive**: Complete market view (local + industry-wide)
        """)
    
    # Add business tools guide
    with st.expander("📚 Business Tools Guide", expanded=False):
        # Display the concise guide suitable for sidebar
        guide_text = display_link_guide()
        st.markdown(guide_text)
        
        st.markdown("---")
        st.markdown("💡 **Tip:** When you see links in your insights, they're specifically chosen based on your business needs and performance gaps!")
    
    st.divider()

# --- Main Area ---
if merchant_id:
    # Get merchant data
    merchant_row, comparison_df_local, comparison_df_cluster, local_competitors, cluster_peers, cluster_averages = get_comparison_data(merchant_id, merchants)

    if merchant_row is not None:
        # Create a header with merchant info
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.markdown(f"### {merchant_row.get('industry', 'Business')} Analysis")
            st.markdown(f"**Location:** {merchant_row.get('city', 'N/A')} | **Store Type:** {merchant_row.get('store_type', 'N/A')}")
        
        # Quick stats in cards
        with col2:
            # Find avg transaction value comparison for delta calculation
            avg_txn_delta = None
            if comparison_df_cluster is not None:
                avg_txn_row = comparison_df_cluster[comparison_df_cluster['Metric'] == 'Avg Txn Value']
                if not avg_txn_row.empty:
                    merchant_raw = avg_txn_row.iloc[0]['Merchant Raw']
                    cluster_raw = avg_txn_row.iloc[0]['Cluster Raw']
                    if cluster_raw > 0:
                        avg_txn_delta = f"{((merchant_raw - cluster_raw) / cluster_raw * 100):.1f}%"
            
            st.metric(
                "Average Transaction",
                f"₹{merchant_row.get('avg_txn_value', 0):.2f}",
                avg_txn_delta
            )
        
        with col3:
            # Find daily transaction count comparison for delta calculation
            daily_txn_delta = None
            if comparison_df_cluster is not None:
                daily_txn_row = comparison_df_cluster[comparison_df_cluster['Metric'] == 'Daily Txn Count']
                if not daily_txn_row.empty:
                    merchant_raw = daily_txn_row.iloc[0]['Merchant Raw']
                    cluster_raw = daily_txn_row.iloc[0]['Cluster Raw']
                    if cluster_raw > 0:
                        daily_txn_delta = f"{((merchant_raw - cluster_raw) / cluster_raw * 100):.1f}%"
            
            st.metric(
                "Daily Customers",
                f"{merchant_row.get('daily_txn_count', 0)}",
                daily_txn_delta
            )

        # Main content in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Key Insights", "📈 Performance", "📋 Details"])

        with tab1:
            # Quick Insights
            try:
                st.markdown("### 📊 Key Insights")
                
                # Add debug button to clear cache
                col_debug1, col_debug2 = st.columns([3, 1])
                with col_debug2:
                    if st.button("🔄 Regenerate", help="Clear cache and regenerate insights"):
                        st.session_state.crisp_insights = None
                        st.session_state.impact_data = None
                        st.cache_data.clear()
                        st.rerun()
                
                # Initialize session state for insights if not exists
                if 'crisp_insights' not in st.session_state:
                    st.session_state.crisp_insights = None
                if 'impact_data' not in st.session_state:
                    st.session_state.impact_data = None
                
                # Only generate insights if they don't exist in session state
                if st.session_state.crisp_insights is None:
                    with st.spinner("Generating insights..."):
                        try:
                            print(f"DEBUG: Starting insight generation for merchant {merchant_id}")
                            st.session_state.crisp_insights, st.session_state.impact_data = generate_crisp_insights(
                                merchant_row,
                                comparison_df_local,
                                comparison_df_cluster,
                                cluster_peers,
                                cluster_averages
                            )
                            print(f"DEBUG: Insight generation completed successfully")
                            print(f"DEBUG: Impact data received: {st.session_state.impact_data}")
                        except Exception as e:
                            print(f"DEBUG: Exception during insight generation: {e}")
                            print(f"DEBUG: Exception traceback: {traceback.format_exc()}")
                            st.error(f"Error generating insights: {e}")
                            st.session_state.crisp_insights = "Error generating insights"
                            st.session_state.impact_data = None
                
                # Format insights for display
                formatted_insights = format_insights_for_display(st.session_state.crisp_insights, st.session_state.impact_data)
                
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
                                
                                # Add bars for current, expected, and cluster average
                                x_labels = ['Current', 'Cluster Avg']
                                y_values = [data['current'], data['cluster_avg']]
                                colors = ['#4dabf7', '#e67700']
                                
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
                                details_html += f"<span style='color: #e67700;' title='Average of businesses with similar performance patterns in your industry'>●</span> Cluster Average: {data['cluster_avg']:.2f}"
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
                st.markdown("### Local Market (Same Area)")
                if comparison_df_local is not None:
                    for _, row in comparison_df_local.iterrows():
                        status_class = "status-good" if "✅" in str(row['Performance']) else "status-warning" if "⚠️" in str(row['Performance']) else "status-bad"
                        
                        # Display performance section - using local data
                        impact_pct = ((row['Merchant Raw'] - row['Local Raw']) / row['Local Raw'] * 100) if row['Local Raw'] != 0 else 0
                        st.markdown(f"""
                        <div class="performance-section">
                            <div class="performance-title">{row['Metric']}</div>
                            <div class="performance-metrics">
                                <div class="performance-metric">
                                    <p>Your Value: <span class="performance-value">{row['Merchant Value']}</span></p>
                                    <p>Local Area Average: <span class="performance-average">{row['Local Avg']}</span></p>
                                </div>
                                <div class="performance-metric">
                                    <p class="performance-status {status_class}">{row['Performance']}</p>
                                </div>
                            </div>
                            <div class="performance-details">
                                <div class="performance-detail-item">
                                    <span class="performance-detail-value">Impact:</span> {impact_pct:.1f}% vs local market
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display insights separately using Streamlit components
                        insights = get_performance_insights(
                            row['Metric'], 
                            row['Merchant Raw'], 
                            row['Local Raw'], 
                            row['Performance'],
                            merchant_row.get('industry', ''),
                            merchant_row.get('store_type', '')
                        )
                        
                        if insights:
                            st.markdown(f"""
                            <div class="performance-insight">
                                <div class="insight-title">{insights[0]}</div>
                                <div class="insight-content">
                            """, unsafe_allow_html=True)
                            
                            # Display each insight item as a separate line
                            for insight_item in insights[1:]:
                                st.markdown(f"<div style='color: #e0e0e0; font-size: 0.85em; margin: 2px 0;'>{insight_item}</div>", unsafe_allow_html=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                else:
                    st.info("No local competitors found in your area with the same business type.")

            with col2:
                st.markdown("### Industry Cluster (Similar Performance)")
                if comparison_df_cluster is not None:
                    for _, row in comparison_df_cluster.iterrows():
                        status_class = "status-good" if "✅" in str(row['Performance']) else "status-warning" if "⚠️" in str(row['Performance']) else "status-bad"
                        
                        # Display performance section
                        impact_pct = ((row['Merchant Raw'] - row['Cluster Raw']) / row['Cluster Raw'] * 100) if row['Cluster Raw'] != 0 else 0
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
                                    <span class="performance-detail-value">Impact:</span> {impact_pct:.1f}% vs cluster average
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display insights separately using Streamlit components
                        insights = get_performance_insights(
                            row['Metric'], 
                            row['Merchant Raw'], 
                            row['Cluster Raw'], 
                            row['Performance'],
                            merchant_row.get('industry', ''),
                            merchant_row.get('store_type', '')
                        )
                        
                        if insights:
                            st.markdown(f"""
                            <div class="performance-insight">
                                <div class="insight-title">{insights[0]}</div>
                                <div class="insight-content">
                            """, unsafe_allow_html=True)
                            
                            # Display each insight item as a separate line
                            for insight_item in insights[1:]:
                                st.markdown(f"<div style='color: #e0e0e0; font-size: 0.85em; margin: 2px 0;'>{insight_item}</div>", unsafe_allow_html=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                else:
                    st.info("No similar performing businesses found in your industry cluster.")

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
                "📍 Local Market Report",
                comparison_df_local.to_csv(index=False).encode('utf-8'),
                f"{merchant_id}_local_market_comparison.csv",
                "text/csv"
            )
        
        if comparison_df_cluster is not None:
            col_dl2.download_button(
                "🔄 Industry Cluster Report",
                comparison_df_cluster.to_csv(index=False).encode('utf-8'),
                f"{merchant_id}_industry_cluster_analysis.csv",
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