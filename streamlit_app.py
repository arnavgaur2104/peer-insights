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
# Move this to the top, right after imports
st.set_page_config(page_title="Merchant AI Insights v2", page_icon="🧠", layout="wide") # Use wide layout

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
st.title("🧠 AI-Powered Merchant Insight Generator") # Now this is fine

# Stop execution if data loading failed (merchants is critical)
if merchants is None:
    st.warning("Stopping execution: Merchant data loading failed.")
    st.stop()
if competitors is None:
     st.warning("Competitor data loading failed. Comparisons will not be available.")
     # Allow app to continue but comparisons might fail later


# --- Sidebar ---
st.sidebar.header("Select Merchant")
# Initialize merchant_id
merchant_id = None
merchant_id_list = []
if not merchants.empty:
    merchant_id_list = merchants['merchant_id'].unique().tolist() # Use unique IDs

if merchant_id_list:
    merchant_id = st.sidebar.selectbox(
        "Merchant ID",
        sorted(merchant_id_list), # Sort IDs for better usability
        index=0 # Default to the first merchant
    )
else:
    st.sidebar.error("No unique merchant IDs found in the data file.")

st.sidebar.header("Configuration")
show_simple_insights = st.sidebar.checkbox("Show Simple Rule-Based Insights", value=False)

# --- Main Area ---
if merchant_id:
    # --- Get Data using the new function ---
    merchant_row, comparison_df_local, comparison_df_cluster = None, None, None # Initialize variables
    local_competitors, cluster_peers, cluster_averages = None, None, None
    ai_insights = "Insights calculation not reached yet." # Default insight text

    # Ensure competitors df is available before calling comparison function
    if competitors is None:
         st.error("Cannot perform comparison as competitor data failed to load.")
    else:
        try:
            print(f"Calling get_comparison_data for {merchant_id}...") # Also print to terminal

            (merchant_row, comparison_df_local, comparison_df_cluster,
             local_competitors, cluster_peers, cluster_averages) = get_comparison_data(merchant_id, merchants, competitors)

            print("Finished get_comparison_data.") # Also print to terminal

        except Exception as e:
            st.error(f"An error occurred during get_comparison_data or initial processing: {e}")
            st.text(traceback.format_exc()) # Display traceback in the app
            print(f"ERROR in streamlit_app.py main block: {e}") # Print error to terminal
            print(traceback.format_exc())
            merchant_row = None # Ensure merchant_row is None if comparison fails

    # Add info button about clustering and competitors AFTER we have merchant data
    if merchant_row is not None:
        with st.sidebar.expander("ℹ️ How are merchants grouped?", expanded=False):
            st.markdown(f"""
            ### Understanding Your Business Analysis
            
            We analyze your business in two ways:
            
            **1. Local Competitors** 👥
            - Businesses in your area (same pincode)
            - Same industry as you
            - Direct competition for customers
            
            **2. Similar Businesses (Clusters)** 🔄
            - Businesses with similar performance patterns
            - May be in different locations
            - Share similar characteristics like:
                - Transaction patterns
                - Customer behavior
                - Operational efficiency
            
            **Example for {merchant_row.get('industry', 'your industry')}:**
            - Local Competitors: Other {merchant_row.get('industry', 'your industry')} stores in your neighborhood
            - Cluster Peers: {merchant_row.get('industry', 'your industry')} businesses with similar:
                - Average transaction value
                - Daily customer count
                - Refund rates
                - Operational efficiency
            
            This dual analysis helps you:
            - Compare with immediate competition
            - Learn from similar businesses
            - Identify unique opportunities
            """)

    # --- Check if merchant_row was successfully retrieved and processed ---
    if merchant_row is None:
        st.error(f"Merchant {merchant_id} data could not be fully processed or found (merchant_row is None after get_comparison_data). Check data files, IDs, and compare_merchants.py logic.")
    else:
        # --- Display Section (Main Content) ---
        st.header(f"Analysis for Merchant: {merchant_id}")
        st.markdown(f"**Industry:** {merchant_row.get('industry', 'N/A')} | **Store Type:** {merchant_row.get('store_type', 'N/A')} | **Location:** {merchant_row.get('city', 'N/A')}")

        # Display logic using columns
        col1, col2 = st.columns([1, 2]) # Adjust column ratios as needed

        with col1:
            st.subheader("Merchant Profile")
            # Display selected details instead of raw JSON
            profile_disp = {k: v for k, v in merchant_row.items() if k not in ['cluster', 'pincode']} # Hide cluster/pincode here
            try:
                 profile_series = pd.Series(profile_disp, name="Value")
                 # Convert the 'Value' column to string type for display to fix ArrowTypeError
                 st.dataframe(profile_series.astype(str), use_container_width=True)
            except Exception as e:
                 st.error(f"Error displaying merchant profile: {e}")
                 st.write(profile_disp) # Fallback to writing dict

            st.subheader("Cluster Peers")
            if cluster_peers is not None and not cluster_peers.empty:
                 st.metric(label="Peers in Same Cluster", value=len(cluster_peers))
                 # Show some basic info about peers
                 st.dataframe(cluster_peers[['merchant_id', 'city', 'store_type']].head(), use_container_width=True)
            elif merchant_row.get('cluster', -1) == -1:
                 st.warning("Clustering could not be performed for this merchant.")
            else: # Merchant has a cluster, but no peers found
                 st.info("No other merchants found in the same cluster.")

        with col2:
            st.subheader("📊 Performance Comparisons")

            # Check if competitor data was loaded for tabs
            if competitors is None:
                 st.warning("Competitor data was not loaded, cannot show comparisons.")
            else:
                 tab1, tab2 = st.tabs(["Compare vs Local Competitors", "Compare vs Cluster Peers"])

                 with tab1:
                     if comparison_df_local is not None and not comparison_df_local.empty:
                         st.dataframe(comparison_df_local, use_container_width=True, hide_index=True)
                     elif local_competitors is not None and local_competitors.empty:
                          st.info("No local competitors found based on Pincode and Industry.")
                     else: # comparison_df_local is None or local_competitors is None
                          st.warning("Local comparison data could not be generated (competitors might be missing or an error occurred).")


                 with tab2:
                     if comparison_df_cluster is not None and not comparison_df_cluster.empty:
                         st.dataframe(comparison_df_cluster, use_container_width=True, hide_index=True)
                     elif merchant_row.get('cluster', -1) == -1:
                          st.warning("Clustering failed or was not performed, cannot show cluster comparison.")
                     elif cluster_peers is not None and cluster_peers.empty:
                          st.info("No other merchants found in the same cluster for comparison.")
                     else: # comparison_df_cluster is None or cluster_peers is None
                          st.warning("Cluster comparison data could not be generated (clustering might have failed or an error occurred).")


        # --- AI Insights Section ---
        st.divider()
        st.header("💡 Quick Insights")

        # Initialize session state for detailed insights if not exists
        if 'detailed_insights' not in st.session_state:
            st.session_state.detailed_insights = None
        if 'show_detailed_analysis' not in st.session_state:
            st.session_state.show_detailed_analysis = False

        if merchant_row is not None:
            try:
                # Generate quick insights by default
                quick_insights = generate_quick_insights(
                    merchant_row,
                    comparison_df_local,
                    comparison_df_cluster,
                    cluster_peers,
                    cluster_averages
                )
                
                # Display quick insights in a clean, number-focused format
                st.markdown(quick_insights)
                
                # Add toggle button for detailed analysis
                if st.button("📊 Toggle Detailed Analysis", type="secondary"):
                    st.session_state.show_detailed_analysis = not st.session_state.show_detailed_analysis
                
                # Show detailed analysis only if toggled on
                if st.session_state.show_detailed_analysis:
                    st.divider()
                    st.subheader("📊 Detailed Analysis")
                    
                    # Generate detailed insights only if not already generated
                    if st.session_state.detailed_insights is None:
                        with st.spinner("Generating detailed analysis..."):
                            st.session_state.detailed_insights = generate_advanced_ai_insights(
                                merchant_row,
                                comparison_df_local,
                                comparison_df_cluster,
                                cluster_peers,
                                cluster_averages
                            )
                    
                    # Display the stored detailed insights
                    st.markdown(st.session_state.detailed_insights)
            
            except Exception as insight_err:
                st.error(f"Error generating insights: {insight_err}")


        # Optionally display simple insights if checkbox is ticked
        if show_simple_insights:
            st.subheader("Rule-Based Insights (Simple)")
            if comparison_df_local is not None:
                try:
                     simple_insights = generate_insights(comparison_df_local) # Use local comparison for simple rules
                     if simple_insights:
                          for i in simple_insights:
                               st.info(i) # Use info boxes for simple insights
                     else:
                          st.info("No simple insights generated (performance might be good based on local comparison).")
                except NameError:
                     st.error("The function `generate_insights` seems to be missing or not imported correctly from insights_engine.py.")
                     print("ERROR: `generate_insights` not found during call.")
                except Exception as simple_insight_err:
                     st.error(f"Error generating simple insights: {simple_insight_err}")
                     print(f"ERROR generating simple insights: {simple_insight_err}\n{traceback.format_exc()}")
            else:
                 st.warning("Cannot generate simple insights as local comparison data is unavailable.")


        # --- Download Section ---
        st.divider()
        st.subheader("⬇️ Download Data")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        # Ensure dataframes exist before attempting download button creation
        if comparison_df_local is not None and not comparison_df_local.empty:
             try:
                  download_local = comparison_df_local.to_csv(index=False).encode('utf-8')
                  col_dl1.download_button("Local Comparison CSV", download_local, f"{merchant_id}_local_comparison.csv", "text/csv", key="dl_local")
             except Exception as e:
                  col_dl1.error(f"Failed to create local CSV: {e}")
        else:
             col_dl1.info("No local comparison data to download.")

        if comparison_df_cluster is not None and not comparison_df_cluster.empty:
             try:
                  download_cluster = comparison_df_cluster.to_csv(index=False).encode('utf-8')
                  col_dl2.download_button("Cluster Comparison CSV", download_cluster, f"{merchant_id}_cluster_comparison.csv", "text/csv", key="dl_cluster")
             except Exception as e:
                  col_dl2.error(f"Failed to create cluster CSV: {e}")
        else:
             col_dl2.info("No cluster comparison data to download.")

        # Ensure ai_insights string exists before trying to download
        insights_text_to_download = "Insights could not be generated or were not calculated."
        if 'ai_insights' in locals() and isinstance(ai_insights, str):
             insights_text_to_download = ai_insights
        try:
             col_dl3.download_button("Download AI Insights TXT", insights_text_to_download, f"{merchant_id}_insights.txt", "text/plain", key="dl_insights")
        except Exception as e:
             col_dl3.error(f"Failed to create insights TXT: {e}")


# --- Fallback message if no merchant is selected ---
elif not merchant_id and merchant_id_list: # Only show if IDs exist but none selected
    st.info("Select a Merchant ID from the sidebar to begin analysis.")
elif not merchant_id_list:
     # Message if no merchant IDs were loaded at all
     st.error("No merchant data available to select from.")
# The case where data loading fails critically is handled earlier by st.stop()