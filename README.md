# Merchant Insights POC

An AI-powered analytics application that provides actionable insights for merchants based on their payment data and market position.

## Features

- 📊 Performance comparison with local competitors
- 🔄 Similar business analysis through clustering
- 💡 AI-generated actionable insights
- 📈 Detailed metrics analysis
- 🎯 Time-based business recommendations
- 📱 User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google API key for Gemini AI

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd merchant-insights-poc
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Google API key:
   - Get a Google API key for Gemini AI
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your API key:
   ```toml
   GOOGLE_API_KEY = "your-api-key-here"
   ```

## Running the Application

1. Generate sample data:
```bash
python generate_data.py
```

2. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

## Project Structure

- `streamlit_app.py`: Main application file
- `generate_data.py`: Sample data generation
- `compare_merchants.py`: Merchant comparison logic
- `insights_engine.py`: AI insights generation
- `data/`: Directory for merchant and competitor data

## Usage Guide

1. **Select a Merchant**
   - Choose a merchant ID from the sidebar
   - View basic merchant profile and metrics

2. **View Comparisons**
   - Compare with local competitors
   - Analyze performance against similar businesses
   - Review detailed metrics

3. **Get Insights**
   - View quick insights for immediate actions
   - Access detailed analysis for strategic planning
   - Explore time-based recommendations

4. **Download Data**
   - Export comparison data
   - Save insights for future reference

## Understanding the Analysis

The application provides two types of comparisons:

1. **Local Competitors** 👥
   - Businesses in your area
   - Same industry
   - Direct competition

2. **Similar Businesses (Clusters)** 🔄
   - Businesses with similar patterns
   - May be in different locations
   - Share similar characteristics

## Troubleshooting

1. **Data Generation Issues**
   - Ensure Python 3.8+ is installed
   - Check write permissions in the data directory
   - Verify all required packages are installed

2. **API Key Issues**
   - Verify the API key is correctly set in secrets.toml
   - Check API key permissions and quota
   - Ensure the key is for Gemini AI

3. **Application Errors**
   - Check the terminal for error messages
   - Verify all required files are present
   - Ensure data files are properly generated