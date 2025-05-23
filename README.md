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
   
   **Option 1: Using Environment Variable (Recommended)**
   - Get a Google API key for Gemini AI (see detailed steps below)
   - Set the environment variable:
     ```bash
     # On macOS/Linux:
     export GOOGLE_API_KEY="your-api-key-here"
     
     # On Windows:
     set GOOGLE_API_KEY=your-api-key-here
     ```
   
   **Option 2: Using Streamlit Secrets**
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your API key:
     ```toml
     GOOGLE_API_KEY = "your-api-key-here"
     ```

## Google API Key Setup (Detailed Steps)

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Click "Create Project" or select an existing project
4. Give your project a name (e.g., "merchant-insights")
5. Click "Create"

### Step 2: Enable the Gemini API
1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Generative Language API" or "Gemini API"
3. Click on the API and select "Enable"
4. Wait for the API to be enabled

### Step 3: Create API Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the generated API key
4. (Optional but recommended) Click "Restrict Key" to add restrictions:
   - Under "API restrictions", select "Restrict key"
   - Choose "Generative Language API" from the list
   - Click "Save"

### Step 4: Set Up the API Key in Your Project
Choose one of the following methods:

**Method 1: Environment Variable (Recommended for development)**
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export GOOGLE_API_KEY="AIzaSyBzbDCXwtfttdY5KHU3Z4bN-CQFKTRCPD4"

# Reload your shell configuration
source ~/.zshrc  # For zsh users
# OR
source ~/.bashrc  # For bash users

# Or set temporarily for current session:
export GOOGLE_API_KEY="your-actual-api-key-here"
```

**Method 2: Streamlit Secrets (Good for deployment)**
1. Create a `.streamlit` directory in your project root
2. Create a `secrets.toml` file inside it:
```toml
GOOGLE_API_KEY = "your-actual-api-key-here"
```

### Step 5: Verify Setup
1. Start the application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Select a merchant and try generating insights
3. If you see "Google API Key not configured" error, double-check your setup

### API Key Security Tips
- ⚠️ **Never commit your API key to version control**
- 🔒 Add `.streamlit/secrets.toml` to your `.gitignore` file
- 🔄 Rotate your API key regularly
- 📊 Monitor your API usage in Google Cloud Console

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