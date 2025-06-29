import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import folium
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from folium.plugins import HeatMap
import streamlit.components.v1 as components
from datetime import datetime
import joblib
import pickle
from streamlit_folium import st_folium



def set_background_and_styles():
    st.markdown(
    """
    <style>
    /* Overall App Background */
    .stApp {
        background-color: #f0f4f0;
        font-family: 'Arial', sans-serif;
    }
    
    /* Main Content Container */
    .main .block-container {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    
    /* Headings with Better Contrast */
    h1, h2, h3 {
        color: #154734; /* Deep forest green for excellent contrast */
        text-align: center;
        font-weight: bold;
    }
    
    /* Markdown Text */
    .markdown-text-container {
        color: #2c3e50;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #2E8B57, #1a5a32);
        color: white;
    }
    
    /* Sidebar Navigation */
    .stRadio > div > label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1a5a32;
        transform: scale(1.05);
    }
    
    /* Data Frames */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        color: #2c3e50;
    }
    
    /* Text Color */
    p, li {
        color: #2c3e50;
    }
    </style>
    """, 
    unsafe_allow_html=True
)



# Navigation Menu
def create_sidebar_menu():
    st.sidebar.title(" Forest Fire Risk Prediction")
    menu = {
        "🏡 Home": "home",
        "🗺️ Interactive Map": "map",
        "📊 Data Visualization": "visualization",
        "🔍 Model Prediction": "prediction",
        "📂 Datasets": "datasets",
        "💬 Feedback": "feedback"
    }
    choice = st.sidebar.radio("Navigation", list(menu.keys()))
    return menu[choice]

# Home Page
def home_page():
    st.title("Nepal Forest Fire Risk Prediction System ")
    
    st.markdown("""
    ## Project Overview
    
    ### 🌿 Our Mission
    Develop an advanced predictive system to identify and mitigate forest fire risks in Nepal's diverse ecological landscapes.
    
    ### 🔬 Key Components
    - **Climate Data Analysis**: Leveraging district-wise climate information
    - **Satellite Fire Detection**: Using NASA LANCE FIRMS data
    - **Machine Learning Prediction**: Advanced risk assessment models
    
    ### 🌍 Why Nepal?
    - Nepal's forests cover approximately 44.74% of its total land area
    - Climate change and increasing temperatures pose significant fire risks
    - Early detection and prediction can save ecological and economic resources
    
    ### 📊 Data Sources
    - Climate Data: District-wise Climate Data for Nepal
    - Fire Detection: NASA LANCE FIRMS Satellite Data
    
    ### 🎯 Project Goals
    1. Predict forest fire probabilities
    2. Provide early warning systems
    3. Support forest conservation efforts
    """)

# Interactive Map Page
def interactive_map_page():
    st.title("🗺️ Interactive Climate And Forest Fire Maps")
    
    try:
        
        # Load climate data
        file_path = "data/processed/flitered_climate_data.csv"
        climate_data = pd.read_csv(file_path)

        # Create YearMonth column for animation frame
        climate_data['YearMonth'] = climate_data['YEAR'].astype(str) + '-' + climate_data['MONTH'].astype(str).str.zfill(2)

       
        # Map Selection
        map_type = st.selectbox("Select Map Type", [
            "Climate Variables", 
            "Fire Variables"
        ])

        if map_type == "Climate Variables":
            st.markdown("### Climate Variable Visualization\nExplore how different climate parameters vary across Nepal's districts.")

            climate_var = st.selectbox("Select Climate Variable", [
                "Maximum Temperature", 
                "Humidity", 
                "Precipitation", 
                "Wind Speed"
            ])

            if climate_var == "Maximum Temperature":
                st.markdown("### Monthly Maximum Temperature Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See the temperature values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                df_filtered = climate_data.copy()
                min_temp = df_filtered["MaxTemp"].min()
                df_filtered["SizeTemp"] = df_filtered["MaxTemp"] + abs(min_temp) + 1

                fig = px.scatter_mapbox(
                    df_filtered,
                    lat="LAT",
                    lon="LON",
                    size="SizeTemp",
                    color="MaxTemp",
                    animation_frame="YearMonth",  #  # Using YearMonth for animation
                    hover_name="DISTRICT",
                    hover_data={"MaxTemp": ":.2f °C", "LAT": False, "LON": False,"SizeTemp": False},
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",  
                    title="Monthly Maximum Temperature Variation Across Districts of Nepal (2012-2017)",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    sliders=[{"currentvalue": {"prefix": "Month: "}}]
                )
                st.plotly_chart(fig)

            elif climate_var == "Humidity":
                st.markdown("### Monthly Humidity Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See the humidity values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                df_filtered = climate_data.copy()
                fig = px.scatter_mapbox(
                    df_filtered,
                    lat='LAT',
                    lon='LON',
                    color='Humidity',
                    size='Humidity',
                    size_max=10,
                    animation_frame='YearMonth',  # Using YearMonth for animation
                    hover_name='DISTRICT',
                    hover_data={'Humidity': ":.1f %", 'LAT': False, 'LON': False},
                    zoom=5.5,
                    center={'lat': 27.7172, 'lon': 85.3240},
                    title='Monthly Humidity Variation Across Nepal (2012-2017)',
                    color_continuous_scale='Blues',
                    range_color=[climate_data['Humidity'].min(), climate_data['Humidity'].max()],
                    height=600
                )
                fig.update_layout(
                    mapbox_style="open-street-map",  
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig)

            elif climate_var == "Precipitation":
                st.markdown("### Monthly Precipitation Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See precipitation values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                df_filtered = climate_data.copy()
                fig = px.scatter_mapbox(
                    df_filtered,
                    lat='LAT',
                    lon='LON',
                    color='Prep',
                    size=[8]*len(df_filtered),
                    size_max=8,
                    animation_frame='YearMonth',  
                    hover_name='DISTRICT',
                    hover_data={'Prep': ":.1f mm", 'LAT': False, 'LON': False},
                    zoom=5,
                    center={'lat':28.0, 'lon':84.0},
                    title='Monthly Precipitation Across Districts (2012-2017)',
                    color_continuous_scale='Blues',
                    range_color=[0, climate_data['Prep'].max()]
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig)

            elif climate_var == "Wind Speed":
                st.markdown("### Monthly Wind Speed Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See wind speed values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                df_filtered = climate_data.copy()
                fig = px.scatter_mapbox(
                    df_filtered,
                    lat="LAT",
                    lon="LON",
                    size="WindSpeed",
                    color="WindSpeed",
                    animation_frame="YearMonth",  
                    hover_name="DISTRICT",
                    hover_data={"WindSpeed": ":.2f m/s", "LAT": False, "LON": False},
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="Monthly Wind Speed Variation Across Districts (2012-2017)",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    sliders=[{"currentvalue": {"prefix": "Month: "}}]
                )
                st.plotly_chart(fig)

        elif map_type == "Fire Variables":
            st.markdown("### Fire Variables Visualization\nExplore fire-related data across Nepal's districts.")

            firefile_path = "data/processed/combined_fire_climate.csv"
            fire_data = pd.read_csv(firefile_path)

            fire_data['YearMonth'] = fire_data['YEAR'].astype(str) + '-' + fire_data['MONTH'].astype(str).str.zfill(2)
            
            fire_var = st.selectbox("Select Fire Variable", [
                "Fire Count", 
                "Fire Confidence", 
                "Fire Radiative Power",
                "Fire Risk"
            ])
            
            if fire_var == "Fire Count":
                st.markdown("### Monthly Fire Count Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See fire count values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                fig = px.scatter_mapbox(
                    fire_data,
                    lat='LAT',
                    lon='LON',
                    color='Fire_Count',
                    size='Fire_Count',
                    animation_frame='YearMonth',
                    hover_name='DISTRICT',
                    hover_data={'Fire_Count': True, 'LAT': False, 'LON': False},
                    zoom=5,
                    center={'lat': 28.0, 'lon': 84.0},
                    title='Monthly Fire Count Across Districts (2012–2017)',
                    color_continuous_scale='Viridis',
                    range_color=[0, fire_data['Fire_Count'].max()]
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r": 0, "t": 40, "l": 0, "b": 0}
                )
                st.plotly_chart(fig)
            
            elif fire_var == "Fire Confidence":
                st.markdown("### Monthly Fire Confidence Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See fire confidence values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                
                fig = px.scatter_mapbox(
                    fire_data,
                    lat="LAT",
                    lon="LON",
                    size="Confidence",
                    color="Confidence",
                    animation_frame="YearMonth",
                    hover_name="DISTRICT",
                    hover_data={"Confidence": ":.2f", "LAT": False, "LON": False},
                    
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="Monthly Fire Confidence Across Districts of Nepal (2012–2017)",
                    color_continuous_scale="OrRd"
                )
                fig.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    sliders=[{"currentvalue": {"prefix": "Month: "}}]
                )
                st.plotly_chart(fig)
            elif fire_var == "Fire Radiative Power":
                st.markdown("### Monthly Fire Radiative Power Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See fire radiative power values and district names.
                - **Use Play/Pause**: Animation slider at the bottom to view data over months.
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                
                fig = px.scatter_mapbox(
                    fire_data,
                    lat="LAT",
                    lon="LON",
                    size="FRP",
                    color="FRP",
                    animation_frame="YearMonth",
                    hover_name="DISTRICT",
                    hover_data={"FRP": ":.2f MW", "LAT": False, "LON": False},
                    
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="Monthly Fire Radiative Power Across Districts of Nepal (2012–2017)",
                    color_continuous_scale="YlOrRd"
                )
                fig.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    sliders=[{"currentvalue": {"prefix": "Month: "}}]
                )
                st.plotly_chart(fig)
            elif fire_var == "Fire Risk":
                st.markdown("### Monthly Fire Risk Variation")
                
                # Add usage note below the map
                st.markdown("""
                **📌 Map Features:**
                - **Hover Over Districts**: See fire risk values and district names.
                
                - **Zoom In/Out**: Zoom into areas using the map's zoom controls.
                - **Full Screen**: Click on the full-screen button on the map for a larger view.
                """)
                district_risk = fire_data.groupby("DISTRICT").agg({
                   "LAT": "mean",
                   "LON": "mean",
                   "Confidence": "mean",
                   "FRP": "mean",
                   "Fire_Count": "sum"
                }).reset_index()

            
                district_risk['Fire_Risk'] = (district_risk['Confidence'] + district_risk['Fire_Count'] + district_risk['FRP']) / 3

                district_risk['RiskLevel'] = pd.qcut(district_risk['Fire_Risk'], q=3, labels=['Low', 'Medium', 'High'])

                fig = px.scatter_mapbox(
                    district_risk,      
                    lat="LAT",

                    lon="LON",
                    size="Fire_Risk",
                    color="RiskLevel",
                    
                    hover_name="DISTRICT",
                    hover_data={"Fire_Risk": ":.2f", "LAT": False, "LON": False},
                    size_max=15,
                    zoom=5,
                    mapbox_style="open-street-map",
                    title="Monthly Fire Risk Across Districts of Nepal (2012–2017)",
                    color_continuous_scale="YlGnBu",
                    color_discrete_map={
                     "Low": "#ff9999",
                     "Medium": "#ff4d4d",
                    "High": "#990000"
                    }
                )

                fig.update_layout(
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.error("Data files not found. Please ensure the data is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Data Visualization Page
def data_visualization_page():
    st.title("📊 Data Visualization")
    
    st.markdown("""
    ### Exploring Forest Fire Trends
    
    Dive deep into our comprehensive visualizations of climate patterns and fire occurrences across Nepal from 2012-2017.
    """)
    
    
    viz_category = st.radio("Select Category", [
        "Climate Analysis",
        "Fire Trends",
        "Climate-Fire Relationships",
        "District-Specific Analysis"
    ], horizontal=True)
    
    if viz_category == "Climate Analysis":
    
        viz_type = st.radio("Select Visualization Type", [
            "Heatmap View", 
            "Distribution View"
        ], horizontal=True)
        
        if viz_type == "Heatmap View":
           
            climate_param = st.selectbox("Select Climate Parameter", [
                "Temperature",
                "Humidity",
                "Wind Speed",
                "Precipitation"
            ])
            
           
            def create_climate_heatmap(param_name, df_filtered, value_col, cmap='coolwarm', title_prefix="Monthly Average"):
                st.markdown(f"### {title_prefix} {param_name} Heatmap (2012-2017)")
                st.markdown(f"This heatmap shows the average {param_name.lower()} patterns across months and years, highlighting seasonal variations.")
                
                try:
                   
                    pivot_df = df_filtered.pivot_table(index='MONTH', columns='YEAR', values=value_col)
                    
                   
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt='.1f', ax=ax)
                    plt.title(f'{title_prefix} {param_name} Heatmap (2012-2017)')
                    plt.xlabel('Year')
                    plt.ylabel('Month')
                    
                   
                    st.pyplot(fig)
                    
                    
                    interpretations = {
                        "Temperature": """
                        **Interpretation:**
                        - Darker red colors indicate higher temperatures
                        - Darker blue colors indicate lower temperatures
                        - Highest temperatures occur during summer months (May-June)
                        - Lowest temperatures typically in winter months (December-January)
                        """,
                        "Humidity": """
                        **Interpretation:**
                        - Darker blue  colors represent higher humidity levels
                        - White colors represent lower humidity levels
                        - Higher humidity is often associated with monsoon seasons
                        - Highest humidity levels occur during July-September
                        - Dry winter months show significantly lower humidity
                        """,
                        "Wind Speed": """
                        **Interpretation:**
                        - Yellow colors represent higher wind speeds
                        - Darker purple colors represent lower wind speeds
                        - Pre-monsoon months often show increased wind activity
                        - Understanding wind patterns is crucial for fire spread prediction
                        """,
                        "Precipitation": """
                        **Interpretation:**
                        - Darker blue colors represent higher precipitation
                        - White colors represent lower precipitation
                        - Monsoon months (June-September) show distinctly higher precipitation
                        - Winter months are typically drier with minimal rainfall
                        - Precipitation patterns directly impact forest fire risk
                        """
                    }
                    
                    st.markdown(interpretations.get(param_name, ""))
                except Exception as e:
                    st.error(f"Error loading {param_name.lower()} heatmap: {e}")
            
            # Load the data once
            try:
                file_path = "data/processed/flitered_climate_data.csv"
                df_filtered = pd.read_csv(file_path)
                
                # 
                param_mapping = {
                    "Temperature": {"col": "AvgTemp", "cmap": "coolwarm"},
                    "Humidity": {"col": "Humidity", "cmap": "YlGnBu"},
                    "Wind Speed": {"col": "WindSpeed", "cmap": "viridis"},
                    "Precipitation": {"col": "Prep", "cmap": "Blues"}
                }
                
                # Create the appropriate heatmap based on user selection
                if climate_param in param_mapping:
                    create_climate_heatmap(
                        climate_param, 
                        df_filtered, 
                        param_mapping[climate_param]["col"],
                        param_mapping[climate_param]["cmap"]
                    )
            except Exception as e:
                st.error(f"Error loading climate data: {e}")
                
        elif viz_type == "Distribution View":
           
            climate_param = st.selectbox("Select Climate Parameter", [
                "Temperature",
                "Humidity",
                "Wind Speed",
                "Precipitation"
            ])
            
            
            def create_climate_distribution(param_name, df_filtered, value_col, color_scheme='viridis'):
                st.markdown(f"### Monthly {param_name} Distribution (2012-2017)")
                st.markdown(f"This box plot shows the distribution of {param_name.lower()} values across different months, highlighting seasonal variations and outliers.")
                
                try:
                   
                    fig = px.box(df_filtered, 
                             x='MONTH', 
                             y=value_col,
                             color='MONTH',
                             title=f'Monthly {param_name} Distribution (2012-2017)',
                             labels={'MONTH': 'Month', value_col: f'{param_name} {get_unit(param_name)}'})
                    
                    fig.update_layout(xaxis={'tickvals': list(range(1,13)),
                                   'ticktext': ['Jan','Feb','Mar','Apr','May','Jun',
                                                'Jul','Aug','Sep','Oct','Nov','Dec']})
                    
                   
                    st.plotly_chart(fig)
                    
                    # Add parameter-specific interpretation
                    interpretations = {
                        "Temperature": """
                        **Interpretation:*
                        - Summer months(May-June) show higher median temperatures with different variability patterns
                        - Winter months (December-January) show lower median temperatures with less variability
                        -Outliers may indicate extreme weather events or unusual temperature patterns
                        """,
                        "Humidity": """
                        **Interpretation:** 
                        - Monsoon months (July-September) show consistently higher humidity with less variation
                        - Winter months (December-January) show lower humidity with more variability
                        -Outliers may indicate extreme weather events or unusual humidity patterns
                        """,
                        "Wind Speed": """
                        **Interpretation:**
                        - Lower wind speeds are observed during monsoon months (June-September)
                        - Wind speed peaks in the spring months (March-April)
                        - Higher wind speeds can be associated with fire spread risk
                        - Seasonal wind patterns can be observed through median variations
                        - Outliers may indicate extreme weather events or unusual wind patterns
                        """,
                        "Precipitation": """
                        **Interpretation:**
                        - Higher precipitation levels are observed during monsoon months (June-September)
                        - Lower precipitation levels are observed during winter months (December-January)
                        - The box plot shows the distribution of precipitation levels, indicating the variability in rainfall
                        - Understanding precipitation patterns is crucial for fire risk assessment
                        - Outliers may indicate extreme weather events or unusual rainfall patterns
                        """
                       
                    }
                    
                    st.markdown(interpretations.get(param_name, ""))
                except Exception as e:
                    st.error(f"Error loading {param_name.lower()} distribution: {e}")
            
            # Helper function to get appropriate units for each parameter
            def get_unit(param_name):
                units = {
                    "Temperature": "(°C)",
                    "Humidity": "(%)",
                    "Wind Speed": "(m/s)",
                    "Precipitation": "(mm)"
                }
                return units.get(param_name, "")
            
            # Load the data
            try:
                file_path = "data/processed/flitered_climate_data.csv"
                df_filtered = pd.read_csv(file_path)
                
              
                param_mapping = {
                    "Temperature": "AvgTemp",
                    "Humidity": "Humidity",
                    "Wind Speed": "WindSpeed",
                    "Precipitation": "Prep"
                }
                
                
                if climate_param in param_mapping:
                    create_climate_distribution(
                        climate_param, 
                        df_filtered, 
                        param_mapping[climate_param]
                    )
            except Exception as e:
                st.error(f"Error loading climate data: {e}")
    
    elif viz_category == "Fire Trends":
        fire_viz_type = st.selectbox("Select Fire Visualization", [
            "Top Fire-Prone Districts",
            "Annual Fire Count",
            "Average Monthly Fire Occurrence",
            "Monthly Fire Count Trend"
        ])
        
        if fire_viz_type == "Top Fire-Prone Districts":
            st.markdown("### Top 15 Fire-Prone Districts (2012-2017)")
            st.markdown("This bar chart shows the districts with the highest number of forest fires over the 5-year period.")
            
            try:
                file_path = "data/processed/combined_fire_climate.csv"
                df_fire_filtered = pd.read_csv(file_path)
                
                
                fig, ax = plt.subplots(figsize=(14, 7))
                district_fires = df_fire_filtered.groupby('DISTRICT')['Fire_Count'].sum().sort_values(ascending=False)[:15]
                sns.barplot(x=district_fires.values, y=district_fires.index, palette='Reds_r', ax=ax)
                plt.title('Top 15 Fire-Prone Districts (2012-2017)')
                plt.xlabel('Total Fire Count')
                plt.ylabel('District')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                
                
                st.pyplot(fig)
                
                st.markdown("""
                **Key Insights:**
                - These districts should be prioritized for fire prevention measures
                - Geographic and climatic factors likely contribute to higher fire incidence
                - Fire prevention resources can be allocated based on historical risk patterns
                """)
            except Exception as e:
                st.error(f"Error loading top fire-prone districts chart: {e}")
        
        elif fire_viz_type == "Annual Fire Count":
            st.markdown("### Annual Forest Fire Count in Nepal (2012-2017)")
            st.markdown("This bar chart shows the yearly trend of forest fires, highlighting years with higher fire incidences.")
            
            try:
                
                file_path = "data/processed/combined_fire_climate.csv"
                df_fire_filtered = pd.read_csv(file_path)
                
                # Aggregate fire counts by year
                annual_fire_counts = df_fire_filtered.groupby('YEAR', as_index=False)['Fire_Count'].sum()
                
                # Create the bar chart
                fig_annual = px.bar(
                    annual_fire_counts,
                    x='YEAR',
                    y='Fire_Count',
                    labels={'Fire_Count': 'Total Fire Count', 'YEAR': 'Year'},
                    color='Fire_Count',
                    color_continuous_scale='OrRd',
                    title='Annual Forest Fire Count in Nepal (2012–2017)'
                )
                
                # Customize the layout
                fig_annual.update_layout(
                    xaxis=dict(tickmode='linear'),
                    yaxis_title='Total Fire Count',
                    plot_bgcolor='white',
                    hovermode='x unified'
                )
                
                # Display in Streamlit
                st.plotly_chart(fig_annual)
                
                st.markdown("""
                **Insights:**
                - Observe the years with significantly higher fire incidents
                - These patterns may correlate with drought years or specific climate conditions
                """)
            except Exception as e:
                st.error(f"Error loading annual fire count chart: {e}")
                
        elif fire_viz_type == "Average Monthly Fire Occurrence":
            st.markdown("### Average Monthly Fire Occurrence (2012-2017)")
            st.markdown("This line chart shows the average number of fires by month, highlighting seasonal fire patterns.")
            
            try:
                file_path = "data/processed/combined_fire_climate.csv"
                df_fire_filtered = pd.read_csv(file_path)
                
                # Calculate average fire counts by month
                monthly_avg_fire = df_fire_filtered.groupby('MONTH', as_index=False)['Fire_Count'].mean()
                
                # Define month names for better readability
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_avg_fire['Month_Name'] = monthly_avg_fire['MONTH'].apply(lambda x: month_names[x - 1])
                
                # Create the line chart
                fig_monthly = px.line(
                    monthly_avg_fire,
                    x='Month_Name',
                    y='Fire_Count',
                    markers=True,
                    labels={'Fire_Count': 'Average Fire Count', 'Month_Name': 'Month'},
                    title='Average Monthly Fire Occurrence',
                    color_discrete_sequence=['firebrick']
                )
                
                # Customize the layout
                fig_monthly.update_layout(
                    yaxis_title='Average Fire Count',
                    xaxis_title='Month',
                    plot_bgcolor='white',
                    hovermode='x unified'
                )
                
                # Display in Streamlit
                st.plotly_chart(fig_monthly)
                
                st.markdown("""
                **Seasonal Insights:**
                - Fire occurrences show a clear seasonal pattern
                - Peak fire season can be identified for targeted prevention
                """)
            except Exception as e:
                st.error(f"Error loading monthly fire occurrence chart: {e}")
                
        elif fire_viz_type == "Monthly Fire Count Trend":
            st.markdown("### Monthly Forest Fire Counts in Nepal (2012-2017)")
            st.markdown("This line chart shows the detailed monthly trend of forest fires over the entire period.")
            
            try:
                file_path = "data/processed/combined_fire_climate.csv"
                df_fire_filtered = pd.read_csv(file_path)
                
                # Aggregate fire counts by year and month
                monthly_fire_counts = df_fire_filtered.groupby(['YEAR', 'MONTH'], as_index=False)['Fire_Count'].sum()
                
                # Create a 'Date' column for plotting
                monthly_fire_counts['Date'] = pd.to_datetime(monthly_fire_counts[['YEAR', 'MONTH']].assign(DAY=1))
                
                # Sort by date
                monthly_fire_counts.sort_values('Date', inplace=True)
                
                # Plot the monthly fire counts
                fig = px.line(
                    monthly_fire_counts,
                    x='Date',
                    y='Fire_Count',
                    title='Monthly Forest Fire Counts in Nepal (2012–2017)',
                    labels={'Fire_Count': 'Fire Count', 'Date': 'Date'},
                    markers=True
                )
                
                # Customize the layout
                fig.update_traces(line=dict(color='firebrick', width=2), marker=dict(size=6))
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Fire Count',
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                # Display in Streamlit
                st.plotly_chart(fig)
                
                st.markdown("""
                **Timeline Analysis:**
                - Observe the cyclical nature of fire occurrences
                - Identify specific months/years with unusually high fire activity
                """)
            except Exception as e:
                st.error(f"Error loading monthly fire count trend chart: {e}")
    
    elif viz_category == "Climate-Fire Relationships":
        climate_fire_viz_type = st.selectbox("Select Climate-Fire Visualization", [
            "Correlation Heatmap",
            "Climate Parameters vs Fire Metrics",
            "Fire Probability by Temperature and Humidity"
        ])
        
        try:
            file_path = "data/processed/combined_fire_climate.csv"
            df = pd.read_csv(file_path)
            
            if climate_fire_viz_type == "Correlation Heatmap":
                st.markdown("### Correlation Between Climate Variables and Fire Metrics")
                st.markdown("This heatmap shows the correlations between different climate variables and fire metrics.")
                
                # Select relevant columns
                fire_climate_vars = ['Prep', 'AvgTemp', 'MaxTemp', 'Humidity', 'WindSpeed',
                                   'Brightness', 'Confidence', 'FRP', 'Fire_Count']
                
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(df[fire_climate_vars].corr()[['Fire_Count', 'Confidence']].sort_values(by='Fire_Count', ascending=False),
                          annot=True, cmap='RdYlGn', vmin=-1, vmax=1, center=0)
                plt.title('Correlation Between Climate Variables and Fire Metrics')
                
            
                st.pyplot(fig)
                
                st.markdown("""
                **Key Insights from Correlation Heatmap:**
                - Fire_Count and FRP (Fire Radiative Power) are highly correlated (0.9)
                - Humidity has a notable negative correlation (-0.28) with Fire_Count and (-0.49) with Confidence
                - Brightness and Confidence are very strongly correlated (0.97), indicating that brighter fires are detected with higher confidence
                - Wind speed shows a weak positive correlation (0.22) with fire count
                - Temperature metrics (AvgTemp, MaxTemp) show weak positive correlations with fire occurrence
                - Precipitation shows negative correlations with both fire metrics, confirming that wetter conditions reduce fire probability
                """)
                
            elif climate_fire_viz_type == "Climate Parameters vs Fire Metrics":
                st.markdown("### Relationships Between Climate Parameters and Fire Metrics")
                st.markdown("These scatter plots show how different climate variables relate to fire counts and detection confidence.")
                
                # Create PairGrid visualization
                with plt.style.context('seaborn-v0_8-whitegrid'):
                    g = sns.PairGrid(df, y_vars=['Fire_Count', 'Confidence'],
                                  x_vars=['AvgTemp', 'MaxTemp', 'Humidity', 'WindSpeed', 'Prep'],
                                  height=3, aspect=1.2)
                    g.map(sns.regplot, scatter_kws={'alpha':0.3, 'color':'orange'},
                        line_kws={'color':'red'})
                    g.fig.suptitle('Climate Parameters vs Fire Metrics', y=1.05)
                    
                    # Display in Streamlit
                    st.pyplot(g.fig)
                
                st.markdown("""
                **Key Insights from Climate Parameters vs. Fire Metrics:**
                - The scatter plots show the relationships between various climate parameters and fire metrics
                - The orange scatter points with red trend lines reveal important patterns:
                  - Humidity shows a negative correlation with both fire count and confidence, indicating fewer fires occur in more humid conditions
                  - Wind speed appears to have a moderate positive correlation with confidence in fire detection
                  - Temperature variables (AvgTemp, MaxTemp) show slight positive correlations with fire count
                  - Precipitation shows a negative correlation with fire metrics, as expected
                - These relationships can inform fire prediction models by identifying the most influential climate factors
                """)
                
            elif climate_fire_viz_type == "Fire Probability by Temperature and Humidity":
                st.markdown("### Fire Probability Based on Temperature and Humidity")
                st.markdown("This heatmap shows how the probability of fire occurrence varies with different combinations of temperature and humidity.")
                
                # Create bins for climate variables
                df['Temp_bin'] = pd.cut(df['AvgTemp'], bins=5)
                df['Humidity_bin'] = pd.cut(df['Humidity'], bins=5)
                
                # Calculate fire probability
                prob_table = df.groupby(['Temp_bin', 'Humidity_bin'])['Fire_Count'].apply(
                    lambda x: (x > 0).mean()).unstack()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(prob_table, annot=True, fmt=".0%", cmap="YlOrRd")
                plt.title('Probability of Fire Occurrence by Temperature and Humidity')
                plt.xlabel('Humidity Range')
                plt.ylabel('Temperature Range')
                
                # Display in Streamlit
                st.pyplot(fig)
                
                st.markdown("""
                **Key Insights from Fire Probability Heatmap:**
                - This heatmap clearly demonstrates how fire probability increases with:
                  1. Higher temperatures (moving down the y-axis)
                  2. Lower humidity (moving left on the x-axis)
                - The highest fire probabilities (80%) occur in the hottest and driest conditions
                - The lowest fire probabilities (0-3%) occur in cooler and more humid environments
                - There's a clear gradient showing how the combination of these two factors affects fire risk
                - This visualization can directly inform early warning systems based on temperature and humidity forecasts
                """)
                
        except Exception as e:
            st.error(f"Error in climate-fire relationship analysis: {e}")
    
    elif viz_category == "District-Specific Analysis":
        st.markdown("### District-Level Fire Trend Analysis")
        st.markdown("This interactive visualization allows you to explore fire trends for specific districts in Nepal.")
        
        try:
            file_path = "data/processed/combined_fire_climate.csv"
            df = pd.read_csv(file_path)
            df['MONTH_YEAR'] = df['MONTH'].apply(lambda x: f"{x:02d}") + '-' + df['YEAR'].astype(str)
            districts = sorted(df['DISTRICT'].unique())
            
            # District selection dropdown
            selected_district = st.selectbox("Select District:", districts)
            
            # View mode selection
            view_mode = st.radio("View Mode:", ["Monthly", "Yearly"], horizontal=True)
            
            # Filter data for selected district
            district_data = df[df['DISTRICT'] == selected_district]
            
            if view_mode == "Monthly":
                # Monthly view - only showing months with fires
                plot_data = district_data[district_data['Fire_Count'] > 0]
                if plot_data.empty:
                    st.info(f"No fire records found for {selected_district} (2012-2017)")
                else:
                    # Create a Plotly bar chart for better interactivity
                    fig = px.bar(
                        plot_data.sort_values(['YEAR', 'MONTH']),
                        x='MONTH_YEAR',
                        y='Fire_Count',
                        title=f'Monthly Fire Occurrences in {selected_district} (2012-2017)<br><sup>Active fire months only</sup>',
                        labels={'MONTH_YEAR': 'Month-Year (MM-YYYY)', 'Fire_Count': 'Fire Count'},
                        color='Fire_Count',
                        color_continuous_scale='OrRd'
                    )
                    
                    fig.update_layout(
                        xaxis=dict(tickangle=45),
                        yaxis_title='Fire Count',
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig)
            else:  # Yearly view
                yearly_data = district_data.groupby('YEAR')['Fire_Count'].sum().reset_index()
                
                # Create a Plotly bar chart for yearly view
                fig = px.bar(
                    yearly_data,
                    x='YEAR',
                    y='Fire_Count',
                    title=f'Annual Fire Summary for {selected_district} (2012-2017)',
                    labels={'YEAR': 'Year', 'Fire_Count': 'Total Fire Count'},
                    color='Fire_Count',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    xaxis=dict(tickmode='linear'),
                    yaxis_title='Total Fire Count'
                )
                
                st.plotly_chart(fig)
            
            # Add some context for the selected district
            climate_summary = district_data.groupby('DISTRICT').agg({
                'AvgTemp': 'mean',
                'Humidity': 'mean',
                'Prep': 'mean',
                'WindSpeed': 'mean',
                'Fire_Count': 'sum'
            }).reset_index()
            
            if not climate_summary.empty:
                st.markdown(f"### Climate Summary for {selected_district}")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Avg Temp (°C)", f"{climate_summary['AvgTemp'].values[0]:.1f}")
                with col2:
                    st.metric("Humidity (%)", f"{climate_summary['Humidity'].values[0]:.1f}")
                with col3:
                    st.metric("Precip (mm)", f"{climate_summary['Prep'].values[0]:.1f}")
                with col4:
                    st.metric("Wind Speed (m/s)", f"{climate_summary['WindSpeed'].values[0]:.1f}")
                with col5:
                    st.metric("Total Fires", f"{int(climate_summary['Fire_Count'].values[0])}")
                    
                st.markdown("""
                **Insights for Fire Management:**
                - Identify peak fire seasons for this district
                - Plan resource allocation for fire prevention during high-risk months
                - Compare with climate patterns to develop early warning systems
                """)
        except Exception as e:
            st.error(f"Error in district-specific analysis: {e}")



def model_prediction_page():
    # Page header with styled title and description
    st.markdown("<h1 class='main-header'>🔥 Forest Fire Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: rgba(255, 235, 205, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='font-size: 16px; margin-bottom: 0;'>
            This tool helps predict forest fire risk based on climate parameters and location data. 
            Enter the required information below to get a risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    model_dir = os.path.join("models")
    try:
        # Load models and encoders
        risk_model = joblib.load(os.path.join(model_dir, 'risk_model.pkl'))
        fire_model = joblib.load(os.path.join(model_dir, 'fire_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        district_encoder = joblib.load(os.path.join(model_dir, 'district_encoder.pkl'))
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return
    
    # Load combined dataset to extract district and coordinates
    try:
        combined_df = pd.read_csv("data/processed/combined_fire_climate.csv")
    except Exception as e:
        st.error(f"⚠️ Error loading combined dataset: {e}")
        return

    # Preprocess district column
    combined_df['DISTRICT'] = combined_df['DISTRICT'].str.lower().str.strip()
    district_list = sorted(combined_df['DISTRICT'].unique())
    
    # --- User Input ---
    st.subheader("📍 Select Location")
    
    # Select District from dropdown with search
    district = st.selectbox(
        "Select District", 
        district_list,
        help="Choose the district for which you want to predict fire risk"
    )
        
    # Automatically retrieve lat/lon from data
    try:
        lat_lon_row = combined_df[combined_df['DISTRICT'] == district][['LAT', 'LON']].dropna().iloc[0]
        lat = lat_lon_row['LAT']
        lon = lat_lon_row['LON']
        
        # Display custom map with pin emoji instead of default red dot
        import folium
        from streamlit_folium import folium_static
        
        # Create a folium map centered at the district's coordinates
        m = folium.Map(location=[lat, lon], zoom_start=8)
        
        # Add a marker with custom icon (pin emoji)
        folium.Marker(
            location=[lat, lon],
            popup=district.title(),
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(12, 36),
                html=f'<div style="font-size: 24px;">📍</div>',
            )
        ).add_to(m)
        
        # Display the map
        folium_static(m)
        
        st.markdown(f"""
        <div style='margin-top: 10px; padding: 10px; background-color:transparent; border-radius: 5px;'>
            <b>Coordinates:</b> Lat: {lat:.4f}, Lon: {lon:.4f}
        </div>
        """, unsafe_allow_html=True)
        
    except IndexError:
        st.error("❌ Latitude and Longitude not found for selected district.")
        return
    except Exception as e:
        st.error(f"Error loading map: {e}")
        # Fallback to text-only location
        try:
            lat_lon_row = combined_df[combined_df['DISTRICT'] == district][['LAT', 'LON']].dropna().iloc[0]
            lat = lat_lon_row['LAT']
            lon = lat_lon_row['LON']
            
            st.markdown(f"""
            <div style='margin-top: 10px; padding: 10px; background-color: transparent; border-radius: 5px;'>
                <b>Selected District:</b> {district.title()}<br>
                <b>Coordinates:</b> Lat: {lat:.4f}, Lon: {lon:.4f}
            </div>
            """, unsafe_allow_html=True)
            
        except IndexError:
            st.error("❌ Latitude and Longitude not found for selected district.")
            return

    # --- Climate Parameters ---
    st.subheader("🌡️ Climate Parameters")
    
    with st.container():
        st.markdown("""
        <div style='background-color: transparent; padding: 15px; border-radius: 10px; border-left: 5px solid #FF5733;'>
        """, unsafe_allow_html=True)
        
        # Date-based inputs
        current_month = datetime.now().month
        date_col1, date_col2 = st.columns([3, 1])
        
        with date_col1:
            month = st.slider(
                "Month", 
                min_value=1, 
                max_value=12, 
                value=current_month,
                format="%d",
                help="Select the month for prediction (1-12)"
            )
            
            # Month name display
            month_names = ["January", "February", "March", "April", "May", "June", 
                         "July", "August", "September", "October", "November", "December"]
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 15px;'>
                <span style='font-weight: bold; color: #FF5733;'>{month_names[month-1]}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with date_col2:
            use_current_data = st.checkbox("Use current weather data", value=False, 
                                         help="Fetch latest weather data for the selected location")
        
        # Two columns for climate parameters
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            prep = st.number_input(
                "Precipitation (mm)", 
                min_value=0.0, 
                value=26.7,
                format="%.1f",
                help="Average precipitation in millimeters",
                disabled=use_current_data
            )
            
            avg_temp = st.number_input(
                "Average Temperature (°C)", 
                min_value=-10.0, 
                value=27.43,
                format="%.2f",
                help="Average temperature in degrees Celsius",
                disabled=use_current_data
            )
            
            # Make humidity consistent with other inputs
            humidity = st.number_input(
                "Humidity (%)", 
                min_value=0, 
                max_value=100, 
                value=30,
                format="%d",
                help="Relative humidity percentage",
                disabled=use_current_data
            )
        
        with param_col2:
            max_temp = st.number_input(
                "Max Temperature (°C)", 
                min_value=-10.0, 
                value=34.93,
                format="%.2f",
                help="Maximum temperature in degrees Celsius",
                disabled=use_current_data
            )
            
            wind_speed = st.number_input(
                "Wind Speed (m/s)", 
                min_value=0.0, 
                value=2.86,
                format="%.2f",
                help="Wind speed in meters per second",
                disabled=use_current_data
            )
            
            # Empty slot to balance the layout or add another parameter
            soil_moisture = st.number_input(
                "Soil Moisture (%)", 
                min_value=0, 
                max_value=100, 
                value=15,
                format="%d",
                help="Soil moisture percentage (optional)",
                disabled=use_current_data
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Fetch weather data if checkbox is checked
        if use_current_data:
            with st.spinner("Fetching current weather data..."):
                # This would be replaced with actual API call in production
                time.sleep(1)  # Simulate API call
                
                st.success("✅ Weather data fetched successfully!")
                st.info("Using real-time weather data for prediction")
                
                # In a real implementation, these values would come from the API
                prep = 10.2
                avg_temp = 28.5
                max_temp = 35.3
                humidity = 45
                wind_speed = 3.2
    
    # Add notes field
    notes = st.text_area("📝 Notes (optional)", 
                        placeholder="Add any additional context or observations...",
                        help="Enter any relevant notes about this prediction")
    
    # Predict button with styling
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button(
            "🔮 Predict Fire Risk", 
            use_container_width=True,
            help="Click to calculate fire risk based on input parameters"
        )
    
    # Show a divider before results
    if predict_button:
        st.markdown("<hr style='margin: 30px 0; border: none; height: 1px; background-color: #ddd;'>", unsafe_allow_html=True)
        
        # Construct input data
        input_df = pd.DataFrame({
            'DISTRICT': [district],
            'MONTH': [month],
            'LAT': [lat],
            'LON': [lon],
            'Prep': [prep],
            'AvgTemp': [avg_temp],
            'MaxTemp': [max_temp],
            'Humidity': [humidity],
            'WindSpeed': [wind_speed]
        })

        # Add cyclical month features
        input_df['Month_sin'] = np.sin(2 * np.pi * input_df['MONTH'] / 12)
        input_df['Month_cos'] = np.cos(2 * np.pi * input_df['MONTH'] / 12)

        # One-hot encode district
        district_cols = [f"DISTRICT_{d}" for d in district_encoder.categories_[0]]
        district_encoded = district_encoder.transform(input_df[['DISTRICT']])
        for i, col in enumerate(district_cols):
            input_df[col] = district_encoded[0][i]

        # Fill any missing district columns with 0 (in case of unseen districts)
        for col in district_cols:
            if col not in input_df:
                input_df[col] = 0

        # Define full feature list
        features = ['Prep', 'AvgTemp', 'MaxTemp', 'Humidity', 'WindSpeed',
                    'Month_sin', 'Month_cos', 'LAT', 'LON'] + district_cols

        with st.spinner("Calculating risk assessment..."):
            X_input = input_df[features]
            X_scaled = scaler.transform(X_input)

            # Predict fire risk and occurrence
            risk_value = risk_model.predict(X_scaled)[0]
            fire_probability = fire_model.predict_proba(X_scaled)[0][1] * 100

            # Combine for final confidence estimation
            risk_factor = min(risk_value / 40, 1.0)
            adjusted_confidence = fire_probability * (0.8 + 0.2 * risk_factor)

            # Get risk categories
            risk_category = get_risk_category(risk_value)
            confidence_level = get_confidence_label(adjusted_confidence)
        
        # Display results in an attractive format
        st.markdown("<h2 class='sub-header'>🧪 Prediction Results</h2>", unsafe_allow_html=True)
        
        # Create columns for results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            # Risk Score Gauge - Improved color contrast
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fire Risk Score", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': "darkgray"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 15], 'color': "#4CAF50"},    # Darker green
                        {'range': [15, 25], 'color': "#FFC107"},   # Darker yellow
                        {'range': [25, 35], 'color': "#FF9800"},   # Darker orange
                        {'range': [35, 40], 'color': "#F44336"}    # Darker red
                    ],
                    'threshold': {
                        'line': {'color': "#B71C1C", 'width': 4},  # Darker red
                        'thickness': 0.75,
                        'value': risk_value
                    }
                }
            ))
            # FIX: Change transparent to rgba(0,0,0,0) for plotly
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), 
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk category with improved contrast
            risk_colors = {
                "Low": "#2E7D32",       # Darker Green
                "Moderate": "#F57F17",  # Darker Yellow
                "High": "#E65100",      # Darker Orange
                "Extreme": "#B71C1C"    # Darker Red
            }
            
            st.markdown(f"""
            <div style='background-color: {risk_colors[risk_category]}22; padding: 15px; border-radius: 10px; 
                border: 2px solid {risk_colors[risk_category]}; text-align: center; margin-top: 10px;'>
                <h3 style='margin: 0; color: {risk_colors[risk_category]}'>
                    {risk_category} Risk
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Add metric for direct number display
            st.metric(
                "Risk Score", 
                f"{risk_value:.1f}/40", 
                delta=None, 
                delta_color="off"
            )
        
        with res_col2:
            # Confidence Gauge with improved contrast
            fig2 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = adjusted_confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%"},
                title = {'text': "Fire Occurrence Confidence", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': "darkgray"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 20], 'color': "#0D47A1"},     # Darker blue
                        {'range': [20, 40], 'color': "#0097A7"},    # Darker cyan
                        {'range': [40, 60], 'color': "#FBC02D"},    # Darker yellow
                        {'range': [60, 80], 'color': "#F57C00"},    # Darker orange
                        {'range': [80, 100], 'color': "#C62828"}    # Darker red
                    ],
                    'threshold': {
                        'line': {'color': "#B71C1C", 'width': 4},   # Darker red
                        'thickness': 0.75,
                        'value': adjusted_confidence
                    }
                }
            ))
            # FIX: Change transparent to rgba(0,0,0,0) for plotly
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), 
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Confidence level with improved contrast
            confidence_colors = {
                "Very Low": "#0D47A1",    # Darker Blue
                "Low": "#006064",         # Darker Cyan
                "Moderate": "#F57F17",    # Darker Yellow
                "High": "#E65100",        # Darker Orange
                "Very High": "#B71C1C"    # Darker Red
            }
            
            st.markdown(f"""
            <div style='background-color: {confidence_colors[confidence_level]}22; padding: 15px; border-radius: 10px; 
                border: 2px solid {confidence_colors[confidence_level]}; text-align: center; margin-top: 10px;'>
                <h3 style='margin: 0; color: {confidence_colors[confidence_level]}'>
                    {confidence_level} Confidence
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Add metric for direct percentage display
            st.metric(
                "Confidence Score", 
                f"{adjusted_confidence:.1f}%", 
                delta=None, 
                delta_color="off"
            )
        
        
        # Export section
        st.subheader("📥 Export Results")
        
        # Create a CSV for download
        csv_data = f"""Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
District,{district.title()}
Month,{month_names[month-1]}
Coordinates,{lat:.4f},{lon:.4f}
Precipitation (mm),{prep}
Average Temperature (°C),{avg_temp}
Maximum Temperature (°C),{max_temp}
Humidity (%),{humidity}
Wind Speed (m/s),{wind_speed}
Risk Score,{risk_value:.2f}
Risk Category,{risk_category}
Confidence Score,{adjusted_confidence:.2f}%
Confidence Level,{confidence_level}
Notes,{notes}
"""
        
        # Download button
        st.download_button(
            label="📊 Download Report",
            data=csv_data,
            file_name=f"fire_risk_{district}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Add disclaimer about prediction accuracy
        st.markdown("""
        <div style='background-color: transparent; padding: 10px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #ffc107;'>
            <p style='margin: 0; font-size: 14px;'>⚠️ <strong>Note:</strong> These predictions are based on historical data and climate patterns. 
            Actual fire conditions may vary. Use this information as a guide and refer to official sources for critical decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a timestamp
        st.markdown(f"""
        <div style='text-align: center; color: gray; font-size: 12px; margin-top: 30px;'>
            Prediction generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}
        </div>
        """, unsafe_allow_html=True)

# Helper functions (keep the same logic)
def classify_fire_risk(confidence):
    if confidence < 50:
        return "Low"
    elif confidence < 80:
        return "Moderate"
    else:
        return "High"

def get_risk_category(risk_value):
    if risk_value < 15:
        return "Low"
    elif risk_value < 25:
        return "Moderate"
    elif risk_value < 35:
        return "High"
    else:
        return "Extreme"

def get_confidence_label(confidence):
    if confidence < 20:
        return "Very Low"
    elif confidence < 40:
        return "Low"
    elif confidence < 60:
        return "Moderate"
    elif confidence < 80:
        return "High"
    else:
        return "Very High"

def datasets_page():
    st.title("📂 Forest Fire Datasets")
    
    st.markdown("Explore the datasets used for climate analysis, fire detection, and forest coverage in Nepal.")
    
    dataset_options = {
        "Climate Data": {
            "raw_path": "data/raw/climate_data_nepal_district_wise_monthly.csv",
            "filtered_path": "data/processed/flitered_climate_data.csv",
            "description": "District-wise monthly climate data from the MERRA2 dataset (2012–2017)."
        },
        "Fire Data": {
            "raw_path": "data/raw/modis/modis_2012_2017_all_districts.csv",
            "filtered_path": "data/processed/filtered_fire_data.csv",
            "description": "MODIS satellite fire detection data from NASA LANCE FIRMS (2012–2017)."
        },
        "Forest Coverage Data": {
            "raw_paths": [
                "data/raw/forest_coverage/forest-coverage-by-district.csv",
                "data/raw/forest_coverage/forest-coverage-by-province.csv"
            ],
            "description": "Forest coverage data by district and province."
        }
    }

    selected_dataset = st.selectbox("Select a Dataset", list(dataset_options.keys()))
    st.markdown(f"**Description:** {dataset_options[selected_dataset]['description']}")

    if selected_dataset in ["Climate Data", "Fire Data"]:
        view_option = st.radio("Choose an option", ["View Raw Data", "View Filtered Data", "Download Data"])
        
        if view_option == "View Raw Data":
            df = pd.read_csv(dataset_options[selected_dataset]["raw_path"])
            st.markdown("#### Raw Data Preview")
            st.dataframe(df.head(100))
            if st.button("Load Full Data"):
                st.dataframe(df)

            if selected_dataset == "Climate Data":
                st.markdown("**🛈 Column Descriptions for Climate Data:**")
                st.markdown("""
                - **PRECTOT**: Precipitation (mm/day)  
                - **PS**: Surface Pressure (kPa)  
                - **QV2M**: Specific Humidity at 2m (g/kg)  
                - **RH2M**: Relative Humidity at 2m (%)  
                - **T2M**: Temperature at 2m (°C)  
                - **T2MWET**: Wet Bulb Temp at 2m (°C)  
                - **T2M_MAX/MIN/RANGE**: Max, Min, and Range of Temp at 2m (°C)  
                - **TS**: Earth Skin Temp (°C)  
                - **WS10M/WS50M**: Wind Speed at 10m/50m (m/s)  
                - **_MAX/MIN/RANGE**: Max, Min, and Range values of wind speed  
                -**Note**: Before January 2018, precipitation is reported as an accumulation; after, it is an average rate.                   
                """)

            elif selected_dataset == "Fire Data":
                st.markdown("**🛈 Column Descriptions for Fire Data:**")
                st.markdown("""
                - **brightness**: Brightness temperature of the fire pixel (in K)  
                - **confidence**: Detection confidence (range 0–100)  
                - **thermaldata**: Band 31 brightness temperature of the pixel (in K)  
                - **frp**: Fire radiative power (in MW - megawatts)  
                - **daynight**: Detected during the day or night (`D` for Day, `N` for Night)  
                """)

        elif view_option == "View Filtered Data":
            df = pd.read_csv(dataset_options[selected_dataset]["filtered_path"])
            st.markdown("#### Filtered Data Preview")
            st.dataframe(df.head(100))
            if st.button("Load Full Data"):
                st.dataframe(df)
            st.markdown("🔹 **Note**: This is a filtered version of the raw data, focusing on relevant features.")

        elif view_option == "Download Data":
            download_choice = st.radio("Download", ["Raw Data", "Filtered Data"])
            if download_choice == "Raw Data":
                with open(dataset_options[selected_dataset]["raw_path"], "rb") as f:
                    st.download_button("📥 Download Raw Data", f, file_name="raw_data.csv")
            else:
                with open(dataset_options[selected_dataset]["filtered_path"], "rb") as f:
                    st.download_button("📥 Download Filtered Data", f, file_name="filtered_data.csv")

    elif selected_dataset == "Forest Coverage Data":
        st.markdown("#### Raw Data Preview (By District)")
        df_district = pd.read_csv(dataset_options[selected_dataset]["raw_paths"][0])
        st.dataframe(df_district.head(100))
        if st.button("Load Full District Data"):
            st.dataframe(df_district)

        st.markdown("#### Raw Data Preview (By Province)")
        df_province = pd.read_csv(dataset_options[selected_dataset]["raw_paths"][1])
        st.dataframe(df_province.head(100))
        if st.button("Load Full Province Data"):
            st.dataframe(df_province)

        with open(dataset_options[selected_dataset]["raw_paths"][0], "rb") as f:
            st.download_button("📥 Download District Forest Coverage", f, file_name="district_forest_coverage.csv")

        with open(dataset_options[selected_dataset]["raw_paths"][1], "rb") as f:
            st.download_button("📥 Download Province Forest Coverage", f, file_name="province_forest_coverage.csv")

# Feedback Page
def feedback_page():
    st.title("💬 Provide Feedback")
    
    st.markdown("""
    ### Help Us Improve
    
    Your insights are crucial in enhancing our forest fire prediction system.
    """)
    
    # Feedback form
    name = st.text_input("Your Name")
    email = st.text_input("Email Address")
    feedback = st.text_area("Your Feedback")
    
    if st.button("Submit Feedback"):
        if not name or not email or not feedback:
            st.warning("Please fill out all fields before submitting.")
        else:
            # Prepare feedback entry
            feedback_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Name": name,
                "Email": email,
                "Feedback": feedback
            }
            
            # Save to CSV
            feedback_file = "feedback_data.csv"
            file_exists = os.path.exists(feedback_file)
            
            try:
                df = pd.DataFrame([feedback_entry])
                df.to_csv(feedback_file, mode='a', header=not file_exists, index=False)
                st.success("Thank you for your feedback! It has been recorded.")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

# Main App Logic
def main():
    page = create_sidebar_menu()
    
    if page == "home":
        home_page()
    elif page == "map":
        interactive_map_page()
    elif page == "visualization":
        data_visualization_page()
    elif page == "prediction":
        model_prediction_page()
    elif page == "datasets":
        datasets_page()
    elif page == "feedback":
        feedback_page()

if __name__ == "__main__":
    main()