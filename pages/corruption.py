import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium, folium_static


def make_sidebar():
        st.sidebar.markdown("### Menú")
        st.sidebar.page_link("pages/corruption.py", label="Corrupción", icon="⭐")
        st.sidebar.page_link("pages/education.py", label="Educación", icon="⭐")
        st.sidebar.page_link("pages/gdpevolution.py", label="Evolución GDP per cápita", icon="⭐")
        st.sidebar.page_link("pages/gdppercap.py", label="GDP per capita PPP", icon="⭐")
        st.sidebar.page_link("pages/unemployment.py", label="Tasa de desempleo", icon="⭐")



def mapping_demo():
    @st.cache_data
    def load_data():
        europe_df = gpd.read_file('europe/europe.shp')
        
        
        cpi_df = pd.read_csv("cpi_1998_2015.csv")
        corr_df = pd.read_csv("ti-corruption-perception-index.csv")
        corr_df = corr_df.pivot(index='Entity', columns='Year', values='Corruption Perception Index - Transparency International (2018)').reset_index()
        corr_df.columns = corr_df.columns.astype(str)
        cpi_df = cpi_df.merge(corr_df[['Entity', "2016", "2017", "2018"]], left_on="Jurisdiction", right_on="Entity", how="left").drop("Entity", axis=1)
        year_range = [str(year) for year in range(1998, 2012)]
        cpi_df = cpi_df.replace('-', np.nan)
        cpi_df.iloc[:, 1:] = cpi_df.iloc[:, 1:].astype(float)
        cpi_df[year_range] = cpi_df[year_range] * 10
        cpi_df_merged = europe_df.merge(cpi_df, how="left", left_on="NAME_ENGL", right_on="Jurisdiction").drop("Jurisdiction", axis=1)

        return cpi_df_merged
    
    
    def display_map(df, year):
        year = str(year)
        map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)


        choropleth = folium.Choropleth(
            geo_data="europe.geojson",
            data=df,
            columns = ["NAME_ENGL", year],
            key_on="feature.properties.NAME",
            line_opacity=0.5,
            highlight=True,
            fill_color="YlGn",
            legend_name=f"CPI index in {year}",
            nan_fill_color="gray",
            nan_fill_opacity=0.05,
        )
        choropleth.geojson.add_to(map)
        # choropleth.add_to(map)

        for feature in choropleth.geojson.data['features']:
            country_name = feature['properties']["NAME"]
            feature['properties']['cpi'] = 'CPI : ' + str(round(df.loc[df["NAME_ENGL"] == country_name, year].values[0], 2)) + '' if country_name in df["NAME_ENGL"].unique() else "N/A"

        
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['NAME', 'cpi'], labels=False)
            )
        

        st_map = st_folium(map, width=750, height=600)
        # st_map = folium_static(map, width=750, height=600)
        
        start_year, end_year = 2000, int(year)
        if st_map['last_active_drawing']:
            
            s = f"{st_map['last_active_drawing']['properties']['NAME']} Corruption Perceptions Index from {start_year} to {end_year}"
            st.write(s)

            gdp_evo = df.loc[df["NAME_ENGL"] == st_map['last_active_drawing']['properties']['NAME'], [str(year) for year in range(start_year, end_year+1)]]
            gdp_evo.columns = gdp_evo.columns.astype(str)
            gdp_evo = gdp_evo.T

            st.line_chart(gdp_evo, use_container_width=True)


    def display_map_quartiles(data, year):
        year = str(year)
        data[['NAME_ENGL', year]] = data[['NAME_ENGL', year]].dropna()  # Remove rows with NaN values for the selected year

        # Calculate quartiles
        quartiles = pd.qcut(data[year], 4, labels=[1, 2, 3, 4])
        data['Quartile'] = quartiles

        # Create a Folium map
        map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)

        # Create a choropleth map with quartile coloring
        choropleth = folium.Choropleth(
            geo_data="europe.geojson",
            data=data,
            columns=["NAME_ENGL", 'Quartile'],
            key_on="feature.properties.NAME",
            line_opacity=0.5,
            highlight=True,
            fill_color="YlGn",
            fill_opacity=0.8,
            legend_name=f"Unemployment Rate Quartiles in {year}",
            nan_fill_color="gray",
            nan_fill_opacity=0.05,
        )
        choropleth.geojson.add_to(map)
        # choropleth.add_to(map)

        # Add unemployment rate tooltip
        for feature in choropleth.geojson.data['features']:
            country_name = feature['properties']["NAME"]
            feature['properties']['unemployment'] = 'Unemployment Quartile: ' + str(round(data.loc[data["NAME_ENGL"] == country_name, 'Quartile'].values[0], 2)) if country_name in data["NAME_ENGL"].unique() else "N/A"

        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['NAME', 'unemployment'], labels=False)
        )

        # Display the map in Streamlit
        st_map = st_folium(map, width=750, height=600)
        # st_map = folium_static(map, width=750, height=600)


        # Show line chart of unemployment rate and historical rank over years for selected country
        start_year, end_year = 2000, int(year)
        if st_map['last_active_drawing']:
            selected_country = st_map['last_active_drawing']['properties']['NAME']
            s = f"{selected_country} Corruption Perceptions Index rank from {start_year} to {end_year}"
            st.write(s)
            
            historical_rank_data = []

            for yr in range(start_year, end_year + 1):
                year_str = str(yr)
                if year_str in data.columns:
                    yearly_data = data[['NAME_ENGL', year_str]].dropna()
                    yearly_data['Rank'] = yearly_data[year_str].rank(ascending=False, method="first")  # Use ascending=False for GDP per capita rank (higher value = better rank)
                    if selected_country in yearly_data['NAME_ENGL'].values:
                        country_rank = yearly_data[yearly_data['NAME_ENGL'] == selected_country]['Rank'].values[0]
                        historical_rank_data.append((year_str, country_rank))

            # Convert the historical rank data to a DataFrame
            historical_rank_df = pd.DataFrame(historical_rank_data, columns=['Year', 'Rank'])
            historical_rank_df = historical_rank_df.set_index('Year')

            # Plot the rank data
            st.line_chart(historical_rank_df, use_container_width=True)

    data = load_data()

    c1, c2 = st.columns([0.7, 0.3])
    with c2:
        data_type = st.radio("Select the data you want to see", options=["Observed Data", "Quartiles"])
    with c1:
        year = st.select_slider("Year", options=[year for year in range(2000, 2019)], value=2018)
    if data_type == "Observed Data":
        display_map(data, year)
    else:
        display_map_quartiles(data, year)
    
    


st.title("Corruption Perceptions Index en Europa")
make_sidebar()
# st.set_page_config(page_title="Europe Corruption Perceptions Index (CPI)", page_icon="🌍")
# st.markdown("Europe Corruption Perceptions Index (CPI)")
# st.sidebar.header("Europe Corruption Perceptions Index (CPI)")


mapping_demo()