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
from branca.element import Template, MacroElement
import streamlit_folium


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
        unemployment_df = pd.read_csv("unemployment_data.csv")
        unemployment_df = unemployment_df.loc[:, ["Country Name", "Country Code"]+[str(year) for year in range(1995, 2024)]][unemployment_df["Country Code"].isin(europe_df["ISO3_CODE"].unique())]
        unemployment_df_merged = europe_df.merge(unemployment_df, how="left", left_on="ISO3_CODE", right_on="Country Code").drop(["Country Code", "Country Name"], axis=1)

        def calculate_returns(df, initial_year=2000, final_year=2023):
            average_unemployment = df.loc[:, [str(year) for year in range(initial_year, final_year+1)]].mean(axis=1)
            return average_unemployment
        unemployment_df_merged['Average Unemployment'] = calculate_returns(unemployment_df_merged, initial_year=2000, final_year=2023)

        return unemployment_df_merged
    
    
    def display_map(df, year=2020):
        year = str(year)
        map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)

        # europe_df = gpd.read_file('europe/europe.shp')
        # st.write(europe_df)
        choropleth = folium.Choropleth(
            geo_data="europe.geojson",
            data=df,
            columns = ["NAME_ENGL", year],
            key_on="feature.properties.NAME",
            line_opacity=0.5,
            highlight=True,
            fill_color="YlGn",
            legend_name=f"Unemployment rate in {year}",
            nan_fill_color="gray",
            nan_fill_opacity=0.05,
            fill_opacity=1,
            show=True
        )
        choropleth.geojson.add_to(map)
        # choropleth.add_to(map)

        for feature in choropleth.geojson.data['features']:
            country_name = feature['properties']["NAME"]
            feature['properties']['unemployment'] = 'Unemployment : ' + str(round(df.loc[df["NAME_ENGL"] == country_name, year].values[0], 2)) + ' %' if country_name in df["NAME_ENGL"].unique() else "N/A"

        
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['NAME', 'unemployment'], labels=False)
            )
        

        st_map = st_folium(map, width=750, height=600)
        # st_map = folium_static(map, width=750, height=600)
        
        if year == "Average Unemployment":
            start_year, end_year = 2000, 2023
        else:
            start_year, end_year = 2000, int(year)


        if st_map['last_active_drawing']:
            
            s = f"{st_map['last_active_drawing']['properties']['NAME']} unemployment rate from {start_year} to {end_year} (%)"
            st.write(s)

            gdp_evo = df.loc[df["NAME_ENGL"] == st_map['last_active_drawing']['properties']['NAME'], [str(year) for year in range(start_year, int(end_year)+1)]]
            gdp_evo.columns = gdp_evo.columns.astype(str)
            gdp_evo = gdp_evo.T

            st.line_chart(gdp_evo, use_container_width=True)
    
    
    def add_legend(map_obj, title, colors, labels):
        legend_html = f'''
        <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 120px; 
        border:2px solid grey; z-index:9999; font-size:14px; background-color: white;">
        &nbsp;<b>{title}</b><br>
        &nbsp;<i style="background: {colors[0]}; width: 10px; height: 10px; display: inline-block;"></i>&nbsp;{labels[0]}<br>
        &nbsp;<i style="background: {colors[1]}; width: 10px; height: 10px; display: inline-block;"></i>&nbsp;{labels[1]}<br>
        &nbsp;<i style="background: {colors[2]}; width: 10px; height: 10px; display: inline-block;"></i>&nbsp;{labels[2]}<br>
        &nbsp;<i style="background: {colors[3]}; width: 10px; height: 10px; display: inline-block;"></i>&nbsp;{labels[3]}
        </div>
        '''

        macro = MacroElement()
        macro._template = Template(legend_html)
        map_obj.get_root().add_child(macro)
        # map_obj.get_root().html.add_child(folium.Element(legend_html))

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

        # Add custom legend
        colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8']  # YlGn colormap colors
        labels = ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile']
        add_legend(map, 'Quartiles', colors, labels)

        # Display the map in Streamlit
        st_map = st_folium(map, width=750, height=600)
        # st_map = folium_static(map, width=750, height=600)
        
        if year == "Average Unemployment":
            start_year, end_year = 2000, 2023
        else:
            start_year, end_year = 2000, int(year)
        if st_map['last_active_drawing']:
            selected_country = st_map['last_active_drawing']['properties']['NAME']
            s = f"{selected_country} unemployment rank from {start_year} to {end_year}"
            st.write(s)

            historical_rank_data = []
            for yr in range(start_year, end_year + 1):
                year_str = str(yr)
                if year_str in data.columns:
                    yearly_data = data[['NAME_ENGL', year_str]].dropna()
                    yearly_data['Rank'] = yearly_data[year_str].rank(ascending=True, method="first")
                    if selected_country in yearly_data['NAME_ENGL'].values:
                        country_rank = yearly_data[yearly_data['NAME_ENGL'] == selected_country]['Rank'].values[0]
                        historical_rank_data.append((year_str, country_rank))

            historical_rank_df = pd.DataFrame(historical_rank_data, columns=['Year', 'Rank'])
            historical_rank_df = historical_rank_df.set_index('Year')

            # Prepare the data for the selected country
            st.line_chart(historical_rank_df, use_container_width=True)


    data = load_data()
    c1, c2 = st.columns([0.7, 0.3])
    with c2:
        data_type = st.radio("Select the data you want to see", options=["Observed Data", "Quartiles"])
    with c1:
        year = st.select_slider("Year", options=[year for year in range(2000, 2024)]+["Average Unemployment"], value="Average Unemployment")
    if data_type == "Observed Data":
        display_map(data, year)
    else:
        display_map_quartiles(data, year)

    
    


st.title("Tasa de desempleo en Europa")
make_sidebar()
# st.set_page_config(page_title="Europe Unemployment Rate", page_icon="🌍", )
# st.markdown("Europe Unemployment Rate")
# st.sidebar.header("Europe Unemployment Rate")


mapping_demo()