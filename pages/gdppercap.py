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
import streamlit.components.v1 as components
from folium import FeatureGroup, LayerControl, Element


def make_sidebar():
        st.sidebar.markdown("### Menú")
        st.sidebar.page_link("pages/corruption.py", label="Corrupción", icon="⭐")
        st.sidebar.page_link("pages/education.py", label="Educación", icon="⭐")
        st.sidebar.page_link("pages/gdpevolution.py", label="Evolución GDP per cápita", icon="⭐")
        st.sidebar.page_link("pages/gdppercap.py", label="GDP per capita PPP", icon="⭐")
        st.sidebar.page_link("pages/unemployment.py", label="Tasa de desempleo", icon="⭐")


def create_custom_legend():
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 110px; 
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; padding: 10px;">
    <h4>Legend</h4>
    <i style="background: #ffffcc; width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Q1: 0-25%<br>
    <i style="background: #c2e699; width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Q2: 26-50%<br>
    <i style="background: #78c679; width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Q3: 51-75%<br>
    <i style="background: #238443; width: 18px; height: 18px; float: left; margin-right: 8px;"></i> Q4: 76-100%<br>
    </div>
    '''
    return legend_html


def add_legend_to_map(m, legend_html):
    legend = Element(legend_html)
    m.get_root().html.add_child(legend)
    return m


def mapping_demo():
    @st.cache_data
    def load_data():
        europe_df = gpd.read_file('europe/europe.shp')
        country_name_mapping = {
            "Türkiye, Republic of" : "Türkiye",
            "Czech Republic" : "Czechia",
            "North Macedonia " : "North Macedonia",
            "Slovak Republic" : "Slovakia",
            "United Kingdom" : "United Kingdom"
        }

        def rename_country_names(df, column, country_name_mapping):
            df[column] = df[column].replace(country_name_mapping)
            return df

        gdp_percap_df = pd.read_excel("gdp_percapita_ppp.xls")
        gdp_percap_df = gdp_percap_df.rename({"GDP per capita, current prices (Purchasing power parity; international dollars per capita)": "Country Code"}, axis=1)
        gdp_percap_df = rename_country_names(gdp_percap_df, 'Country Code', country_name_mapping)
        gdp_percap_df = gdp_percap_df.loc[:, ["Country Code"]+[year for year in range(1995, 2025)]][gdp_percap_df["Country Code"].isin(europe_df["NAME_ENGL"].unique())]
        gdp_percap_df = pd.concat([gdp_percap_df.iloc[:, :1], gdp_percap_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')], axis=1)
        
        gdp_percap_df_merged = europe_df.merge(gdp_percap_df, how="left", left_on="NAME_ENGL", right_on="Country Code").drop(["Country Code"], axis=1)
        return gdp_percap_df_merged
    
    
    def display_map(df, year):
        map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)
        

        choropleth = folium.Choropleth(
            geo_data="europe.geojson",
            data=df,
            columns = ["NAME_ENGL", year],
            key_on="feature.properties.NAME",
            line_opacity=0.5,
            highlight=True,
            fill_color="YlGn",
            legend_name=f"GDP per capita (PPP) in {year}",
            nan_fill_color="gray",
            nan_fill_opacity=0.05,
            fill_opacity=1,
        )
        choropleth.geojson.add_to(map)
        # choropleth.add_to(map)

        for feature in choropleth.geojson.data['features']:
            country_name = feature['properties']["NAME"]
            feature['properties']['gdp_per_cap'] = 'GDP per capita: ' + str(df.loc[df["NAME_ENGL"] == country_name, year].values[0]) + ' PPP$' if country_name in df["NAME_ENGL"].unique() else "N/A"

        # choropleth.geojson = df.geometry
        # choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields=["NAME_ENGL", 2023], aliases=["Country", "GDP per capita"]))
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['NAME', 'gdp_per_cap'], labels=False)
            )
        

        st_map = st_folium(map, width=750, height=600)
        # st_map = folium_static(map, width=750, height=600)
        
        start_year, end_year = 2000, int(year)
        if st_map['last_active_drawing']:
            
            s = f"{st_map['last_active_drawing']['properties']['NAME']} GDP per capita evolution from {start_year} to {end_year}"
            st.write(s)

            gdp_evo = df.loc[df["NAME_ENGL"] == st_map['last_active_drawing']['properties']['NAME'], [year for year in range(start_year, end_year+1)]]
            gdp_evo.columns = gdp_evo.columns.astype(str)
            gdp_evo = gdp_evo.T
            # st.write(gdp_evo)

            st.line_chart(gdp_evo, use_container_width=True)
    
    
    def display_map_quartiles(data, year):
        # year = str(year)
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
        # st.write(data)
        if st_map['last_active_drawing']:
            selected_country = st_map['last_active_drawing']['properties']['NAME']
            s = f"{selected_country} GDP per capita rank from {start_year} to {end_year}"
            st.write(s)
            
            historical_rank_data = []
            # st.write(data)

            for yr in range(start_year, end_year + 1):
                year_str = int(yr)
                # st.write(data.columns)
                if year_str in data.columns:
                    # st.write("sfasdfasd")
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
        year = st.slider("Year", min_value=2000, max_value=2023, value=2023, step=1)
    if data_type == "Observed Data":
        display_map(data, year)
    else:
        display_map_quartiles(data, year)
        
        
        # map = display_map_quartiles(data, year)
        # legend_html = create_custom_legend()
        # map = add_legend_to_map(map, legend_html)
        # map_html = map._repr_html_()
        # components.html(map_html, height=600)

        
        
        # map_html = map._repr_html_()
        # test1 = components.html(map_html, height=500)
        # st.write(test1)
    
    
st.title("Europe GDP per capita")

make_sidebar()


mapping_demo()