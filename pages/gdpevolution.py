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
from streamlit_folium import st_folium
import altair as alt


def make_sidebar():
        st.sidebar.markdown("### Menú")
        st.sidebar.page_link("pages/corruption.py", label="Corrupción", icon="⭐")
        st.sidebar.page_link("pages/education.py", label="Educación", icon="⭐")
        st.sidebar.page_link("pages/gdpevolution.py", label="Evolución GDP per cápita", icon="⭐")
        st.sidebar.page_link("pages/gdppercap.py", label="GDP per capita PPP", icon="⭐")
        st.sidebar.page_link("pages/unemployment.py", label="Tasa de desempleo", icon="⭐")


def normalize_data(df):
    countries = df['NAME_ENGL'].unique()
    normalized_data = pd.DataFrame()
    
    for country in countries:
        country_data = df[df['NAME_ENGL'] == country].copy()
        base_value = country_data.iloc[:, 1].values[0]  # Assuming first year is in the second column
        country_data.iloc[:, 1:] = country_data.iloc[:, 1:].apply(lambda x: (x / base_value) * 100)
        normalized_data = pd.concat([normalized_data, country_data])
    
    return normalized_data


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
    
    
    def display_map(df, beg_year, end_year):
        df = df.set_index('NAME_ENGL').T
        df.index = df.index.astype(str)
        df = df.reset_index().melt(id_vars='index', var_name='Country', value_name='Growth')
        df = df.rename(columns={'index': 'Year'})
        df["Growth"] = df["Growth"].round(2)

        # Create the Altair plot
        line_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('Year', title='Year'),
            y=alt.Y('Growth:Q', title='Accumulated GDP Growth (%)'),
            color=alt.Color('Country:N', legend=alt.Legend(columns=2, symbolLimit=1000)),
            tooltip=['Year', 'Country', 'Growth']
        ).properties(
            title=f'GDP per capita (PPP) in European countries from {beg_year} to {end_year}',
            width=750,  # Increased width
            height=500   # Increased height
        )

        # Rotate the x-axis labels if necessary
        line_chart = line_chart.configure_axisX(
            labelAngle=45  # Rotate x-axis labels for better readability
        )

        # Display the plot in Streamlit
        st.altair_chart(line_chart, use_container_width=True)

    data = load_data()
    beg_year, end_year = st.slider("Year", min_value=2000, max_value=2023, value=(2000, 2023), step=1)
    countries = st.multiselect("Countries", data["NAME_ENGL"].unique(), ["Spain", "France", "Germany", "Italy", "United Kingdom"])
    if countries:
        data = data[data["NAME_ENGL"].isin(countries)]
        years = data.columns[5:]
        filtered_years = [year for year in years if year >= beg_year and year <= end_year]
        data = data.loc[:, ["NAME_ENGL"]+list(filtered_years)]
        normalized_data = normalize_data(data)
        # st.write(normalized_data)
        display_map(normalized_data, beg_year, end_year)
    
    

st.title("Evolución del GDP per cápita en Europa")
make_sidebar()
# st.set_page_config(page_title="Europe GDP per capita (PPP)", page_icon="🌍", )
# st.markdown("Europe GDP per capita (PPP)")
# st.sidebar.header("Europe GDP per capita (PPP)")


mapping_demo()