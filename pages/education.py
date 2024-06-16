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
import folium.plugins
import altair as alt
from jinja2 import Template
import branca
from branca.element import Template, MacroElement

def make_sidebar():
        st.sidebar.markdown("### Menú")
        st.sidebar.page_link("pages/corruption.py", label="Corrupción", icon="⭐")
        st.sidebar.page_link("pages/education.py", label="Educación", icon="⭐")
        st.sidebar.page_link("pages/gdpevolution.py", label="Evolución GDP per cápita", icon="⭐")
        st.sidebar.page_link("pages/gdppercap.py", label="GDP per capita PPP", icon="⭐")
        st.sidebar.page_link("pages/unemployment.py", label="Tasa de desempleo", icon="⭐")

def mapping_demo():
    @st.cache_data
    def load_data_europe():
        return gpd.read_file('europe/europe.shp')

    @st.cache_data
    def load_data_pisa(_europe_df, test_type):
        
        if test_type == "Science":
            pisa_science = pd.read_csv("pisa_science_clean.csv")
            pisa_science = _europe_df.merge(pisa_science, how="left", left_on="NAME_ENGL", right_on="Country").drop("Country", axis=1)
            return pisa_science
        elif test_type == "Maths":
            pisa_maths = pd.read_csv("pisa_maths_clean.csv")
            pisa_maths = _europe_df.merge(pisa_maths, how="left", left_on="NAME_ENGL", right_on="Country").drop("Country", axis=1)
            return pisa_maths
        elif test_type == "Reading":
            pisa_reading = pd.read_csv("pisa_reading_clean.csv")
            pisa_reading = _europe_df.merge(pisa_reading, how="left", left_on="NAME_ENGL", right_on="Country").drop("Country", axis=1)
            return pisa_reading
        else:
            pisa_reading = pd.read_csv("pisa_reading_clean.csv")
            pisa_science = pd.read_csv("pisa_science_clean.csv")
            pisa_maths = pd.read_csv("pisa_maths_clean.csv")

            cols = pisa_science.columns[1:]
            means = []
            for a, b, c in zip(pisa_reading[cols].iterrows(), pisa_maths[cols].iterrows(), pisa_science[cols].iterrows()):
                year = a[1]
                a, b, c = a[1].to_numpy(), b[1].to_numpy(), c[1].to_numpy()
                arr = np.array([a,b,c])
                mean = np.round(np.nanmean(arr, axis=0))
                means.append(mean)

            cols = pisa_reading.columns[6:]
            for idx, (a, b) in enumerate(zip(pisa_reading[cols].iterrows(), pisa_maths[cols].iterrows())):
                year = a[1]
                a, b = a[1].to_numpy(), b[1].to_numpy()
                arr = np.array([a,b])
                mean = np.round(np.nanmean(arr, axis=0))
                means[idx] = np.append(means[idx], mean)

            pisa_means_df = pd.DataFrame(means, columns=pisa_reading.columns[1:])
            pisa_means_df["Country"] = pisa_reading["Country"]
            cols = pisa_means_df.columns
            cols = cols[-1:].append(cols[:-1])
            pisa_means_df = pisa_means_df[cols]

            pisa_merged_df = _europe_df.merge(pisa_means_df, how="left", left_on="NAME_ENGL", right_on="Country").drop("Country", axis=1)

            return pisa_merged_df
        
    @st.cache_data
    def load_data_spending(_europe_df):
        edu_spending_df = pd.read_csv("total-government-expenditure-on-education-gdp.csv")
        edu_spending_df = edu_spending_df.pivot(index='Entity', columns='Year', values='Historical and more recent expenditure estimates').reset_index()
        edu_spending_df = edu_spending_df[edu_spending_df["Entity"].isin(_europe_df["NAME_ENGL"].unique())].reset_index().drop(["index"],axis=1)
        edu_spending_df = edu_spending_df.loc[:, ["Entity"]+[year for year in range(1995, 2023)]]
        edu_spending_df.columns = edu_spending_df.columns.astype(str)
        edu_spending_merged = _europe_df.merge(edu_spending_df, how="left", left_on="NAME_ENGL", right_on="Entity").drop("Entity", axis=1)
        
        return edu_spending_merged
    
    
    def display_map_pisa(df_pisa, df_spend, year, test_type):
        year = str(year)


        c1, c2 = st.columns(2)

        with c1:
            map = folium.Map(location=[56, 118], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)

            # PISA DATA
            choropleth_pisa = folium.Choropleth(
                geo_data="europe.geojson",
                data=df_pisa,
                columns = ["NAME_ENGL", year],
                key_on="feature.properties.NAME",
                line_opacity=0.5,
                highlight=True,
                fill_color="YlGn",
                legend_name=f"CPI index in {year}",
                nan_fill_color="gray",
                nan_fill_opacity=0.05,
            )
            choropleth_pisa.geojson.add_to(map)

            for feature in choropleth_pisa.geojson.data['features']:
                country_name = feature['properties']["NAME"]
                feature['properties']['pisa'] = f'PISA {test_type} : ' + str(round(df_pisa.loc[df_pisa["NAME_ENGL"] == country_name, year].values[0], 2)) + '' if country_name in df_pisa["NAME_ENGL"].unique() else "N/A"

            
            choropleth_pisa.geojson.add_child(folium.features.GeoJsonTooltip(['NAME', 'pisa'], labels=False))
            st_map = st_folium(map, width=600, height=600)

            if st_map['last_active_drawing']:
                st.write(f"{st_map['last_active_drawing']['properties']['NAME']} selected in left map") 

            start_year, end_year = 2000, 2018
            if st_map['last_active_drawing']:
                selected_country = st_map['last_active_drawing']['properties']['NAME']
                st.write(country_name)
                s = f"{selected_country} PISA {test_type} score {start_year} to {end_year}"
                st.write(s)

                gdp_evo = df_pisa.loc[df_pisa["NAME_ENGL"] == selected_country, [str(year) for year in range(start_year, end_year+1, 3)]]
                gdp_evo.columns = gdp_evo.columns.astype(str)
                gdp_evo = gdp_evo.T

                st.line_chart(gdp_evo, use_container_width=True)
        
        with c2:

            map = folium.Map(location=[56, 118], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)
            # EDU SPENDING DATA
            choropleth_spend = folium.Choropleth(
                geo_data="europe.geojson",
                data=df_spend,
                columns = ["NAME_ENGL", year],
                key_on="feature.properties.NAME",
                line_opacity=0.5,
                highlight=True,
                fill_color="YlGn",
                legend_name=f"CPI index in {year}",
                nan_fill_color="gray",
                nan_fill_opacity=0.05,
            )
            choropleth_spend.geojson.add_to(map)

            


            for feature in choropleth_spend.geojson.data['features']:
                country_name = feature['properties']["NAME"]
                feature['properties']['spend'] = f'Education spending  : ' + str(round(df_spend.loc[df_spend["NAME_ENGL"] == country_name, year].values[0], 2)) + ' (% GDP)' if country_name in df_pisa["NAME_ENGL"].unique() else "N/A"

            choropleth_spend.geojson.add_child(
                folium.features.GeoJsonTooltip(['NAME', 'spend'], labels=False)
                )

            
            st_map = st_folium(map, width=600, height=600)
            # st.write(map)
            if st_map['last_active_drawing']:
                st.write(f"{st_map['last_active_drawing']['properties']['NAME']} selected in left map")
            
            start_year, end_year = 2000, 2018
            if st_map['last_active_drawing']:
                selected_country = st_map['last_active_drawing']['properties']['NAME']
                st.write(country_name)
                s = f"{selected_country} PISA {test_type} score {start_year} to {end_year}"
                st.write(s)

                gdp_evo = df_spend.loc[df_spend["NAME_ENGL"] == selected_country, [str(year) for year in range(start_year, end_year+1, 3)]]
                gdp_evo.columns = gdp_evo.columns.astype(str)
                gdp_evo = gdp_evo.T

                st.line_chart(gdp_evo, use_container_width=True)

    def display_map(df_pisa, df_spend, year, test_type):
            year = str(year)
            map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)

            
            # PISA DATA
            choropleth_pisa = folium.Choropleth(
                geo_data="europe.geojson",
                data=df_pisa,
                columns = ["NAME_ENGL", year],
                key_on="feature.properties.NAME",
                line_opacity=0.5,
                highlight=True,
                fill_color="YlGn",
                legend_name=f"CPI index in {year}",
                nan_fill_color="gray",
                nan_fill_opacity=0.05,
            )
            choropleth_pisa.geojson.add_to(map)

            for feature in choropleth_pisa.geojson.data['features']:
                country_name = feature['properties']["NAME"]
                feature['properties']['pisa'] = f'PISA {test_type} : ' + str(round(df_pisa.loc[df_pisa["NAME_ENGL"] == country_name, year].values[0], 2)) + '' if country_name in df_pisa["NAME_ENGL"].unique() else "N/A"

            
            choropleth_pisa.geojson.add_child(
                folium.features.GeoJsonTooltip(['NAME', 'pisa'], labels=False)
                )
            

            # # EDU SPENDING DATA
            # choropleth_spend = folium.Choropleth(
            #     geo_data="europe.geojson",
            #     data=df_spend,
            #     columns = ["NAME_ENGL", year],
            #     key_on="feature.properties.NAME",
            #     line_opacity=0.5,
            #     highlight=True,
            #     fill_color="YlGn",
            #     legend_name=f"CPI index in {year}",
            #     nan_fill_color="gray",
            #     nan_fill_opacity=0.05,
            # )
            # choropleth_spend.geojson.add_to(map.m2)

            # for feature in choropleth_spend.geojson.data['features']:
            #     country_name = feature['properties']["NAME"]
            #     feature['properties']['spend'] = f'Education spending  : ' + str(round(df_spend.loc[df_spend["NAME_ENGL"] == country_name, year].values[0], 2)) + ' (% GDP)' if country_name in df_pisa["NAME_ENGL"].unique() else "N/A"

            # choropleth_spend.geojson.add_child(
            #     folium.features.GeoJsonTooltip(['NAME', 'spend'], labels=False)
            #     )

            
            # st_map = st_folium(map, width=1500, height=600)
            # # st.write(map)
            
            
            st_map = st_folium(map, width=750, height=600)
            start_year, end_year = 2000, 2018
            if st_map['last_active_drawing']:
                s = f"PISA {test_type} in {st_map['last_active_drawing']['properties']['NAME']} from {start_year} to {end_year}"
                st.write(s)

                gdp_evo = df_pisa.loc[df_pisa["NAME_ENGL"] == st_map['last_active_drawing']['properties']['NAME'], [str(year) for year in range(start_year, end_year+1, 3)]]
                gdp_evo.columns = gdp_evo.columns.astype(str)
                gdp_evo = gdp_evo.T

                st.line_chart(gdp_evo, use_container_width=True)

    def display_map_test(df_pisa, df_spend, year, test_type):
                year = str(year)
                map = folium.Map(location=[56, 16], zoom_start=3.4, tiles="cartodbpositron", zoom_control=False, scrollWheelZoom=False, dragging=False)

                
                # PISA DATA
                choropleth_pisa = folium.Choropleth(
                    geo_data="europe.geojson",
                    data=df_pisa,
                    columns = ["NAME_ENGL", year],
                    key_on="feature.properties.NAME",
                    line_opacity=0.5,
                    highlight=True,
                    fill_color="YlGn",
                    legend_name=f"CPI index in {year}",
                    nan_fill_color="gray",
                    nan_fill_opacity=0.05,
                )
                choropleth_pisa.geojson.add_to(map)
                
                # macro = MacroElement()
                # macro._template = Template(template)
                # map.get_root().add_child(macro)


                for feature in choropleth_pisa.geojson.data['features']:
                    c_name = feature['properties']["NAME"]
                    feature['properties']['pisa'] = f'PISA {test_type} : ' + str(round(df_pisa.loc[df_pisa["NAME_ENGL"] == c_name, year].values[0], 2)) + '' if c_name in df_pisa["NAME_ENGL"].unique() else "N/A"

                
                choropleth_pisa.geojson.add_child(
                    folium.features.GeoJsonTooltip(['NAME', 'pisa'], labels=False)
                    )                
                
                st_map = st_folium(map, width=750, height=600)
                # st_map = folium_static(map, width=750, height=600)
                
                x_label = 'Education Spending(% GDP)'  # Replace with your descriptive x axis label
                y_label = f'PISA {test_type} Score'  # Replace with your descriptive y axis label
                start_year, end_year = 2000, int(year)
                if st_map['last_active_drawing']:
                    selected_country_name = st_map['last_active_drawing']['properties']['NAME']
                    temp = pd.merge(df_pisa[["NAME_ENGL", year]], df_spend[["NAME_ENGL", year]], on="NAME_ENGL").rename({year+"_x": "y", year+"_y": "x"}, axis=1)
                    temp['highlight'] = temp['NAME_ENGL'] == selected_country_name
                    
                    
                    title = alt.TitleParams(f'Pisa {test_type} Score vs Education Spending (%GPD) for European Countries', anchor='middle')
                    c = alt.Chart(temp, title=title).mark_circle(size=60).encode(
                        x=alt.X('x', title=x_label, scale=alt.Scale(domain=[2, 9], bins=[2, 3, 4, 5, 6, 7, 8, 9])),
                        y=alt.Y('y', title=y_label, scale=alt.Scale(domain=[390, 550], bins=[390, 420, 450, 480, 510, 550])),
                        color=alt.condition(
                            alt.datum.highlight,
                            alt.value('red'),  # Color for the highlighted country
                            alt.value('green')  # Color for other countries (change this to suit your data)
                        ),
                            size=alt.condition(
                                                alt.datum.highlight,
                                                alt.value(150),  # Size for the highlighted country
                                                alt.value(60)  # Size for other countries
                                            ),
                        tooltip=[
                            alt.Tooltip('NAME_ENGL', title='Country'),  # Country name
                            alt.Tooltip('x', title=x_label),  # Descriptive x value
                            alt.Tooltip('y', title=y_label)   # Descriptive y value
                        ]
                    )

                    st.altair_chart(c, use_container_width=True)


                    # Get PISA scores for the selected country
                    pisa_evo = df_pisa.loc[df_pisa["NAME_ENGL"] == selected_country_name, ["NAME_ENGL"] + [str(year) for year in range(start_year, end_year + 1, 3)]]
                    pisa_evo = pisa_evo.melt(id_vars=["NAME_ENGL"], var_name='Year', value_name='PISA Score')

                    # Get GDP spending for the selected country
                    gdp_evo = df_spend.loc[df_spend["NAME_ENGL"] == selected_country_name, ["NAME_ENGL"] + [str(year) for year in range(start_year, end_year + 1, 3)]]
                    gdp_evo = gdp_evo.melt(id_vars=["NAME_ENGL"], var_name='Year', value_name='GDP Spending')

                    # Add a column to identify the line in the legend
                    pisa_evo['Indicator'] = 'PISA Score'
                    gdp_evo['Indicator'] = 'GDP Spending'

                    # Define scales for both y-axes
                    y_scale_left = alt.Scale(domain=[400, 600])  # Adjust this range as needed for PISA scores
                    y_scale_right = alt.Scale(domain=[0, 10])  # Adjust this range as needed for GDP spending

                    # Create PISA score chart
                    pisa_chart = alt.Chart(pisa_evo).mark_line().encode(
                        x=alt.X('Year:T', axis=alt.Axis(title='Year')),
                        y=alt.Y('PISA Score:Q', axis=alt.Axis(title='PISA Score'), scale=y_scale_left),
                        color=alt.Color('Indicator:N', scale=alt.Scale(domain=['PISA Score', 'GDP Spending'], range=['blue', 'green']), legend=alt.Legend(title="Indicator"))
                    )

                    # Create GDP spending chart
                    gdp_chart = alt.Chart(gdp_evo).mark_line().encode(
                        x=alt.X('Year:T', axis=alt.Axis(title='Year')),
                        y=alt.Y('GDP Spending:Q', axis=alt.Axis(title='GDP Spending', orient='right'), scale=y_scale_right),
                        color=alt.Color('Indicator:N', scale=alt.Scale(domain=['PISA Score', 'GDP Spending'], range=['blue', 'green']), legend=alt.Legend(title="Indicator"))
                    )

                    # Combine both charts using layer
                    title = alt.TitleParams(f'Pisa {test_type} Score and Education Spending (%GDP) for {selected_country_name}', anchor='middle')
                    combined_chart = alt.layer(
                        pisa_chart,
                        gdp_chart, title=title
                    ).resolve_scale(
                        y='independent'  # Use independent scales for the y-axes
                    ).properties(
                        width=600,
                        height=400
                    )

                    # Display the chart in Streamlit
                    st.altair_chart(combined_chart, use_container_width=True)


                else:
                    temp = pd.merge(df_pisa[["NAME_ENGL", year]], df_spend[["NAME_ENGL", year]], on="NAME_ENGL").rename({year+"_x": "y", year+"_y": "x"}, axis=1)
                    title = alt.TitleParams(f'Pisa {test_type} Score vs Education Spending (%GPD) for European Countries in {year}', anchor='middle')
                    c = alt.Chart(temp, title=title).mark_circle(size=60).encode(
                        x=alt.X('x', title=x_label, scale=alt.Scale(domain=[2, 9], bins=[2, 3, 4, 5, 6, 7, 8, 9])),
                        y=alt.Y('y', title=y_label, scale=alt.Scale(domain=[390, 550], bins=[390, 420, 450, 480, 510, 550])),
                        color=alt.value('green'),
                        tooltip=[
                            alt.Tooltip('NAME_ENGL', title='Country'),  # Country name
                            alt.Tooltip('x', title=x_label),  # Descriptive x value
                            alt.Tooltip('y', title=y_label)   # Descriptive y value
                        ]
                    )

                    st.altair_chart(c, use_container_width=True)

    
    
    
    test_type = st.selectbox("Pisa Test", ["Average", "Reading", "Maths", "Science"], index=0)
    europe_df = load_data_europe()
    data_pisa = load_data_pisa(europe_df, test_type)
    data_spend = load_data_spending(europe_df)
    min_value = 2000 if test_type != "Science" else 2006

    year = st.slider("Year", min_value=min_value, max_value=2018, value=2018, step=3)
    # display_map_pisa(data_pisa, data_spend, year, test_type)
    # display_map(data_pisa, data_spend, year, test_type)
    display_map_test(data_pisa, data_spend, year, test_type)
    
    

st.title("Prueba PISA en Europa")

make_sidebar()
# st.set_page_config(page_title="Europe PISA Scores", page_icon="🌍")
# st.markdown("Europe PISA Scores")
# st.sidebar.header("Europe PISA Scores")


mapping_demo()