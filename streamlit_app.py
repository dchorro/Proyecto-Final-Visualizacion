import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():

    # def make_sidebar():
    #     st.sidebar.markdown("### Menú")
    #     st.sidebar.page_link("pages/corruption.py", label="Corrupción", icon="⭐")
    #     st.sidebar.page_link("pages/education.py", label="Educación", icon="⭐")
    #     st.sidebar.page_link("pages/gdpevolution.py", label="Evolución GDP per cápita", icon="⭐")
    #     st.sidebar.page_link("pages/gdppercap.py", label="GDP per capita PPP", icon="⭐")
    #     st.sidebar.page_link("pages/unemployment.py", label="Tasa de desempleo", icon="⭐")

    # make_sidebar()
    st.switch_page("pages/gdppercap.py")



    # st.set_page_config(
    #     page_title="Hello",
    #     page_icon="👋",
    # )

    # st.write("# Welcome to Streamlit! 👋")

    # st.sidebar.success("Select a demo above.")

    # st.markdown(
    #     """
    #     Streamlit is an open-source app framework built specifically for
    #     Machine Learning and Data Science projects.
    #     **👈 Select a demo from the sidebar** to see some examples
    #     of what Streamlit can do!
    #     ### Want to learn more?
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into our [documentation](https://docs.streamlit.io)
    #     - Ask a question in our [community
    #     forums](https://discuss.streamlit.io)
    #     ### See more complex demos
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #     Dataset](https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # """
    # )


if __name__ == "__main__":
    run()
