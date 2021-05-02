import streamlit as st

# Framework for rendering multiple applications
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """
        To add a new application
        -------------------------
        @param (string) title: Title of the application
        @param (func) func: Function to render the application
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        app = st.selectbox(
            'Navigate to other pages',
            self.apps,
            format_func=lambda app: app['title'])
        app['function']()
