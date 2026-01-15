import streamlit as st
import requests

st.title("Smart Surveillance Dashboard")

alerts = requests.get("http://127.0.0.1:5000/alerts").json()

st.subheader("Alerts")
for alert in alerts:
    st.write(f"{alert['time']} - {alert['msg']}")
