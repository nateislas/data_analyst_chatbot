#!/bin/bash
# Run the Streamlit app for the Data Analyst Chatbot

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run data_analyst_chatbot/app.py
