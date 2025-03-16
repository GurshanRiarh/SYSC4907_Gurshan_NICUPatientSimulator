#!/usr/bin/env python
"""
Interactive Vital Signs Plotting Module

This module provides functions to create interactive Plotly plots for simulated
vital signs data stored in JSON format. The plots include heart rate, respiratory 
rate, body temperature, and their respective delta (change) values.
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime

# =============================================================================
# Global Variables and Directory Setup
# =============================================================================
# Determine the project output directory dynamically.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Patient_Data")

# =============================================================================
# Main Plotting Functions
# =============================================================================
def plot_interactive_vital_signs(patient_id, shift_length):
    """
    Generate interactive Plotly plots for a patient's vital signs.

    Reads the simulation JSON data for the given patient, extracts the 
    timestamps and various vital sign metrics (heart rate, respiratory rate,
    body temperature, and their delta values), and creates interactive plots 
    saved as HTML files.

    Args:
        patient_id (str): Unique identifier for the patient.
        shift_length (int): The duration of the simulation (in hours).

    Returns:
        None
    """
    json_input_dir = os.path.join(OUTPUT_DIR, patient_id, "json")
    json_file = os.path.join(json_input_dir, f"{patient_id}_Data.json")
    plot_output_dir = os.path.join(OUTPUT_DIR, patient_id, "interactive_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    if not os.path.exists(json_file):
        print(f"Error: JSON file not found at {json_file}")
        return

    # Load simulation data from JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    # Parse timestamps and convert to datetime objects
    timestamps = [entry["timestamp"] for entry in data["data"]]
    time_values = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Extract vital sign values from JSON data
    heart_rates = [int(entry["vital_signs"]["heart_rate"].split()[0]) for entry in data["data"]]
    respiratory_rates = [int(entry["vital_signs"]["respiratory_rate"].split()[0]) for entry in data["data"]]
    body_temperatures = [float(entry["vital_signs"]["body_temperature"].split()[0]) for entry in data["data"]]

    # Extract delta values for vital signs (if available)
    hr_deltas, resp_rate_deltas, temp_deltas = [], [], []
    valid_time_values = []  # Timestamps corresponding to HR delta values
    for i, entry in enumerate(data["data"]):
        if "HR_Delta" in entry["vital_signs"] and entry["vital_signs"]["HR_Delta"] is not None:
            # Remove any '%' sign before converting to float
            hr_deltas.append(float(entry["vital_signs"]["HR_Delta"].split()[0].replace('%','')))
            valid_time_values.append(time_values[i])
        if "RespRate_Delta" in entry["vital_signs"] and entry["vital_signs"]["RespRate_Delta"] is not None:
            resp_rate_deltas.append(float(entry["vital_signs"]["RespRate_Delta"].split()[0]))
        if "Temp_Delta" in entry["vital_signs"] and entry["vital_signs"]["Temp_Delta"] is not None:
            temp_deltas.append(float(entry["vital_signs"]["Temp_Delta"].split()[0]))

    # Determine plot titles and y-axis labels based on patient conditions
    condition = data.get("Conditions", [])
    if "Bradycardia" in condition and "Tachycardia" in condition:
        hr_title = "Heart Rate - Extreme Bradycardia & Tachycardia Swings"
        hr_ylabel = "Heart Rate (bpm) [40-230]"
    elif "Bradycardia" in condition:
        hr_title = "Heart Rate - Bradycardia"
        hr_ylabel = "Heart Rate (bpm) [70-140]"
    elif "Tachycardia" in condition:
        hr_title = "Heart Rate - Tachycardia"
        hr_ylabel = "Heart Rate (bpm) [160-230]"
    else:
        hr_title = "Heart Rate - Normal"
        hr_ylabel = "Heart Rate (bpm) [100-160]"

    # Create interactive plots for each vital sign and their delta values
    create_interactive_plot(time_values, heart_rates, "Heart Rate (bpm)", "Heart Rate", f"{patient_id}_Heart_Rate", plot_output_dir, shift_length)
    create_interactive_plot(time_values, respiratory_rates, "Respiratory Rate (breaths/min)", "Respiratory Rate", f"{patient_id}_Respiratory_Rate", plot_output_dir, shift_length)
    create_interactive_plot(time_values, body_temperatures, "Body Temperature (°C)", "Body Temperature", f"{patient_id}_Body_Temperature", plot_output_dir, shift_length)
    create_interactive_plot(valid_time_values, hr_deltas, "HR Delta (% change)", "HR Delta", f"{patient_id}_HR_Delta", plot_output_dir, shift_length)
    create_interactive_plot(valid_time_values, resp_rate_deltas, "Resp Rate Delta (breaths/min)", "Resp Rate Delta", f"{patient_id}_RespRate_Delta", plot_output_dir, shift_length)
    create_interactive_plot(valid_time_values, temp_deltas, "Temp Delta (°C)", "Temp Delta", f"{patient_id}_Temp_Delta", plot_output_dir, shift_length)

def create_interactive_plot(time_values, values, ylabel, title, filename, output_dir, shift_length):
    """
    Create and save an interactive Plotly plot as an HTML file.

    This function generates a scatter plot with lines and markers using the 
    provided time series data and saves it as an HTML file in the specified output directory.

    Args:
        time_values (list): List of datetime objects representing the x-axis.
        values (list): List of values to plot on the y-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        filename (str): Base filename for the output HTML file.
        output_dir (str): Directory where the HTML file will be saved.
        shift_length (int): Simulation duration (in hours) to include in the plot title.

    Returns:
        None
    """
    # Determine tick positions and labels to avoid clutter
    step = max(1, int(len(time_values) / 10))
    tick_values = [time_values[i] for i in range(0, len(time_values), step)]
    tick_text = [dt.strftime("%H:%M") for dt in tick_values]

    # Create a Plotly scatter trace for the data
    trace_data = go.Scatter(
        x=time_values,
        y=values,
        mode='lines+markers',
        name='Data',
        line=dict(color='blue')
    )

    # Define the layout with title, axis labels, legend, and hover mode
    layout = go.Layout(
        title=f"{title} - Shift Length: {shift_length} hrs",
        xaxis=dict(title='Time', tickvals=tick_values, ticktext=tick_text),
        yaxis=dict(title=ylabel),
        legend=dict(x=0.1, y=0.9),
        hovermode='closest'
    )

    # Create the figure and write to an HTML file
    fig = go.Figure(data=[trace_data], layout=layout)
    output_file = os.path.join(output_dir, f"{filename}.html")
    fig.write_html(output_file)
    print(f"{title} interactive plot saved as {output_file}")
