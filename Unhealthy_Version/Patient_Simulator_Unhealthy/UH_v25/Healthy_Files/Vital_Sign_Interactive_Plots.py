#!/usr/bin/env python
"""
Interactive Vital Signs Plotting Module

This module reads simulated patient data from JSON files and creates interactive 
Plotly plots for various vital signs (heart rate, respiratory rate, body temperature) 
and their delta values. It also overlays intervention events on the plots.
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, time

# =============================================================================
# Main Plotting Functions
# =============================================================================
def plot_interactive_vital_signs(patient_id, shift_length):
    """
    Generate interactive Plotly plots for a patient's vital signs and interventions.

    This function reads the simulation JSON data for a given patient, extracts 
    the timestamps, vital sign measurements (heart rate, respiratory rate, body 
    temperature), and their delta values, as well as intervention information.
    It then calls the helper function to create interactive HTML plots that 
    overlay intervention markers on the main data.

    Args:
        patient_id (str): Unique identifier for the patient.
        shift_length (int): Duration of the simulation in hours.

    Returns:
        None
    """
    # Set up input and output directories (hardcoded path)
    json_input_dir = os.path.join("/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/Output_Patient_Data", patient_id, "json")
    json_file = os.path.join(json_input_dir, f"{patient_id}_Data.json")
    plot_output_dir = os.path.join("/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/Output_Patient_Data", patient_id, "interactive_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Load simulation data from JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Parse timestamps and convert them to datetime objects
    timestamps = [entry["timestamp"] for entry in data["data"]]
    time_values = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Extract vital sign data from simulation entries
    heart_rates = [int(entry["vital_signs"]["heart_rate"].split()[0]) for entry in data["data"]]
    respiratory_rates = [int(entry["vital_signs"]["respiratory_rate"].split()[0]) for entry in data["data"]]
    body_temperatures = [float(entry["vital_signs"]["body_temperature"].split()[0]) for entry in data["data"]]
    hrv_values = [int(entry["vital_signs"]["HR_Delta"].split()[0]) if entry["vital_signs"]["HR_Delta"] else None for entry in data["data"]]
    resp_rate_delta_values = [int(entry["vital_signs"]["RespRate_Delta"].split()[0]) if entry["vital_signs"]["RespRate_Delta"] else None for entry in data["data"]]
    temp_delta_values = [float(entry["vital_signs"]["Temp_Delta"].split()[0]) if entry["vital_signs"]["Temp_Delta"] else None for entry in data["data"]]

    # Extract intervention events: times, descriptions, and assign colors
    intervention_times = []
    intervention_descriptions = []
    intervention_colors = []
    for entry in data["data"]:
        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        if "intervention" in entry and entry["intervention"]:
            for intervention in entry["intervention"]:
                intervention_times.append(timestamp)
                detail = intervention["detail"] if intervention["detail"] else ""
                intervention_descriptions.append(f"{intervention['type']} - {detail}")
                if intervention["type"] == "pain management":
                    intervention_colors.append('red')
                elif intervention["type"] == "lighting adjustment":
                    intervention_colors.append('blue')
                elif intervention["type"] == "position change":
                    intervention_colors.append('green')
                elif intervention["type"] == "diaper change":
                    intervention_colors.append('brown')
                elif intervention["type"] == "feeding":
                    intervention_colors.append('orange')
                elif intervention["type"] == "oxygen administration":
                    intervention_colors.append('purple')
                elif intervention["type"] == "clinical assessment":
                    intervention_colors.append('pink')
                else:
                    intervention_colors.append('grey')

    # Create interactive plots for each vital sign and their delta values
    create_interactive_plot(time_values, heart_rates, "Heart Rate (bpm)", f"{patient_id}_Heart_Rate",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)
    create_interactive_plot(time_values, respiratory_rates, "Respiratory Rate (breaths/min)", f"{patient_id}_Respiratory_Rate",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)
    create_interactive_plot(time_values, body_temperatures, "Body Temperature (°C)", f"{patient_id}_Body_Temperature",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)
    create_interactive_plot(time_values, hrv_values, "HR_Delta (Δ bpm)", f"{patient_id}_HR_Delta",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)
    create_interactive_plot(time_values, resp_rate_delta_values, "RespRate_Delta (Δ breaths/min)", f"{patient_id}_RespRate_Delta",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)
    create_interactive_plot(time_values, temp_delta_values, "Temp_Delta (Δ°C)", f"{patient_id}_Temp_Delta",
                              plot_output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors)

def create_interactive_plot(time_values, values, ylabel, title, output_dir, shift_length, intervention_times, intervention_descriptions, intervention_colors):
    """
    Create an interactive Plotly plot for a given time series and overlay intervention markers.

    This function generates a scatter plot with lines and markers for the primary 
    data, then overlays intervention events as separate markers with distinct colors. 
    The resulting figure is saved as an HTML file.

    Args:
        time_values (list): List of datetime objects for the x-axis.
        values (list): List of y-axis values.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        output_dir (str): Directory to save the HTML file.
        shift_length (int): Simulation duration (in hours) included in the plot title.
        intervention_times (list): List of datetime objects when interventions occurred.
        intervention_descriptions (list): List of descriptions for the interventions.
        intervention_colors (list): List of colors for each intervention marker.

    Returns:
        None
    """
    # Determine tick positions and labels to prevent clutter
    step = max(1, int(len(time_values) / 10))
    tick_values = [time_values[i] for i in range(0, len(time_values), step)]
    tick_text = [dt.strftime("%H:%M") for dt in tick_values]

    # Create the primary trace for the main data series
    trace_data = go.Scatter(
        x=time_values,
        y=values,
        mode='lines+markers',
        name='Data',
        line=dict(color='blue')
    )

    # Compute y-values for intervention markers by matching intervention times to data points
    intervention_y_values = []
    for t in intervention_times:
        if t in time_values:
            idx = time_values.index(t)
            intervention_y_values.append(values[idx])
        else:
            time_diffs = [abs((t - tv).total_seconds()) for tv in time_values]
            idx = time_diffs.index(min(time_diffs))
            intervention_y_values.append(values[idx])

    # Create a separate trace for interventions
    trace_interventions = go.Scatter(
        x=intervention_times,
        y=intervention_y_values,
        mode='markers',
        marker=dict(color=intervention_colors, size=10, symbol='x'),
        name="Interventions",
        text=intervention_descriptions,
        hovertemplate='Time: %{x}<br>Value: %{y}<br>Intervention: %{text}'
    )

    # Define the layout for the plot
    layout = go.Layout(
        title=f"{title} - Shift Length: {shift_length} hrs",
        xaxis=dict(
            title='Time',
            tickvals=tick_values,
            ticktext=tick_text
        ),
        yaxis=dict(title=ylabel),
        legend=dict(x=0.1, y=0.9),
        hovermode='closest'
    )

    # Create the figure and save as an HTML file
    fig = go.Figure(data=[trace_data, trace_interventions], layout=layout)
    output_file = os.path.join(output_dir, f"{title}.html")
    fig.write_html(output_file)
    print(f"{title} interactive plot saved as {output_file}")
