#!/usr/bin/env python
"""
Static Vital Signs Plotting Module

This module reads simulated patient data from a JSON file and creates static 
plots for various vital signs (heart rate, respiratory rate, body temperature) 
and their delta values (HR_Delta, RespRate_Delta, Temp_Delta). It also overlays 
intervention markers on the plots.
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# Main Plotting Functions
# =============================================================================
def plot_vital_signs(patient_id, shift_length):
    """
    Generate static plots for a patient's vital signs and intervention events.

    This function reads the simulation JSON file for the given patient, extracts
    timestamps, vital sign values (heart rate, respiratory rate, body temperature) 
    and their delta values, as well as intervention event data. It then creates a 
    series of static PNG plots with interventions overlaid.

    Args:
        patient_id (str): Unique identifier for the patient.
        shift_length (int): Duration of the simulation (in hours).

    Returns:
        None
    """
    # Define input and output directories using absolute paths
    json_input_dir = os.path.join("/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/Output_Patient_Data", patient_id, "json")
    json_file = os.path.join(json_input_dir, f"{patient_id}_Data.json")
    plot_output_dir = os.path.join("/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/Output_Patient_Data", patient_id, "static_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Load simulation data from JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert timestamps from strings to datetime objects
    timestamps = [entry["timestamp"] for entry in data["data"]]
    time_values = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Extract vital sign data from JSON entries
    heart_rates = [int(entry["vital_signs"]["heart_rate"].split()[0]) for entry in data["data"]]
    respiratory_rates = [int(entry["vital_signs"]["respiratory_rate"].split()[0]) for entry in data["data"]]
    body_temperatures = [float(entry["vital_signs"]["body_temperature"].split()[0]) for entry in data["data"]]
    hrv_values = [int(entry["vital_signs"]["HR_Delta"].split()[0]) if entry["vital_signs"]["HR_Delta"] else None for entry in data["data"]]
    resp_rate_delta_values = [int(entry["vital_signs"]["RespRate_Delta"].split()[0]) if entry["vital_signs"]["RespRate_Delta"] else None for entry in data["data"]]
    temp_delta_values = [float(entry["vital_signs"]["Temp_Delta"].split()[0]) if entry["vital_signs"]["Temp_Delta"] else None for entry in data["data"]]

    # Extract intervention events: record the timestamp and assign a color based on intervention type
    intervention_times = []
    intervention_colors = []
    for entry in data["data"]:
        timestamp = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        if "intervention" in entry and entry["intervention"]:
            for intervention in entry["intervention"]:
                intervention_times.append(timestamp)
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

    # Generate plots for each vital sign and its delta values
    plot_data(time_values, heart_rates, "Heart Rate (bpm)", f"{patient_id}_Heart_Rate", plot_output_dir, shift_length, intervention_times, intervention_colors)
    plot_data(time_values, respiratory_rates, "Respiratory Rate (breaths/min)", f"{patient_id}_Respiratory_Rate", plot_output_dir, shift_length, intervention_times, intervention_colors)
    plot_data(time_values, body_temperatures, "Body Temperature (°C)", f"{patient_id}_Body_Temperature", plot_output_dir, shift_length, intervention_times, intervention_colors)
    plot_data(time_values, hrv_values, "HR_Delta (Δ bpm)", f"{patient_id}_HR_Delta", plot_output_dir, shift_length, intervention_times, intervention_colors)
    plot_data(time_values, resp_rate_delta_values, "RespRate_Delta (Δ breaths/min)", f"{patient_id}_RespRate_Delta", plot_output_dir, shift_length, intervention_times, intervention_colors)
    plot_data(time_values, temp_delta_values, "Temp_Delta (Δ°C)", f"{patient_id}_Temp_Delta", plot_output_dir, shift_length, intervention_times, intervention_colors)

def plot_data(time_values, values, ylabel, title, output_dir, shift_length, intervention_times, intervention_colors):
    """
    Create and save a static time series plot with intervention markers.

    This function uses Matplotlib to generate a plot of the given time series data.
    It overlays intervention events as markers and saves the resulting plot as a PNG file.

    Args:
        time_values (list): List of datetime objects for the x-axis.
        values (list): List of y-axis values.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        output_dir (str): Directory where the plot image will be saved.
        shift_length (int): Simulation duration (in hours) to include in the plot title.
        intervention_times (list): List of datetime objects when interventions occurred.
        intervention_colors (list): List of colors for each intervention marker.

    Returns:
        None
    """
    # Format time values as strings for the x-axis labels
    time_values_str = [dt.strftime("%H:%M") for dt in time_values]

    # Create the main plot for the data series
    plt.figure(figsize=(14, 7))
    plt.plot(time_values_str, values, marker='o', linestyle='-', label="Data", color='blue')

    # Identify indices where interventions occur and extract corresponding y-values and colors
    intervention_indices = []
    intervention_y_values = []
    matched_intervention_colors = []
    for idx, t in enumerate(time_values):
        if t in intervention_times:
            intervention_indices.append(idx)
            intervention_y_values.append(values[idx])
            color_idx = intervention_times.index(t)
            matched_intervention_colors.append(intervention_colors[color_idx])

    # Overlay intervention markers if any are found
    if intervention_indices:
        plt.scatter(
            [time_values_str[i] for i in intervention_indices],
            intervention_y_values,
            color=matched_intervention_colors,
            marker='x',
            s=100,
            label="Interventions"
        )

    # Set plot labels, title, and formatting
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{title} - Shift Length: {shift_length} hrs")
    plt.xticks(rotation=45)

    # Determine tick step to prevent label clutter
    step = max(1, len(time_values_str) // 10)
    plt.xticks(ticks=range(0, len(time_values_str), step), labels=time_values_str[::step])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the output directory as a PNG file
    output_file = os.path.join(output_dir, f"{title}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"{title} plot saved as {output_file}")
