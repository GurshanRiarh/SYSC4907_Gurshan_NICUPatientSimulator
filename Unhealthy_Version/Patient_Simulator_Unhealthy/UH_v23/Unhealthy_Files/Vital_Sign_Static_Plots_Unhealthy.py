#!/usr/bin/env python
"""
Static Vital Signs Plotting Module

This module reads simulated patient data from a JSON file and creates static plots 
for various vital signs (heart rate, respiratory rate, body temperature) and their 
delta (change) values using Matplotlib. The plots are saved as PNG files.
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import matplotlib
matplotlib.use('Agg')  # Prevents Matplotlib GUI errors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# Global Directory Setup
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Patient_Data")

# =============================================================================
# Main Plotting Functions
# =============================================================================
def plot_vital_signs(patient_id, shift_length):
    """
    Generate static plots for a patient's vital signs.

    This function reads a JSON file containing the simulated patient data and 
    creates static PNG plots for heart rate, respiratory rate, body temperature, 
    and their delta values. The plots are saved in the patient's 'static_plots' 
    folder within the project output directory.

    Args:
        patient_id (str): Unique identifier for the patient.
        shift_length (int): Duration of the simulation in hours.

    Returns:
        None
    """
    json_input_dir = os.path.join(OUTPUT_DIR, patient_id, "json")
    json_file = os.path.join(json_input_dir, f"{patient_id}_Data.json")
    plot_output_dir = os.path.join(OUTPUT_DIR, patient_id, "static_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    if not os.path.exists(json_file):
        print(f"Error: JSON file not found at {json_file}")
        return

    # Load patient simulation data from JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert timestamps to datetime objects
    timestamps = [entry["timestamp"] for entry in data["data"]]
    time_values = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    # Extract vital sign values
    heart_rates = [int(entry["vital_signs"]["heart_rate"].split()[0]) for entry in data["data"]]
    respiratory_rates = [int(entry["vital_signs"]["respiratory_rate"].split()[0]) for entry in data["data"]]
    body_temperatures = [float(entry["vital_signs"]["body_temperature"].split()[0]) for entry in data["data"]]

    # Extract delta values for vital signs, if available
    hr_deltas, resp_rate_deltas, temp_deltas = [], [], []
    valid_time_values = []
    for i, entry in enumerate(data["data"]):
        if "HR_Delta" in entry["vital_signs"] and entry["vital_signs"]["HR_Delta"] is not None:
            hr_deltas.append(float(entry["vital_signs"]["HR_Delta"].split()[0].replace('%','')))
            valid_time_values.append(time_values[i])
        if "RespRate_Delta" in entry["vital_signs"] and entry["vital_signs"]["RespRate_Delta"] is not None:
            resp_rate_deltas.append(float(entry["vital_signs"]["RespRate_Delta"].split()[0]))
        if "Temp_Delta" in entry["vital_signs"] and entry["vital_signs"]["Temp_Delta"] is not None:
            temp_deltas.append(float(entry["vital_signs"]["Temp_Delta"].split()[0]))

    # Determine plot parameters based on patient conditions
    conditions = data.get("Conditions", [])
    if isinstance(conditions, str):
        conditions = [conditions]
    elif not isinstance(conditions, list):
        conditions = []

    if "Bradycardia" in conditions and "Tachycardia" in conditions:
        hr_title = "Heart Rate (bpm) - Bradycardia & Tachycardia Swings"
        hr_ylabel = "Heart Rate (bpm)"
        hr_ylim = (70, 230)
    elif "Bradycardia" in conditions:
        hr_title = "Heart Rate (bpm) - Bradycardia"
        hr_ylabel = "Heart Rate (bpm)"
        hr_ylim = (70, 140)
    elif "Tachycardia" in conditions:
        hr_title = "Heart Rate (bpm) - Tachycardia"
        hr_ylabel = "Heart Rate (bpm)"
        hr_ylim = (160, 230)
    else:
        hr_title = "Heart Rate (bpm) - Normal"
        hr_ylabel = "Heart Rate (bpm)"
        hr_ylim = (100, 160)

    print(f"DEBUG: Conditions Extracted - {conditions}")

    # Create and save plots for each vital sign and their delta values
    plot_data(time_values, heart_rates, hr_ylabel, hr_title, f"{patient_id}_Heart_Rate", plot_output_dir, shift_length, ylim=hr_ylim)
    plot_data(time_values, respiratory_rates, "Respiratory Rate (breaths/min)", "Respiratory Rate", f"{patient_id}_Respiratory_Rate", plot_output_dir, shift_length)
    plot_data(time_values, body_temperatures, "Body Temperature (°C)", "Body Temperature", f"{patient_id}_Body_Temperature", plot_output_dir, shift_length)
    plot_data(valid_time_values, hr_deltas, "Heart Rate Delta (% change)", "HR Delta", f"{patient_id}_HR_Delta", plot_output_dir, shift_length)
    plot_data(valid_time_values, resp_rate_deltas, "Respiratory Rate Delta (breaths/min)", "Resp Rate Delta", f"{patient_id}_RespRate_Delta", plot_output_dir, shift_length)
    plot_data(valid_time_values, temp_deltas, "Body Temperature Delta (°C)", "Temp Delta", f"{patient_id}_Temp_Delta", plot_output_dir, shift_length)

def plot_data(time_values, values, ylabel, title, filename, output_dir, shift_length, ylim=None):
    """
    Create and save a static time series plot as a PNG file.

    This function uses Matplotlib to plot the provided time series data, formats
    the x-axis with time labels, and saves the resulting plot image.

    Args:
        time_values (list): List of datetime objects representing the x-axis.
        values (list): List of values for the y-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        filename (str): Base name for the output file.
        output_dir (str): Directory where the PNG file will be saved.
        shift_length (int): Simulation duration (in hours) to include in the plot title.
        ylim (tuple, optional): y-axis limits (min, max). Defaults to None.

    Returns:
        None
    """
    # Format time values as strings for plotting
    time_values_str = [dt.strftime("%H:%M") for dt in time_values]
    plt.figure(figsize=(14, 7))
    plt.plot(time_values_str, values, marker='o', linestyle='-', label="Data", color='blue')
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{title} - Shift Length: {shift_length} hrs")
    plt.xticks(rotation=45)
    if ylim:
        plt.ylim(ylim)
    step = max(1, len(time_values_str) // 10)
    plt.xticks(ticks=range(0, len(time_values_str), step), labels=time_values_str[::step])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"{title} plot saved as {output_file}")
