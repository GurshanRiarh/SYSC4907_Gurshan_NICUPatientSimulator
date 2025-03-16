#!/usr/bin/env python
"""
Execute Simulation Module

This module runs the Flask web application for the NICU Patient Simulator.
It imports simulation modules for healthy and unhealthy patients, handles
simulation execution based on user input, and provides endpoints for
downloading and displaying simulation data and plots.
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import os
import zipfile
import io

# =============================================================================
# Directory Setup and Path Insertion
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is two levels up from this file (e.g., UH_v21)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Paths for simulation modules
healthy_files_path = os.path.join(PROJECT_ROOT, "Healthy_Files")
unhealthy_files_path = os.path.join(PROJECT_ROOT, "Unhealthy_Files")

# Insert these paths into sys.path to ensure modules can be imported
sys.path.insert(0, healthy_files_path)
sys.path.insert(0, unhealthy_files_path)

# Debug prints to verify paths
print("Healthy files path:", healthy_files_path)
print("Unhealthy files path:", unhealthy_files_path)
print("sys.path:", sys.path)

# =============================================================================
# Module Imports
# =============================================================================
from NICU_Simulator_Parmas import NeonateSimulator as HealthySimulator, PatientProfile as HealthyProfile
from NICU_Simulator_Params_Unhealthy import NeonateSimulator as UnhealthySimulator, PatientProfile as UnhealthyProfile
from Vital_Sign_Static_Plots_Unhealthy import plot_vital_signs
from Vital_Sign_Interactive_Plots_Unhealthy import plot_interactive_vital_signs
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# =============================================================================
# Flask Application Initialization
# =============================================================================
app = Flask(__name__)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Patient_Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Route: Home Page
# =============================================================================
@app.route('/')
def home():
    """
    Render the simulation mode selection page.
    
    Returns:
        Rendered HTML template for simulation_mode.
    """
    return render_template('simulation_mode.html')

# =============================================================================
# Route: Simulation Setup Page
# =============================================================================
@app.route('/setup')
def setup():
    """
    Render the simulation setup (input form) page.
    
    Returns:
        Rendered HTML template for index.
    """
    return render_template('index.html')

# =============================================================================
# Route: Simulation Execution
# =============================================================================
@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Execute the patient simulation based on form input and render results.

    Processes form data, determines simulation mode (neonate or preterm),
    sets up the appropriate patient profile and simulator, runs the simulation,
    exports data and plots, and finally renders the result page.

    Returns:
        Rendered HTML template for result page on success, or JSON error message.
    """
    try:
        data = request.form
        simulation_mode = data.get('simulation_mode', 'preterm')
        patient_id = data['patient_id']
        start_time = data['start_time']
        weight = float(data['weight'])
        gender = data['gender']
        condition = data['condition']  # Selected condition from dropdown

        # Retrieve selected interventions from checkboxes
        selected_interventions = request.form.getlist('interventions')
        print(f"DEBUG: Selected interventions - {selected_interventions}")
        print(f"DEBUG: Simulation mode: {simulation_mode}")
        
        # Determine condition status
        is_bradycardic = condition in ["bradycardia", "both"]
        is_tachycardic = condition in ["tachycardia", "both"]

        selected_conditions = []
        if is_bradycardic:
            selected_conditions.append("Bradycardia")
        if is_tachycardic:
            selected_conditions.append("Tachycardia")
        print(f"DEBUG: Selected conditions - {selected_conditions}")

        # Set up simulation based on the selected mode
        if simulation_mode == "neonate":
            Age_days = int(data['Age'])
            patient_profile = UnhealthyProfile(
                Age_days=Age_days,
                weight=weight,
                conditions=selected_conditions,
                is_bradycardic=is_bradycardic,
                is_tachycardic=is_tachycardic
            )
            simulator = UnhealthySimulator(
                patient_profile=patient_profile,
                patient_id=patient_id,
                gender=gender,
                start_time=start_time,
                selected_interventions=selected_interventions
            )
        else:
            # Preterm simulation expects gestational age in weeks/days (e.g., "34/6")
            if '/' in data['Age']:
                gestational_age_weeks, gestational_age_days = map(int, data['Age'].split('/'))
            else:
                return jsonify({"error": "Invalid format for gestational age. Expected weeks/days (e.g., 34/6)."}), 400

            patient_profile = HealthyProfile(
                gestational_age_weeks=gestational_age_weeks,
                gestational_age_days=gestational_age_days,
                weight=weight,
                conditions=[]
            )
            simulator = HealthySimulator(
                patient_profile=patient_profile,
                patient_id=patient_id,
                gender=gender,
                start_time=start_time,
                selected_interventions=selected_interventions
            )

        # Create output directories for simulation results
        patient_dir = os.path.join(OUTPUT_DIR, patient_id)
        json_dir = os.path.join(patient_dir, "json")
        csv_dir = os.path.join(patient_dir, "csv")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        
        # Run simulation and export results
        simulator.simulate()
        simulator.export_data()
        simulator.export_gan_values()
        simulator.export_interventions()
        simulator.validate_gan_values()

        # Generate static and interactive plots
        plot_vital_signs(patient_id, simulator.shift_length)
        plot_interactive_vital_signs(patient_id, simulator.shift_length)

        return render_template('result.html', 
                               patient_id=patient_id, 
                               is_bradycardic=(simulation_mode=="neonate" and condition in ["bradycardia", "both"]), 
                               is_tachycardic=(simulation_mode=="neonate" and condition in ["tachycardia", "both"]), 
                               conditions=(condition if simulation_mode=="neonate" else "Healthy"))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =============================================================================
# Route: Download Individual Files
# =============================================================================
@app.route('/download/<patient_id>/<filename>')
def download_file(patient_id, filename):
    """
    Download an individual file (JSON or CSV) for a given patient.

    If filename equals "all_data.zip", delegates to create_zip_file().

    Args:
        patient_id (str): Patient identifier.
        filename (str): Name of the file to download.

    Returns:
        Flask response with the requested file or a JSON error message.
    """
    if filename == "all_data.zip":
        return create_zip_file(patient_id)

    patient_dir = os.path.join(OUTPUT_DIR, patient_id)
    file_paths = [
        os.path.join(patient_dir, "json", filename),
        os.path.join(patient_dir, "csv", filename)
    ]

    for file_path in file_paths:
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

    return jsonify({"error": "File not found."}), 404

# =============================================================================
# Route: Download Interactive Plots ZIP
# =============================================================================
@app.route('/download/<patient_id>/interactive_plots.zip')
def download_interactive_plots(patient_id):
    """
    Download a ZIP archive of interactive plots for a given patient.

    Checks if the interactive plots directory exists and contains valid plot files,
    creates a ZIP archive in memory, and returns it as a downloadable file.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        Flask response with the ZIP file, or JSON error if no valid files are found.
    """
    plot_dir = os.path.join(OUTPUT_DIR, patient_id, "interactive_plots")
    print(f"DEBUG: Looking for interactive plots in -> {os.path.abspath(plot_dir)}")

    if not os.path.exists(plot_dir) or not any(f.endswith((".html", ".png")) for f in os.listdir(plot_dir)):
        return jsonify({"error": "No interactive plots found for this patient."}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(plot_dir):
            for file in files:
                if file.endswith(".html") or file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, plot_dir)
                    zipf.write(file_path, arcname)

    if len(memory_file.getvalue()) == 0:
        return jsonify({"error": "Failed to create ZIP file. No valid plot files found."}), 500

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{patient_id}_interactive_plots.zip"
    )

# =============================================================================
# Route: Display Interactive Plot
# =============================================================================
@app.route('/display_plot/<patient_id>/<plot_type>')
def display_plot(patient_id, plot_type):
    """
    Display an interactive plot for a given patient and plot type.

    Determines the plot filename based on the plot_type and returns the file
    if it exists.

    Args:
        patient_id (str): Patient identifier.
        plot_type (str): Type of plot to display (e.g., "Heart_Rate", "Respiratory_Rate", etc.).

    Returns:
        Flask response with the plot file or a 404 message if not found.
    """
    plot_files = {
        "Heart_Rate": f"{patient_id}_Heart_Rate.html",
        "Respiratory_Rate": f"{patient_id}_Respiratory_Rate.html",
        "Body_Temperature": f"{patient_id}_Body_Temperature.html",
        "HR_Delta": f"{patient_id}_HR_Delta.html",
        "RespRate_Delta": f"{patient_id}_RespRate_Delta.html",
        "Temp_Delta": f"{patient_id}_Temp_Delta.html"
    }
    plot_filename = plot_files.get(plot_type)
    if plot_filename:
        plot_file_path = os.path.join(OUTPUT_DIR, patient_id, "interactive_plots", plot_filename)
        if os.path.exists(plot_file_path):
            return send_file(plot_file_path)
    return "Plot not found", 404

# =============================================================================
# Route: Display CSV File
# =============================================================================
@app.route('/display_csv/<patient_id>/<csv_filename>')
def display_csv(patient_id, csv_filename):
    """
    Display the content of a CSV file for a given patient.

    Reads the CSV file and returns its content wrapped in a <pre> tag.

    Args:
        patient_id (str): Patient identifier.
        csv_filename (str): Name of the CSV file.

    Returns:
        HTML content showing the CSV file, or an error message if not found.
    """
    csv_path = os.path.join(OUTPUT_DIR, patient_id, "csv", csv_filename)
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    return "CSV file not found", 404

# =============================================================================
# Route: Display JSON Data
# =============================================================================
@app.route('/display_json/<patient_id>')
def display_json(patient_id):
    """
    Display the JSON simulation data for a given patient.

    Reads the JSON file and returns its content wrapped in a <pre> tag.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        HTML content showing the JSON data, or an error message if not found.
    """
    json_path = os.path.join(OUTPUT_DIR, patient_id, "json", f"{patient_id}_Data.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    return "JSON file not found", 404

# =============================================================================
# Helper Function: Create ZIP File of All Data
# =============================================================================
def create_zip_file(patient_id):
    """
    Create a ZIP archive containing all simulation data for a given patient.

    Archives JSON files, CSV files, and static plots into a single ZIP file,
    and returns it as a downloadable file.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        Flask response with the ZIP file.
    """
    patient_dir = os.path.join(OUTPUT_DIR, patient_id)
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add JSON files
        json_dir = os.path.join(patient_dir, "json")
        if os.path.exists(json_dir):
            for root, _, files in os.walk(json_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("json", file)
                    zipf.write(file_path, arcname)
        
        # Add CSV files
        csv_dir = os.path.join(patient_dir, "csv")
        if os.path.exists(csv_dir):
            for root, _, files in os.walk(csv_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("csv", file)
                    zipf.write(file_path, arcname)

        # Add static plots
        static_plots_dir = os.path.join(patient_dir, "static_plots")
        if os.path.exists(static_plots_dir):
            for root, _, files in os.walk(static_plots_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, patient_dir)
                    zipf.write(file_path, arcname)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{patient_id}_all_data.zip"
    )

# =============================================================================
# Flask Application Runner
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True)
