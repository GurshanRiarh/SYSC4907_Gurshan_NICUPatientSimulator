#!/usr/bin/env python
"""
NICU Simulator Parameters for Unhealthy Patients

This module defines the necessary components for simulating vital signs 
for neonates with potential unhealthy conditions. It loads pretrained GAN 
models for heart rate and respiration rate, provides a helper function to 
generate synthetic data, and defines classes for storing patient profiles and 
running the simulation.
"""

# =============================================================================
# Imports and Warnings
# =============================================================================
import json
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pandas as pd
from scipy.ndimage import gaussian_filter1d  
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# =============================================================================
# Global Constants and Device Setup
# =============================================================================
delta = 'Δ'
celsius = '°C'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine project directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Trained_GAN_Path_Files")

# =============================================================================
# GAN Generator Definition
# =============================================================================
class StandardGenerator(nn.Module):
    """
    Standard GAN generator used for generating synthetic heart rate and 
    respiration rate data.
    
    Architecture:
        Linear -> ReLU -> BatchNorm1d -> Linear -> ReLU -> Linear -> Sigmoid
    """
    def __init__(self, noise_dim):
        """
        Initialize the StandardGenerator.
        
        Args:
            noise_dim (int): Dimensionality of the input noise vector.
        """
        super(StandardGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, noise):
        """
        Perform a forward pass through the generator.
        
        Args:
            noise (torch.Tensor): Input noise tensor.
            
        Returns:
            torch.Tensor: Generated synthetic output.
        """
        return self.model(noise)

# Set noise dimension
noise_dim = 10

# =============================================================================
# Load Pretrained GAN Models
# =============================================================================
# Dynamically load GAN model paths and initialize models for heart rate and respiration.
heart_rate_model_path = os.path.join(MODEL_DIR, "trained_generator_HeartRate_again.pth")
heart_rate_generator = StandardGenerator(noise_dim).to(device)
heart_rate_generator.load_state_dict(torch.load(heart_rate_model_path, map_location=device, weights_only=True))
heart_rate_generator.eval()

respiration_rate_model_path = os.path.join(MODEL_DIR, "trained_generator_ResprationRate_again.pth")
respiration_rate_generator = StandardGenerator(noise_dim).to(device)
respiration_rate_generator.load_state_dict(torch.load(respiration_rate_model_path, map_location=device, weights_only=True))
respiration_rate_generator.eval()

# =============================================================================
# Synthetic Data Generation Function
# =============================================================================
def generate_synthetic_data(generator, steps, normalize=False, min_val=0, max_val=0, sequence_output=False):
    """
    Generate synthetic data using the specified GAN generator.
    
    Args:
        generator (nn.Module): Pretrained GAN generator.
        steps (int): Number of data points (or time steps) to generate.
        normalize (bool): Whether to scale the generated output.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.
        sequence_output (bool): If True, returns a flattened array; otherwise, returns the first column.
    
    Returns:
        np.ndarray: Array of synthetic data.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn((steps, noise_dim)).to(device)
        synthetic_data = generator(noise).cpu().numpy()
        
    if normalize:
        synthetic_data = synthetic_data * (max_val - min_val) + min_val

    if sequence_output:
        return synthetic_data.flatten()[:steps]
    else:
        return synthetic_data[:, 0]

# =============================================================================
# Patient Profile Class
# =============================================================================
class PatientProfile:
    """
    Represents a patient profile for simulation purposes.
    
    Attributes:
        gestational_age_weeks (int): Gestational age in weeks.
        gestational_age_days (int): Additional days of gestational age.
        weight (float): Patient weight in kilograms.
        conditions (list or str): List or string describing patient conditions.
    """
    def __init__(self, gestational_age_weeks, gestational_age_days, weight, conditions):
        """
        Initialize a PatientProfile.
        
        Args:
            gestational_age_weeks (int): Gestational age in weeks.
            gestational_age_days (int): Gestational age in days.
            weight (float): Weight in kilograms.
            conditions (list or str): Patient conditions.
        """
        self.gestational_age_weeks = gestational_age_weeks
        self.gestational_age_days = gestational_age_days
        self.weight = weight
        self.conditions = conditions

# =============================================================================
# Output Directory Setup
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Patient_Data")

# =============================================================================
# Neonate Simulator Class
# =============================================================================
class NeonateSimulator:
    """
    Simulator for generating synthetic vital signs and interventions for neonate patients.
    
    This class uses pretrained GAN models to create synthetic heart rate and respiration rate data,
    applies smoothing filters, calculates changes in vital signs, and simulates interventions over a 
    given shift length.
    """
    def __init__(self, patient_profile, shift_length=8, patient_id=None, interval_minutes=5,
                 gender=None, start_time=None, selected_interventions=None):
        """
        Initialize the NeonateSimulator.
        
        Args:
            patient_profile (PatientProfile): Patient profile data.
            shift_length (int): Simulation duration in hours.
            patient_id (str): Unique patient identifier.
            interval_minutes (int): Minutes between simulation intervals.
            gender (str, optional): Patient gender. Randomly chosen if not provided.
            start_time (str, optional): Start time in "HH:MM:SS" format; defaults to current time.
            selected_interventions (list, optional): List of interventions to simulate.
        """
        print("DEBUG: Loaded NeonateSimulator from:", __file__, "with selected_interventions =", selected_interventions)
        self.patient_profile = patient_profile
        self.shift_length = shift_length
        self.interval_minutes = interval_minutes
        current_date = datetime.now().date()
        self.start_time = datetime.strptime(start_time, "%H:%M:%S").replace(
            year=current_date.year, month=current_date.month, day=current_date.day
        ) if start_time else datetime.now()
        self.data = []
        self.patient_id = patient_id
        self.gender = gender if gender is not None else random.choice(['Male', 'Female'])
        self.gan_generated_hr = []
        self.gan_generated_rr = []
        self.hrv_data = []
        self.current_state = "awake"
        self.main_output_dir = OUTPUT_DIR
        os.makedirs(self.main_output_dir, exist_ok=True)
        self.patient_dir = os.path.join(self.main_output_dir, self.patient_id)
        self.patient_json_dir = os.path.join(self.patient_dir, "json")
        self.patient_csv_dir = os.path.join(self.patient_dir, "csv")
        os.makedirs(self.patient_json_dir, exist_ok=True)
        os.makedirs(self.patient_csv_dir, exist_ok=True)
        self.num_intervals = int((self.shift_length * 60) / self.interval_minutes)

        # Default intervention list if not provided
        self.selected_interventions = selected_interventions if selected_interventions is not None else [
            "feeding", "medication", "diaper change", "position change", "oxygen administration",
            "lighting adjustment", "lab test", "imaging", "family visitation", "pain management"
        ]
        print("Healthy mode selected")
        # Generate synthetic heart rate data and apply smoothing
        self.synthetic_heart_rate_data = generate_synthetic_data(
            heart_rate_generator, self.num_intervals, normalize=True, min_val=110, max_val=150, sequence_output=False
        )
        self.synthetic_heart_rate_data = gaussian_filter1d(self.synthetic_heart_rate_data, sigma=0.3)

        # Generate synthetic respiration rate data and apply smoothing
        self.synthetic_respiration_rate_data = generate_synthetic_data(
            respiration_rate_generator, self.num_intervals, normalize=True, min_val=30, max_val=100, sequence_output=False
        )
        self.synthetic_respiration_rate_data = gaussian_filter1d(self.synthetic_respiration_rate_data, sigma=0.6)

        # Create iterators for the synthetic data streams
        self.synthetic_heart_rate_iter = iter(self.synthetic_heart_rate_data)
        self.synthetic_respiration_rate_iter = iter(self.synthetic_respiration_rate_data)

        # Calculate delta values for heart rate and respiration rate
        self.calculate_HR_Delta()
        self.calculate_RespRate_Delta()

        # Define intervention rates and covariance
        self.intervention_rates = {
            "diaper change": 0.01,
            "position change": 0.005,
            "oxygen administration": 0.01,
            "lighting adjustment": 0.01,
            "lab test": 0.005,
            "imaging": 0.0025,
            "family visitation": 0.04
        }
        self.intervention_types = list(self.intervention_rates.keys())
        self.intervention_covariance = np.array([
            [1.0, 0.2, 0.1, 0.1, 0.0, 0.0, 0.3],
            [0.2, 1.0, 0.2, 0.3, 0.1, 0.0, 0.1],
            [0.1, 0.2, 1.0, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.3, 0.1, 1.0, 0.2, 0.2, 0.4],
            [0.0, 0.1, 0.1, 0.2, 1.0, 0.3, 0.0],
            [0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0],
            [0.3, 0.1, 0.2, 0.4, 0.0, 0.0, 1.0]
        ])

        # Generate schedules for various interventions
        self.feeding_schedule = self.generate_feeding_schedule()
        self.medication_schedule = self.generate_medication_schedule()
        self.diaper_change_schedule = self.generate_diaper_change_schedule()
        self.position_change_schedule = self.generate_position_change_schedule()
        self.last_family_visit_time = None
        self.family_visit_interval = timedelta(hours=3)
        self.current_position = None

    # -------------------------------------------------------------------------
    # Vital Sign Delta Calculation Methods
    # -------------------------------------------------------------------------
    def calculate_HR_Delta(self):
        """
        Calculate the change (delta) in heart rate between consecutive intervals.
        
        Populates the hrv_data list with delta values.
        """
        hr_values = list(self.synthetic_heart_rate_data)
        self.hrv_data = [None]
        significant_change_threshold = 10
        for i in range(1, len(hr_values)):
            prev = hr_values[i - 1]
            current = hr_values[i]
            delta = math.floor(abs(current - prev) * 10) / 10
            self.hrv_data.append(delta)

    def calculate_RespRate_Delta(self):
        """
        Calculate the change (delta) in respiration rate between consecutive intervals.
        
        Populates the resp_rate_delta_data list with delta values.
        """
        rr_values = list(self.synthetic_respiration_rate_data)
        self.resp_rate_delta_data = []
        for i in range(len(rr_values)):
            if i == 0:
                self.resp_rate_delta_data.append(None)
            else:
                self.resp_rate_delta_data.append(abs(rr_values[i] - rr_values[i - 1]))

    # -------------------------------------------------------------------------
    # Intervention Schedule Generation Methods
    # -------------------------------------------------------------------------
    def generate_feeding_schedule(self):
        """
        Generate a feeding schedule based on patient gestational age and conditions.
        
        Returns:
            list or None: List of datetime objects representing feeding times, 
            or None if the gestational age is less than 34 weeks.
        """
        feeding_times = []
        current_time = self.start_time
        if self.patient_profile.gestational_age_weeks < 34:
            return None
        feeding_interval_hours = 3
        if "feeding intolerance" in self.patient_profile.conditions:
            feeding_interval_hours = 4
        while current_time < self.start_time + timedelta(hours=self.shift_length):
            feeding_times.append(current_time)
            current_time += timedelta(hours=feeding_interval_hours)
        return feeding_times

    def generate_medication_schedule(self):
        """
        Generate a medication schedule for patients with 'infection' condition.
        
        Returns:
            list: List of medication events as dictionaries with time, type, and detail.
        """
        medication_schedule = []
        if "infection" in self.patient_profile.conditions:
            medication_interval_hours = 6
            current_time = self.start_time
            while current_time < self.start_time + timedelta(hours=self.shift_length):
                medication_schedule.append({
                    "time": current_time,
                    "type": "medication",
                    "detail": "antibiotic administration"
                })
                current_time += timedelta(hours=medication_interval_hours)
        return medication_schedule

    def generate_diaper_change_schedule(self):
        """
        Generate a schedule for diaper changes.
        
        Returns:
            list: List of datetime objects for scheduled diaper changes.
        """
        diaper_change_times = []
        current_time = self.start_time
        while current_time < self.start_time + timedelta(hours=self.shift_length):
            diaper_change_times.append(current_time)
            current_time += timedelta(hours=3)
        return diaper_change_times

    def generate_position_change_schedule(self):
        """
        Generate a schedule for patient position changes.
        
        Returns:
            list: List of datetime objects for scheduled position changes.
        """
        position_change_times = []
        current_time = self.start_time
        while current_time < self.start_time + timedelta(hours=self.shift_length):
            position_change_times.append(current_time)
            current_time += timedelta(hours=4)
        return position_change_times

    def adjust_intervention_rates(self, current_time):
        """
        Adjust intervention rates based on the current simulation time.
        
        Args:
            current_time (datetime): The current time in the simulation.
        
        Returns:
            dict: A dictionary of adjusted intervention rates.
        """
        adjusted_rates = self.intervention_rates.copy()
        hour = current_time.hour
        visitation_start = 8
        visitation_end = 20
        if hour < visitation_start or hour >= visitation_end:
            adjusted_rates["family visitation"] = 0
            adjusted_rates["lab test"] = 0
            adjusted_rates["imaging"] = 0
        else:
            adjusted_rates["family visitation"] = self.intervention_rates["family visitation"]
        if hour >= 22 or hour < 6:
            adjusted_rates["lighting adjustment"] *= 0.5
            adjusted_rates["position change"] *= 0.8
        return adjusted_rates

    # -------------------------------------------------------------------------
    # Simulation and Vital Signs Generation Methods
    # -------------------------------------------------------------------------
    def simulate(self):
        """
        Run the full simulation over the specified shift length.
        
        Iterates over simulation intervals, generating vital signs and applying 
        scheduled as well as random interventions. Each simulation step is appended 
        to the data list.
        """
        current_time = self.start_time
        end_time = self.start_time + timedelta(hours=self.shift_length)
        hrv_iter = iter(self.hrv_data)
        resp_rate_delta_iter = iter(self.resp_rate_delta_data)
        previous_body_temperature = None

        while current_time < end_time:
            # Determine sleep/awake state based on hour
            self.current_state = "sleeping" if current_time.hour >= 22 or current_time.hour < 6 else "awake"
            body_temperature = round(random.uniform(36.5, 37.4), 1)
            temp_delta = None if previous_body_temperature is None else round(abs(body_temperature - previous_body_temperature), 1)
            previous_body_temperature = body_temperature

            # Generate a simulation entry with timestamp and vital signs
            entry = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vital_signs": self.generate_vital_signs(hrv_iter, resp_rate_delta_iter, body_temperature, temp_delta),
            }

            # Initialize interventions list
            interventions = []

            # Scheduled Feeding
            if self.feeding_schedule and "feeding" in self.selected_interventions and any(
                abs((current_time - feed_time).total_seconds()) < (self.interval_minutes * 60)
                for feed_time in self.feeding_schedule
            ):
                interventions.append({
                    "type": "feeding",
                    "detail": f"{random.randint(10, 30)} ml formula (scheduled)"
                })

            # Scheduled Medication
            if self.medication_schedule and "medication" in self.selected_interventions and self.medication_schedule[0]["time"] <= current_time:
                medication = self.medication_schedule.pop(0)
                interventions.append({
                    "type": medication["type"],
                    "detail": medication["detail"]
                })

            # Scheduled Diaper Change
            if self.diaper_change_schedule and "diaper change" in self.selected_interventions and any(
                abs((current_time - change_time).total_seconds()) < (self.interval_minutes * 60)
                for change_time in self.diaper_change_schedule
            ):
                interventions.append({
                    "type": "diaper change",
                    "detail": "Scheduled diaper change"
                })

            # Scheduled Position Change
            if self.position_change_schedule and "position change" in self.selected_interventions and any(
                abs((current_time - pos_time).total_seconds()) < (self.interval_minutes * 60)
                for pos_time in self.position_change_schedule
            ):
                interventions.append({
                    "type": "position change",
                    "detail": self.get_intervention_detail("position change")
                })

            # Generate random interventions
            random_interventions = self.generate_interventions(current_time, existing_interventions=interventions)
            if random_interventions:
                interventions.extend(random_interventions)

            # Determine pain level and add pain management if necessary
            entry["pain_level"] = self.determine_pain_level(entry["vital_signs"])
            if entry["pain_level"] == "High" and "pain management" in self.selected_interventions:
                interventions.append({
                    "type": "pain management",
                    "detail": random.choice(["provide non-nutritive sucking", "swaddling"])
                })

            # Remove duplicate intervention types
            unique_interventions = []
            intervention_types = set()
            for interv in interventions:
                if interv['type'] not in intervention_types:
                    unique_interventions.append(interv)
                    intervention_types.add(interv['type'])
            if unique_interventions:
                entry["intervention"] = unique_interventions

            self.data.append(entry)
            current_time += timedelta(minutes=self.interval_minutes)

    def generate_vital_signs(self, hrv_iter, resp_rate_delta_iter, body_temperature, temp_delta):
        """
        Generate a dictionary of vital signs for the current simulation step.
        
        Retrieves heart rate and respiration rate values from their iterators,
        computes delta displays, and formats all vital signs as strings.
        
        Args:
            hrv_iter (iterator): Iterator for heart rate delta values.
            resp_rate_delta_iter (iterator): Iterator for respiration rate delta values.
            body_temperature (float): Current body temperature.
            temp_delta (float): Temperature change since the last interval.
        
        Returns:
            dict: Dictionary containing formatted vital sign readings.
        """
        try:
            heart_rate = round(next(self.synthetic_heart_rate_iter))
        except StopIteration:
            self.synthetic_heart_rate_iter = iter(self.synthetic_heart_rate_data)
            heart_rate = round(next(self.synthetic_heart_rate_iter))
        self.gan_generated_hr.append(heart_rate)

        try:
            respiratory_rate = round(next(self.synthetic_respiration_rate_iter))
        except StopIteration:
            self.synthetic_respiration_rate_iter = iter(self.synthetic_respiration_rate_data)
            respiratory_rate = round(next(self.synthetic_respiration_rate_iter))
        self.gan_generated_rr.append(respiratory_rate)

        try:
            hrv = next(hrv_iter)
            hrv_display = f"{round(hrv, 1)} Δ" if hrv is not None else None
        except StopIteration:
            hrv_iter = iter(self.hrv_data)
            hrv = next(hrv_iter)
            hrv_display = f"{round(hrv, 1)} Δ" if hrv is not None else None

        try:
            resp_rate_delta = next(resp_rate_delta_iter)
            resp_rate_delta_display = f"{round(resp_rate_delta)} Δ breaths/min" if resp_rate_delta is not None else None
        except StopIteration:
            resp_rate_delta_iter = iter(self.resp_rate_delta_data)
            resp_rate_delta = next(resp_rate_delta_iter)
            resp_rate_delta_display = f"{round(resp_rate_delta)} Δ breaths/min" if resp_rate_delta is not None else None

        return {
            "heart_rate": f"{heart_rate} bpm",
            "respiratory_rate": f"{respiratory_rate} breaths/min",
            "body_temperature": f"{body_temperature} °C",
            "HR_Delta": hrv_display,
            "RespRate_Delta": resp_rate_delta_display,
            "Temp_Delta": f"{temp_delta} °C" if temp_delta is not None else None,
        }

    def get_intervention_detail(self, intervention_type):
        """
        Get the detail description for an intervention.
        
        For a "position change", selects a new patient position; otherwise, 
        returns a randomly selected detail for other intervention types.
        
        Args:
            intervention_type (str): Type of intervention.
        
        Returns:
            str or None: The intervention detail.
        """
        if intervention_type == "position change":
            possible_positions = ["supine", "prone", "left lateral", "right lateral"]
            if self.current_position in possible_positions:
                possible_positions.remove(self.current_position)
            new_position = random.choice(possible_positions)
            self.current_position = new_position
            return new_position
        else:
            details = {
                "oxygen administration": random.choice(["increase flow", "adjust FiO2"]),
                "lighting adjustment": random.choice(["dim lights", "reduce noise"]),
                "lab test": random.choice(["blood gas analysis", "CBC", "electrolyte panel"]),
                "imaging": random.choice(["chest X-ray", "abdominal ultrasound"]),
                "family visitation": None,
            }
            return details.get(intervention_type, None)

    def generate_interventions(self, current_time, existing_interventions=[]):
        """
        Generate random interventions based on adjusted rates.
        
        Uses a Poisson process to determine if an intervention occurs. 
        Prevents duplicate interventions within the same interval.
        
        Args:
            current_time (datetime): The current simulation time.
            existing_interventions (list): Interventions already scheduled for the interval.
        
        Returns:
            list or None: List of intervention dictionaries, or None if none generated.
        """
        interventions = []
        adjusted_rates = self.adjust_intervention_rates(current_time)
        existing_types = [interv['type'] for interv in existing_interventions]

        for intervention_type, rate in adjusted_rates.items():
            if intervention_type not in self.selected_interventions:
                continue
            if intervention_type == "family visitation":
                if self.last_family_visit_time and (current_time - self.last_family_visit_time) < self.family_visit_interval:
                    continue
                else:
                    if np.random.poisson(rate) > 0:
                        self.last_family_visit_time = current_time
                        interventions.append({
                            "type": "family visitation",
                            "detail": self.get_intervention_detail(intervention_type)
                        })
            elif intervention_type == "position change":
                if intervention_type not in existing_types and random.random() < 0.05:
                    interventions.append({
                        "type": "position change",
                        "detail": self.get_intervention_detail(intervention_type)
                    })
            else:
                if intervention_type not in existing_types and np.random.poisson(rate) > 0:
                    intervention_detail = self.get_intervention_detail(intervention_type)
                    interventions.append({
                        "type": intervention_type,
                        "detail": intervention_detail
                    })
        return interventions if interventions else None

    def determine_pain_level(self, vital_signs):
        """
        Determine the pain level based on vital sign thresholds.
        
        Considers heart rate, respiratory rate, and body temperature to calculate 
        a pain score. Returns "High", "Moderate", or "Low" accordingly.
        
        Args:
            vital_signs (dict): Dictionary of vital sign strings.
        
        Returns:
            str: Pain level ("High", "Moderate", or "Low").
        """
        score = 0
        heart_rate = int(vital_signs["heart_rate"].split(' ')[0])
        respiratory_rate = int(vital_signs["respiratory_rate"].split(' ')[0])
        body_temperature = float(vital_signs["body_temperature"].split(' ')[0])

        if heart_rate > 160 or heart_rate < 120:
            score += 1
        if respiratory_rate > 70 or respiratory_rate < 35:
            score += 1
        if body_temperature < 36.0 or body_temperature > 37.5:
            score += 1

        if score >= 4:
            return "High"
        elif score == 3:
            return "Moderate"
        else:
            return "Low"

    # -------------------------------------------------------------------------
    # Data Export Methods
    # -------------------------------------------------------------------------
    def export_data(self):
        """
        Export the simulation data to a JSON file.
        
        Converts any NumPy float32 values to standard Python floats.
        The JSON file is saved in the patient's JSON directory.
        """
        self.output_file = os.path.join(self.patient_json_dir, f"{self.patient_id}_Data.json")
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj

        Pre_Conepetual_Age = f"{self.patient_profile.gestational_age_weeks}/{self.patient_profile.gestational_age_days}"
        output_data = {
            "patient_id": self.patient_id,
            "Sex": self.gender,
            "Post-Conceptual Age": f"{Pre_Conepetual_Age} Weeks/Days",
            "Weight": f"{self.patient_profile.weight} kg",
            "Conditions": self.patient_profile.conditions,
            "data": convert_floats(self.data)
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Exported JSON: {self.output_file}")

    def export_gan_values(self):
        """
        Export the GAN-generated values to CSV files.
        
        Exports heart rate, respiration rate, heart rate delta, respiration rate delta, 
        and temperature delta to separate CSV files in the patient's CSV directory.
        """
        hr_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_Heart_Rate_Normal.csv")
        rr_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_Respiration_Rate.csv")
        hrv_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_HR_Delta.csv")
        resp_rate_delta_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_RespRate_Delta.csv")
        temp_delta_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_Temp_Delta.csv")

        hr_df = pd.DataFrame(self.gan_generated_hr, columns=["GAN_Heart_Rate"])
        rr_df = pd.DataFrame(self.gan_generated_rr, columns=["GAN_Respiration_Rate"])
        hrv_df = pd.DataFrame(self.hrv_data[:len(self.gan_generated_hr)], columns=["GAN_HR_Delta"])
        resp_rate_delta_df = pd.DataFrame(self.resp_rate_delta_data[:len(self.gan_generated_rr)], columns=["GAN_RespRate_Delta"])
        temp_delta_df = pd.DataFrame(
            [entry["vital_signs"]["Temp_Delta"] for entry in self.data if entry["vital_signs"]["Temp_Delta"] is not None],
            columns=["Temp_Delta"]
        )

        hr_df.to_csv(hr_output_file, index=False)
        rr_df.to_csv(rr_output_file, index=False)
        hrv_df.to_csv(hrv_output_file, index=False)
        resp_rate_delta_df.to_csv(resp_rate_delta_output_file, index=False)
        temp_delta_df.to_csv(temp_delta_output_file, index=False)

        print(f"CSV exports completed: {self.patient_csv_dir}")

    def export_interventions(self):
        """
        Export recorded interventions during the simulation to a CSV file.
        
        Extracts intervention details from each simulation entry and saves them
        to a CSV file in the patient's CSV directory.
        """
        interventions_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_Interventions.csv")
        interventions_list = []
        for entry in self.data:
            timestamp = entry["timestamp"]
            interventions = entry.get("intervention")
            if interventions:
                for intervention in interventions:
                    intervention_type = intervention.get("type")
                    intervention_detail = intervention.get("detail")
                    interventions_list.append({
                        "timestamp": timestamp,
                        "intervention_type": intervention_type,
                        "intervention_detail": intervention_detail
                    })
        interventions_df = pd.DataFrame(interventions_list)
        interventions_df.to_csv(interventions_output_file, index=False)
        print(f"Interventions CSV export completed: {self.patient_csv_dir}")

    def validate_gan_values(self):
        """
        Validate that the GAN-generated values match the JSON simulation data.
        
        Checks if the heart rate and respiration rate values generated by the GAN
        match the corresponding values in the exported JSON data.
        """
        json_heart_rates = [int(entry["vital_signs"]["heart_rate"].split(' ')[0])
                            for entry in self.data if "heart_rate" in entry["vital_signs"]]
        json_respiratory_rates = [int(entry["vital_signs"]["respiratory_rate"].split(' ')[0])
                                  for entry in self.data if "respiratory_rate" in entry["vital_signs"]]
        hr_validation = np.array_equal(json_heart_rates, self.gan_generated_hr)
        rr_validation = np.array_equal(json_respiratory_rates, self.gan_generated_rr)

        if hr_validation and rr_validation:
            print("Validation successful: All GAN-generated values match the JSON data.")
        else:
            if not hr_validation:
                print("Validation failed: Mismatch found between GAN-generated heart rate values and JSON data.")
            if not rr_validation:
                print("Validation failed: Mismatch found between GAN-generated respiration rate values and JSON data.")
