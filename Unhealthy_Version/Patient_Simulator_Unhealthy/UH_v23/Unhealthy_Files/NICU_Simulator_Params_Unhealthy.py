#!/usr/bin/env python
"""
NICU Simulator Parameters and Synthetic Data Generation Module

This module sets up the environment, loads pretrained GAN generator models,
and defines functions and classes for generating synthetic vital signs data 
and simulating neonate patient profiles.
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

# Determine project and model directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "Trained_GAN_Path_Files")

# =============================================================================
# GAN Generator Definitions
# =============================================================================
class StandardGenerator(nn.Module):
    """
    Standard GAN generator for Heart Rate and Respiration Rate.
    Architecture: Linear -> ReLU -> BatchNorm -> Linear -> ReLU -> Linear -> Sigmoid.
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
        Forward pass through the generator.

        Args:
            noise (torch.Tensor): Input noise tensor.

        Returns:
            torch.Tensor: Generated synthetic output.
        """
        return self.model(noise)


class BradyTachyGenerator(nn.Module):
    """
    GAN generator for Bradycardia and Tachycardia simulation.
    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid.
    The final layer outputs 96 values (for sequence generation).
    """
    def __init__(self, noise_dim):
        """
        Initialize the BradyTachyGenerator.

        Args:
            noise_dim (int): Dimensionality of the input noise vector.
        """
        super(BradyTachyGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.Sigmoid()
        )

    def forward(self, noise):
        """
        Forward pass through the generator.

        Args:
            noise (torch.Tensor): Input noise tensor.

        Returns:
            torch.Tensor: Generated synthetic sequence data.
        """
        return self.model(noise)


# =============================================================================
# Model Loading and Initialization
# =============================================================================
noise_dim = 10

# Construct model file paths
heart_rate_model_path = os.path.join(MODEL_DIR, "trained_generator_HeartRate_again.pth")
bradycardia_model_path = os.path.join(MODEL_DIR, "trained_generator_Bradycardia.pth")
tachycardia_model_path = os.path.join(MODEL_DIR, "trained_generator_Tachycardia.pth")
respiration_rate_model_path = os.path.join(MODEL_DIR, "trained_generator_ResprationRate_again.pth")

# Load and set models to evaluation mode
heart_rate_generator = StandardGenerator(noise_dim).to(device)
heart_rate_generator.load_state_dict(torch.load(heart_rate_model_path, map_location=device, weights_only=True))
heart_rate_generator.eval()

bradycardia_generator = BradyTachyGenerator(noise_dim).to(device)
bradycardia_generator.load_state_dict(torch.load(bradycardia_model_path, map_location=device, weights_only=True))
bradycardia_generator.eval()

tachycardia_generator = BradyTachyGenerator(noise_dim).to(device)
tachycardia_generator.load_state_dict(torch.load(tachycardia_model_path, map_location=device, weights_only=True))
tachycardia_generator.eval()

respiration_rate_generator = StandardGenerator(noise_dim).to(device)
respiration_rate_generator.load_state_dict(torch.load(respiration_rate_model_path, map_location=device, weights_only=True))
respiration_rate_generator.eval()

# =============================================================================
# Synthetic Data Generation Function
# =============================================================================
def generate_synthetic_data(generator, steps, normalize=False, min_val=0, max_val=0, sequence_output=False):
    """
    Generate synthetic data using a given GAN generator.

    Args:
        generator (nn.Module): Pretrained GAN generator model.
        steps (int): Number of synthetic data points (or steps) to generate.
        normalize (bool): If True, scales the output to a given range.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.
        sequence_output (bool): If True, returns a flattened sequence; otherwise, returns the first column.

    Returns:
        np.ndarray: Generated synthetic data array.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn((steps, noise_dim)).to(device)
        synthetic_data = generator(noise).cpu().numpy()  # For Brady/Tachy GAN, shape may be (steps, 96)

    if normalize:
        synthetic_data = synthetic_data * (max_val - min_val) + min_val

    if sequence_output:
        return synthetic_data.flatten()[:steps]  # Flatten output for sequence mode
    else:
        return synthetic_data[:, 0]  # Use first column (for HR & Respiration generators)

# =============================================================================
# Output Directory Setup
# =============================================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_Patient_Data")

# =============================================================================
# Patient Profile Class
# =============================================================================
class PatientProfile:
    """
    Data structure for storing patient profile information.
    """
    def __init__(self, Age_days, weight, conditions, is_bradycardic=False, is_tachycardic=False):
        """
        Initialize the patient profile.

        Args:
            Age_days (int or str): Age of the patient in days (or a string for post-conceptual age).
            weight (float): Patient's weight in kilograms.
            conditions (list or str): Medical conditions of the patient.
            is_bradycardic (bool): Whether the patient exhibits bradycardia.
            is_tachycardic (bool): Whether the patient exhibits tachycardia.
        """
        print(f"DEBUG: Initializing PatientProfile with Age_days={Age_days}, weight={weight}, conditions={conditions}")
        self.Age_days = Age_days
        self.weight = weight
        self.conditions = conditions
        self.is_bradycardic = is_bradycardic
        self.is_tachycardic = is_tachycardic

# =============================================================================
# Neonate Simulator Class
# =============================================================================
class NeonateSimulator:
    """
    Simulator class for generating synthetic vital signs and interventions for neonate patients.
    """
    def __init__(self, patient_profile, shift_length=8, patient_id=None, interval_minutes=5,
                 gender=None, start_time=None, selected_interventions=None):
        """
        Initialize the simulator with patient data and simulation parameters.
        
        Sets up initial synthetic data, schedules, and output directories.

        Args:
            patient_profile (PatientProfile): Patient profile data.
            shift_length (int): Simulation duration in hours.
            patient_id (str): Unique identifier for the patient.
            interval_minutes (int): Time interval (in minutes) between simulation steps.
            gender (str): Patient gender; if None, randomly chosen.
            start_time (str): Simulation start time in "HH:MM:SS" format; defaults to current time.
            selected_interventions (list): List of interventions to simulate.
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

        # Store selected interventions; default to all if none provided.
        self.selected_interventions = selected_interventions if selected_interventions is not None else [
            "feeding", "medication", "diaper change", "position change", "oxygen administration",
            "lighting adjustment", "lab test", "imaging", "family visitation", "pain management"
        ]

        # Generate synthetic heart rate data based on patient condition
        if self.patient_profile.is_bradycardic and self.patient_profile.is_tachycardic:
            print("Bradycardia & Tachycardia mode selected - Using Normal HR GAN with Spikes")
            # Generate base heart rate data in the range 110-150 BPM
            base_hr = generate_synthetic_data(
                heart_rate_generator, 
                self.num_intervals, 
                normalize=True, 
                min_val=110,
                max_val=150,
                sequence_output=True
            )
            i = 0
            while i < self.num_intervals:
                # With a 15% chance at each interval, initiate a spike event
                if np.random.rand() < 0.15:
                    # Choose spike duration of 3 or 4 intervals (ensuring we don't exceed bounds)
                    spike_duration = np.random.choice([3, 4])
                    spike_duration = min(spike_duration, self.num_intervals - i)
                    # Choose spike type: 50% chance for bradycardia vs. tachycardia
                    if np.random.rand() < 0.5:
                        # Bradycardia spike: reduce BPM to between 100 and 110
                        spike_data = generate_synthetic_data(
                            bradycardia_generator, 
                            spike_duration, 
                            normalize=True, 
                            min_val=100,
                            max_val=110,
                            sequence_output=True
                        )
                    else:
                        # Tachycardia spike: increase BPM to between 150 and 190
                        spike_data = generate_synthetic_data(
                            tachycardia_generator, 
                            spike_duration, 
                            normalize=True, 
                            min_val=150,
                            max_val=190,
                            sequence_output=True
                        )
                    # Replace the base data with spike values for the duration of the event
                    base_hr[i:i+spike_duration] = spike_data
                    i += spike_duration
                else:
                    i += 1
            self.synthetic_heart_rate_data = base_hr

        elif self.patient_profile.is_bradycardic:
            print("Neonate Bradycardia mode selected")
            self.synthetic_heart_rate_data = generate_synthetic_data(
                bradycardia_generator, 
                self.num_intervals, 
                normalize=True, 
                min_val=50,
                max_val=90,
                sequence_output=True
            )
            self.synthetic_heart_rate_data = gaussian_filter1d(self.synthetic_heart_rate_data, sigma=0.3)

        elif self.patient_profile.is_tachycardic:
            print("Tachycardia mode selected")
            self.synthetic_heart_rate_data = generate_synthetic_data(
                tachycardia_generator, 
                self.num_intervals, 
                normalize=True, 
                min_val=140, 
                max_val=210, 
                sequence_output=True
            )
            self.synthetic_heart_rate_data = gaussian_filter1d(self.synthetic_heart_rate_data, sigma=0.1)
        else:
            print("Healthy mode selected")
            self.synthetic_heart_rate_data = generate_synthetic_data(
                heart_rate_generator, 
                self.num_intervals, 
                normalize=True, 
                min_val=110, 
                max_val=150, 
                sequence_output=False
            )
            self.synthetic_heart_rate_data = gaussian_filter1d(self.synthetic_heart_rate_data, sigma=0.5)
        
        # Generate synthetic respiration rate data
        self.synthetic_respiration_rate_data = generate_synthetic_data(
            respiration_rate_generator, 
            self.num_intervals, 
            normalize=True, 
            min_val=30, 
            max_val=100, 
            sequence_output=False
        )
        self.synthetic_respiration_rate_data = gaussian_filter1d(self.synthetic_respiration_rate_data, sigma=0.5)

        # Create iterators for synthetic data streams
        self.synthetic_heart_rate_iter = iter(self.synthetic_heart_rate_data)
        self.synthetic_respiration_rate_iter = iter(self.synthetic_respiration_rate_data)

        # Calculate delta values for HR and Respiration
        self.calculate_HR_Delta()
        self.calculate_RespRate_Delta()

        # Intervention scheduling parameters
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

        # Generate schedules for interventions
        self.feeding_schedule = self.generate_feeding_schedule()
        self.medication_schedule = self.generate_medication_schedule()
        self.diaper_change_schedule = self.generate_diaper_change_schedule()
        self.position_change_schedule = self.generate_position_change_schedule()
        self.last_family_visit_time = None
        self.family_visit_interval = timedelta(hours=3)
        self.current_position = None

    # -------------------------------------------------------------------------
    # Vital Signs Delta Calculations
    # -------------------------------------------------------------------------
    def calculate_HR_Delta(self):
        """
        Calculate the delta (change) in heart rate between consecutive intervals.
        Applies scaling based on patient conditions.
        """
        hr_values = list(self.synthetic_heart_rate_data)
        self.hrv_data = [None]
        if self.patient_profile.is_bradycardic and self.patient_profile.is_tachycardic:
            significant_change_threshold = 30
        elif self.patient_profile.is_bradycardic or self.patient_profile.is_tachycardic:
            significant_change_threshold = 20
        else:
            significant_change_threshold = 10
        
        for i in range(1, len(hr_values)):
            prev = hr_values[i - 1]
            current = hr_values[i]
            delta_val = math.floor(abs(current - prev) * 10) / 10

            if self.patient_profile.is_bradycardic and self.patient_profile.is_tachycardic:
                delta_val = delta_val * 1.5 if delta_val > significant_change_threshold else delta_val
            elif self.patient_profile.is_bradycardic:
                delta_val = delta_val * 1.2 if current < prev and delta_val > significant_change_threshold else delta_val
            elif self.patient_profile.is_tachycardic:
                delta_val = delta_val * 1.2 if current > prev and delta_val > significant_change_threshold else delta_val
            self.hrv_data.append(delta_val)

    def calculate_RespRate_Delta(self):
        """
        Calculate the delta (change) in respiration rate between consecutive intervals.
        """
        rr_values = list(self.synthetic_respiration_rate_data)
        self.resp_rate_delta_data = []
        for i in range(len(rr_values)):
            if i == 0:
                self.resp_rate_delta_data.append(None)
            else:
                self.resp_rate_delta_data.append(abs(rr_values[i] - rr_values[i - 1]))

    # -------------------------------------------------------------------------
    # Intervention Schedule Generation
    # -------------------------------------------------------------------------
    def generate_feeding_schedule(self):
        """
        Generate a feeding schedule based on the patient's age and condition.
        Returns a list of datetime objects when feeding is scheduled.
        """
        feeding_times = []
        current_time = self.start_time
        if self.patient_profile.Age_days is None:
            print("Warning: Age_days is None, defaulting to 28 to allow feeding schedule generation.")
            return None
        if self.patient_profile.Age_days < 28:
            return None
        else:
            feeding_interval_hours = 3
            if "feeding intolerance" in self.patient_profile.conditions:
                feeding_interval_hours = 4
            while current_time < self.start_time + timedelta(hours=self.shift_length):
                feeding_times.append(current_time)
                current_time += timedelta(hours=feeding_interval_hours)
        return feeding_times

    def generate_medication_schedule(self):
        """
        Generate a medication schedule if the patient has an 'infection' condition.
        
        Returns:
            list: A list of dictionaries containing medication time, type, and details.
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
        Generate a schedule for position changes.
        
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
        Adjust intervention rates based on the current time.
        
        Args:
            current_time (datetime): The current simulation time.
        
        Returns:
            dict: Adjusted intervention rates.
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
    # Simulation and Vital Signs Generation
    # -------------------------------------------------------------------------
    def simulate(self):
        """
        Run the full simulation over the shift length.
        
        Iterates over time intervals, generates vital signs and applies scheduled
        and random interventions. Appends each simulation entry to self.data.
        """
        current_time = self.start_time
        end_time = self.start_time + timedelta(hours=self.shift_length)
        hrv_iter = iter(self.hrv_data)
        resp_rate_delta_iter = iter(self.resp_rate_delta_data)
        previous_body_temperature = None
        while current_time < end_time:
            # Update current state based on time of day
            self.current_state = "sleeping" if current_time.hour >= 22 or current_time.hour < 6 else "awake"
            body_temperature = round(random.uniform(36.5, 37.4), 1)
            if previous_body_temperature is None:
                temp_delta = None
            else:
                temp_delta = round(abs(body_temperature - previous_body_temperature), 1)
            previous_body_temperature = body_temperature

            entry = {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vital_signs": self.generate_vital_signs(hrv_iter, resp_rate_delta_iter, body_temperature, temp_delta),
            }

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

            # Random Interventions based on adjusted rates
            random_interventions = self.generate_interventions(current_time, existing_interventions=interventions)
            if random_interventions:
                interventions.extend(random_interventions)

            # Determine pain level and apply pain management if needed
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
        Generate a dictionary of vital signs for a simulation entry.

        Retrieves the next heart rate and respiration rate values from iterators,
        computes HR and respiratory deltas, and formats the values as strings.

        Args:
            hrv_iter (iterator): Iterator over heart rate delta values.
            resp_rate_delta_iter (iterator): Iterator over respiration rate delta values.
            body_temperature (float): Current body temperature.
            temp_delta (float): Change in body temperature since previous interval.

        Returns:
            dict: Vital signs with heart rate, respiratory rate, body temperature, and deltas.
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
        Return intervention details based on intervention type.
        
        For position change, selects a new patient position.
        For other types, returns a random detail string if applicable.

        Args:
            intervention_type (str): Type of intervention.

        Returns:
            str or None: Intervention detail description.
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
        Generate a list of random interventions based on adjusted rates.

        Args:
            current_time (datetime): The current simulation time.
            existing_interventions (list): Already scheduled interventions for the current time step.

        Returns:
            list or None: List of intervention dictionaries if any are generated; otherwise, None.
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

        Uses heart rate, respiratory rate, and body temperature to compute a pain score.

        Args:
            vital_signs (dict): Dictionary of vital sign readings.

        Returns:
            str: "High", "Moderate", or "Low" indicating pain level.
        """
        score = 0
        heart_rate = int(vital_signs["heart_rate"].split(' ')[0])
        respiratory_rate = int(vital_signs["respiratory_rate"].split(' ')[0])
        body_temperature = float(vital_signs["body_temperature"].split(' ')[0])

        if self.patient_profile.is_bradycardic:
            if heart_rate < 100:
                score += 2
            elif heart_rate < 80:
                score += 3
        elif self.patient_profile.is_tachycardic:
            print("I AM USING Tachycardic")
            if heart_rate > 195:
                score += 1
            elif heart_rate > 210:
                score += 2
        else:
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
    # Data Export Functions
    # -------------------------------------------------------------------------
    def export_data(self):
        """
        Export the simulation data to a JSON file.
        Converts NumPy float32 values to Python floats.
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

        if "/" in str(self.patient_profile.Age_days):
            Age = f"Post-Conceptual Age: {self.patient_profile.Age_days} Weeks/Days"
        else:
            Age = f"Age: {self.patient_profile.Age_days} Days"

        output_data = {
            "patient_id": self.patient_id,
            "Sex": self.gender,
            "Age": Age,
            "Weight": f"{self.patient_profile.weight} kg",
            "Conditions": self.patient_profile.conditions,
            "data": convert_floats(self.data)
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Exported JSON: {self.output_file}")

    def export_gan_values(self):
        """
        Export the GAN-generated values (heart rate, respiration rate, deltas) to CSV files.
        """
        if self.patient_profile.is_bradycardic and self.patient_profile.is_tachycardic:
            hr_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_Heart_Rate_BradyTachycardia.csv")
        elif self.patient_profile.is_bradycardic:
            hr_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_Heart_Rate_Bradycardia.csv")
        elif self.patient_profile.is_tachycardic:
            hr_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_GAN_Heart_Rate_Tachycardia.csv")
        else:
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
        Export the interventions recorded during the simulation to a CSV file.
        """
        interventions_output_file = os.path.join(self.patient_csv_dir, f"{self.patient_id}_Interventions.csv")
        interventions_list = []
        for entry in self.data:
            timestamp = entry["timestamp"]
            interventions = entry.get("intervention")
            if interventions:
                for intervention in interventions:
                    interventions_list.append({
                        "timestamp": timestamp,
                        "intervention_type": intervention.get("type"),
                        "intervention_detail": intervention.get("detail")
                    })
        interventions_df = pd.DataFrame(interventions_list)
        interventions_df.to_csv(interventions_output_file, index=False)
        print(f"Interventions CSV export completed: {self.patient_csv_dir}")

    def validate_gan_values(self):
        """
        Validate that the GAN-generated heart and respiration rate values match the JSON data.
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
