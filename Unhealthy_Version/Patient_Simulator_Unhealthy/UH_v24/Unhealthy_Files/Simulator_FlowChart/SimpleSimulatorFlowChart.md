# NICU Patient Simulator Flow

- **User Interface (Web)**
  - **Mode Selection**
    - Preterm (Healthy) Simulation
    - Neonate (Unhealthy) Simulation
  - **Patient Information Input**
    - Patient ID, Start Time, Age, Weight, Gender
    - Select Condition (Normal, Bradycardia, Tachycardia, Both)
    - Choose Interventions

- **Simulation Engine**
  - **Simulator Instantiation**
    - Healthy Simulator (Preterm) OR Unhealthy Simulator (Neonate)
  - **Synthetic Data Generation**
    - Load Pre-trained GAN Models
      - Heart Rate & Respiration Rate Generators
      - (For Neonates) Additional Bradycardia/Tachycardia Generators
    - Generate Vital Signs
      - Apply Noise & Normalization
      - Smoothing & Delta Calculation
  - **Intervention Scheduling & Pain Assessment**
    - Scheduled Interventions (Feeding, Medication, etc.)
    - Random Interventions based on Poisson rates
    - Determine Pain Level from Vital Signs

- **Data Export & Visualization**
  - **Data Export**
    - JSON File with Patient & Simulation Data
    - CSV Files for GAN Outputs & Interventions
  - **Visualization**
    - Static Plots (Matplotlib PNGs)
    - Interactive Plots (Plotly HTML)

- **Web Application (Flask)**
  - Displays Simulation Results
  - Provides Download Options for Data & Plots
