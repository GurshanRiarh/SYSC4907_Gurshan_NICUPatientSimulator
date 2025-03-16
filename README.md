The NICU Patient Simulator is a simulation tool for neonatal intensive care units, designed to generate synthetic vital sign data, model interventions, and provide visualization tools for analysis. The simulator supports both healthy and unhealthy neonates, allowing for the study of conditions like bradycardia and tachycardia.

Features: 
Synthetic Data Generation: Uses GAN-based models to generate heart rate and respiratory rate data.
Simulation Modes: Supports preterm and neonate simulations with various conditions.
Patient Monitoring: Models interventions like feeding, medication, and oxygen administration.
Data Export: Outputs simulation results in JSON and CSV formats.
Visualization: Includes static and interactive plots for vital signs.
Web Interface: Flask-based UI for setting up and running simulations.

Installation: 
1. Clone the repository: git clone https://github.com/your-username/NICU-Patient-Simulator.git cd NICU-Patient-Simulator
2. Install dependencies: pip install -r requirements.txt

Usage: 
1. Run the Flask application: python ExecuteSimulation.py (Also located in UH_v25/Unhealthy_Files/GUI_Files)
2. Access the UI at: http://localhost:5000 (It should auto run after executing the above step)

Outputs: 
JSON Files: Contains patient data with timestamps and all data.
CSV Files: Logs generated heart rate, respiratory rate, body temperature, and intervention data.
Plots: Visualizations for vital sign trends over the shift.

Contributors
Gurshan Riarh
Jesse Levine
