# NICU Patient Simulator

The NICU Patient Simulator is a web-based application designed to generate **synthetic neonatal intensive care unit (NICU) patient data**. It includes both **Preterm** and **Neonate** simulations, supporting normal and abnormal cardiac conditions such as bradycardia or tachycardia.

The system uses **pre-trained GAN (Generative Adversarial Network) models** to generate realistic vital sign data—heart rate, respiratory rate, and body temperature—along with routine interventions (e.g., feeding, medication, diaper change, etc.). It also supports visualization of the results in both **static** and **interactive plots** via Matplotlib and Plotly.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Dependencies & Installation](#dependencies--installation)
4. [Running the Simulator](#running-the-simulator)
5. [Usage & Workflow](#usage--workflow)
6. [Key Modules](#key-modules)
   - [ExecuteSimulation.py](#executesimulationpy)
   - [NICU_Simulator_Params_Unhealthy.py / NICU_Simulator_Parmas.py](#nicu_simulator_params_unhealthypy--nicu_simulator_parmaspy)
   - [Vital_Sign_Static_Plots_Unhealthy.py](#vital_sign_static_plots_unhealthypy)
   - [Vital_Sign_Interactive_Plots_Unhealthy.py](#vital_sign_interactive_plots_unhealthypy)
   - [HTML/CSS/JS Front-End Files](#htmlcssjs-front-end-files)
7. [Data Output](#data-output)
8. [License](#license)
9. [Contact](#contact)


## Features

- **Preterm Simulation**: Simulates neonates with normal heart rate conditions (110–150 bpm).
- **Neonate Simulation**: Simulates neonates with potential bradycardia or tachycardia (or both).
- **Dynamic Interventions**: Random & scheduled interventions (e.g., feeding, diaper change) are added at each time interval.
- **GAN-Driven Synthetic Data**: Trained GAN models produce realistic heart rate, respiratory rate, and temperature values.
- **Interactive & Static Plots**: Quickly visualize time-series data via Plotly (HTML) or Matplotlib (PNG).
- **Flexible Output**: Exports data in JSON and CSV formats, plus a single ZIP archive of all simulation files.


## Project Structure

A simplified overview of the primary files and folders:
```
Project/
│
├── ExecuteSimulation.py
├── NICU_Simulator_Parmas.py
├── NICU_Simulator_Params_Unhealthy.py
├── Vital_Sign_Static_Plots_Unhealthy.py
├── Vital_Sign_Interactive_Plots_Unhealthy.py
│
├── Trained_GAN_Path_Files/
│   ├── trained_generator_HeartRate_again.pth
│   ├── trained_generator_ResprationRate_again.pth
│   ├── trained_generator_Bradycardia.pth
│   ├── trained_generator_Tachycardia.pth
│   └── (...additional model files...)
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── simulation_mode.html
│   └── (...Jinja2 HTML templates used by Flask...)
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   └── result.js
│   └── (...images, e.g. NICU_ENVIRONMENT.jpg if needed...)
│
├── Output_Patient_Data/
│   └── (...simulation output: JSON, CSVs, plots, ZIPs, etc.)
│
└── README.md (this file)
```

## Dependencies & Installation

1. **Python 3.7+** recommended.

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   # or
   .\venv\Scripts\activate      # Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt`, ensure you install:
   - Flask
   - Torch (PyTorch)
   - NumPy
   - SciPy
   - Matplotlib
   - Plotly
   - Pandas

   Example:
   ```bash
   pip install flask torch numpy scipy matplotlib plotly pandas
   ```

4. **(Optional) Place pretrained GAN models** in `Trained_GAN_Path_Files/`. Make sure the filenames match the paths in the code (e.g., `trained_generator_HeartRate_again.pth`, etc.).

---

## Running the Simulator

1. From the project’s root directory, run:

   ```bash
   python ExecuteSimulation.py
   ```

2. The server will start at **`http://127.0.0.1:5000`** and open your default browser automatically (if your environment allows it).

3. You will see the **Simulation Mode** selection page with two buttons:
   - **Preterm Simulation**
   - **Neonate Simulation**

Choose the mode that you want to simulate.

---

## Usage & Workflow

1. **Open the Home Page**: The app shows two main simulation modes:
   - **Preterm Simulation**: For normal heart rate ranges (roughly 110–150 bpm).
   - **Neonate Simulation**: For potentially unhealthy conditions (bradycardia and/or tachycardia).

2. **Fill Out the Form**:
   - **Patient ID** (unique identifier for each simulation).
   - **Start Time** in **HH:MM:SS**.
   - **Patient Condition** (varies if you choose preterm or neonate).
   - **Age** in days (or weeks/days for preterm).
   - **Interventions** to simulate (feeding, medication, etc.).
   - **Weight** and **Sex** of the neonate.

3. **Click “Run Simulation”**:
   - The backend spawns a simulation:
     - Creates synthetic vital sign data via GAN models.
     - Logs scheduled and random interventions.
     - Exports data to JSON and CSV files.
     - Generates plots (both interactive HTML and static PNG).

4. **Check the Results**:
   - You’ll see a **Result** page summarizing the simulation.
   - **Data Files Tab**: Download JSON/CSV or a full ZIP.
   - **JSON Tab**: View raw JSON data inline.
   - **Data Visualization Tab**: Interactive Plotly time-series charts.
   - **Generated CSVs Tab**: Preview or download CSV data from within the browser.

5. **(Optional) Download All Data**:
   - Use **“Download All Data”** to retrieve a ZIP containing every JSON, CSV, and generated plot in one package.

---

## Key Modules

### ExecuteSimulation.py
- **Main Flask application** that wires up all routes:
  - `/` : Home page to choose a simulation mode.
  - `/setup`: The form-based setup page.
  - `/simulate`: Processes form submissions, invokes simulation logic.
  - Additional routes for **downloading** data, **displaying** CSVs/JSON, and generating/serving plots.
- Creates directories for each **Patient ID** in `Output_Patient_Data/` and organizes JSON, CSV, plot files.

### NICU_Simulator_Parmas.py / NICU_Simulator_Params_Unhealthy.py
- Define the **GAN Generators** (PyTorch `nn.Module` classes).
- Load **pretrained models** from `Trained_GAN_Path_Files/`.
- Contain classes:
  - `PatientProfile` – Basic patient info (age, weight, conditions).
  - `NeonateSimulator` – Core class that:
    - Generates synthetic data from the loaded GAN.
    - Applies smoothing, calculates deltas, schedules interventions.
    - Exports final data to JSON, CSV, and validates them.

- **_Parmas.py**: Typically for the preterm or “healthy” mode.  
- **_Params_Unhealthy.py**: For bradycardic/tachycardic neonates.

### Vital_Sign_Static_Plots_Unhealthy.py
- Uses **Matplotlib** to generate **static PNG** time-series plots:
  - Heart Rate
  - Respiratory Rate
  - Body Temperature
  - Deltas (changes) of the above
- Plots are saved in a `static_plots` folder inside each patient’s output directory.

### Vital_Sign_Interactive_Plots_Unhealthy.py
- Uses **Plotly** to build **interactive HTML** plots with hover features, zooming, etc.
- Each plot is saved as an `.html` file in an `interactive_plots` folder inside each patient’s output directory.

### HTML/CSS/JS Front-End Files
- **templates/simulation_mode.html** – Landing page for choosing Preterm or Neonate simulation.
- **templates/index.html** – The main input form collecting patient info and interventions.
- **templates/result.html** – Displays simulation outputs (JSON, CSV, plots) after the run completes.
- **static/css/style.css** – Custom styling for forms, navbars, and result displays.
- **static/js/main.js** – Manages dynamic form interactions (e.g., start time, age, condition dropdown).
- **static/js/result.js** – Manages results page interactions (e.g., loading CSV previews, interactive plot frames).

---

## Data Output

Inside `Output_Patient_Data/` a subdirectory is created for every **Patient ID**:

```
Output_Patient_Data/
  └── MyTestPatient/
      ├── json/
      │   └── MyTestPatient_Data.json
      ├── csv/
      │   ├── MyTestPatient_GAN_Heart_Rate_Normal.csv
      │   ├── MyTestPatient_Interventions.csv
      │   └── ...
      ├── static_plots/
      │   └── PNG plots
      └── interactive_plots/
          └── HTML plots
```

- **JSON**: Contains vital signs data and interventions for every simulated interval.
- **CSV**: Contains the raw numeric columns for heart rate, respiratory rate, deltas, etc.
- **static_plots** (PNG) & **interactive_plots** (HTML): Graphical summaries of your simulation data.

You can download files individually or as `all_data.zip`.

---

## Contact

For additional help or inquiries:
- **Developer**: [GurshanRiarh]
- **Email**: [Gurshan01@gmail.com]
