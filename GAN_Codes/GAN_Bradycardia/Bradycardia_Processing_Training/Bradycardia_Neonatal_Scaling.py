import pandas as pd
import numpy as np

# Load the adult BPM dataset
bpm_file = "/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/GAN_Bradycardia/CSV_DATA/BPM_8hr_5min.csv"
df = pd.read_csv(bpm_file, header=None)  # Assuming no headers in file

# Define adult and neonatal BPM ranges
adult_normal_range = (60, 100)
adult_brady_range = (df.min().min(), 60)  # Anything <60 is bradycardia
neonatal_normal_range = (100, 160)  # Shift lower to allow more bradycardia
neonatal_brady_range = (60, 90)  # Increase bradycardia severity

# Function to rescale values with enhanced bradycardia effect
def rescale_bpm(data, old_min, old_max, new_min, new_max):
    return ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

# Apply rescaling
df_neonatal = df.copy()
# Scale normal BPM values (60-100) -> Neonatal normal range (100-160)
df_neonatal[df >= 60] = rescale_bpm(df[df >= 60], *adult_normal_range, *neonatal_normal_range)

# Scale bradycardia BPM values (<60) -> Neonatal bradycardia range (30-85)
df_neonatal[df < 60] = rescale_bpm(df[df < 60], *adult_brady_range, *neonatal_brady_range)

# Introduce slight random variation to bradycardia values to prevent uniform scaling
brady_mask = df_neonatal < 85
df_neonatal[brady_mask] += np.random.uniform(-5, 5, size=df_neonatal[brady_mask].shape)

# Save the rescaled neonatal dataset
neonatal_bpm_file = "/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/GAN_Bradycardia/Scaled_CSV_DATA/BPM_8hr_5min_Neonatal.csv"
df_neonatal.to_csv(neonatal_bpm_file, index=False)

print(f"Neonatal BPM data saved: {neonatal_bpm_file}")


import pandas as pd

# Load the neonatal BPM dataset
neonatal_bpm_file = "/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/GAN_Bradycardia/Scaled_CSV_DATA/BPM_8hr_5min_Neonatal.csv"
df_neonatal = pd.read_csv(neonatal_bpm_file, header=None)

# Remove the first row if it contains index-like values
if (df_neonatal.iloc[0] == range(96)).all():  # Check if first row is just index values
    df_neonatal = df_neonatal.iloc[1:].reset_index(drop=True)  # Remove it

# Convert to numeric to ensure proper processing
df_neonatal = df_neonatal.apply(pd.to_numeric, errors="coerce")

# Define neonatal bradycardia thresholds
brady_threshold = 100
severe_brady_threshold = 80

# Identify bradycardia instances (BPM < 100)
bradycardia_instances = df_neonatal[df_neonatal < brady_threshold].stack().reset_index()
bradycardia_instances.columns = ["Row", "Column", "BPM"]

# Identify severe bradycardia instances (BPM < 80)
severe_brady_instances = bradycardia_instances[bradycardia_instances["BPM"] < severe_brady_threshold]

# Print the detected bradycardia instances
print("\nBradycardia Instances (BPM < 100):")
print(bradycardia_instances)

print("\nSevere Bradycardia Instances (BPM < 80):")
print(severe_brady_instances)
