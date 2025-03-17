import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BPM sequences from CSV
file_path = "/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/GAN_Bradycardia/Scaled_CSV_DATA/BPM_8hr_5min_Neonatal.csv"
df = pd.read_csv(file_path, header=None)  # Assuming no column headers

df.head()  # Display the first few rows to check the data structure

# Convert data into NumPy array
data = df.to_numpy(dtype=np.float32)  # Shape: (num_samples, 96)


# Normalize the BPM data for GAN training (0 to 1 range)
min_val, max_val = data.min(), data.max()
data_normalized = (data - min_val) / (max_val - min_val)


# Convert to PyTorch tensor
data_normalized = torch.tensor(data_normalized, dtype=torch.float32).to(device)

# Define the GAN model with 96-dimensional input/output
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.model(noise)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    


# Set dimensions
noise_dim = 10  # Random noise input size
output_dim = 96  # Each output is a 96-point sequence

# Initialize GAN models
generator = Generator(noise_dim, output_dim).to(device)
discriminator = Discriminator(output_dim).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
epochs = 5000
batch_size = 16
real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)

for epoch in range(epochs):
    # Train Discriminator
    discriminator.zero_grad()
    idx = torch.randint(0, data_normalized.size(0), (batch_size,))
    real_data = data_normalized[idx].to(device)
    real_loss = criterion(discriminator(real_data), real_labels)

    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_data = generator(noise).detach()
    fake_loss = criterion(discriminator(fake_data), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    generator.zero_grad()
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_data = generator(noise)
    g_loss = criterion(discriminator(fake_data), real_labels)
    g_loss.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate synthetic BPM sequences
with torch.no_grad():
    noise = torch.randn(10, noise_dim).to(device)  # Generate 10 synthetic 96-point sequences
    synthetic_data = generator(noise).cpu().numpy()


# Denormalize the synthetic data back to original BPM range
synthetic_data_denormalized = synthetic_data * (max_val - min_val) + min_val

# Save synthetic data
synthetic_output_file = "/home/Gurshan.R/Documents/GitHub/SYSC4907_Capstone/GAN_Bradycardia/Synthetic_CSV_DATA/BPM_8hr_5min_Synthetic.csv"
np.savetxt(synthetic_output_file, synthetic_data_denormalized, delimiter=",")

print(f"Synthetic BPM data saved: {synthetic_output_file}")

# Plot real vs synthetic data for comparison
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data_denormalized[0], label="Synthetic BPM", linestyle="--", color="orange", alpha=0.8)
plt.plot(data[:10].mean(axis=0), label="Real BPM (Average)", linestyle="-", color="blue", alpha=0.6)
plt.xlabel("Time Steps (5-min intervals)")
plt.ylabel("BPM")
plt.title("Comparison of Real and Synthetic BPM Data")
plt.legend()
plt.grid()
plt.show()