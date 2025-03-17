import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'D:/GitHub/SYSC4907_Capstone/GAN_Heart_v2/infant_2_8h_respiration_rate_outlierRem.csv'
df = pd.read_csv(file_path)

columns = ['respiration_rate']
if columns[0] not in df.columns:
    raise ValueError(f"Column '{columns[0]}' not found in the dataset.")

data = df[columns].to_numpy().reshape(-1, 1)

min_data_value = data.min()
max_data_value = data.max()
print(f"Data range before normalization: {min_data_value} to {max_data_value}")

min_val, max_val = 30, 70
data_normalized = (data - min_val) / (max_val - min_val)
data_normalized = torch.tensor(data_normalized, dtype=torch.float32)

print(f"Data range after normalization: {data_normalized.min().item()} to {data_normalized.max().item()}")

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
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
        return self.model(noise)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

noise_dim = 10
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

epochs = 5000
batch_size = 16
real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)

for epoch in range(epochs):
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

    generator.zero_grad()
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_data = generator(noise)
    g_loss = criterion(discriminator(fake_data), real_labels)
    g_loss.backward()
    optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

with torch.no_grad():
    noise = torch.randn(96, noise_dim).to(device)
    synthetic_data = generator(noise).cpu().numpy()

real_data_denormalized = data_normalized.numpy() * (max_val - min_val) + min_val
synthetic_data_denormalized = synthetic_data * (max_val - min_val) + min_val

print(f"Real Data range after denormalization: {real_data_denormalized.min()} to {real_data_denormalized.max()}")
print(f"Synthetic Data range after denormalization: {synthetic_data_denormalized.min()} to {synthetic_data_denormalized.max()}")

plt.figure(figsize=(10, 6))
plt.plot(real_data_denormalized, label="Real Data", color="blue", alpha=0.6)
plt.plot(synthetic_data_denormalized, label="Synthetic Data", color="orange", linestyle="--", alpha=0.8)
plt.xlabel("Time Steps")
plt.ylabel("Respiration Rate (Breaths Per Minute)")
plt.title("Comparison of Real and Synthetic Neonate Respiration Rate Data")
plt.legend()
plt.show()

torch.save(generator.state_dict(), 'trained_generator_ResprationRate_again.pth')
