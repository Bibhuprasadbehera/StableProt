import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Set path
data_path = '/home/bibhu/Documents/temstampto/experiments/prepared_data_full.pt'

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    exit(1)

print(f"Loading {data_path}...")
data = torch.load(data_path)

# Extract temperatures from train, val, test
train_temps = data['train_temps']
val_temps = data['val_temps']
test_temps = data['test_temps']

all_temps = train_temps + val_temps + test_temps
all_temps = np.array(all_temps)

print("\n=== Temperature Distribution ===")
print(f"Total sequences: {len(all_temps)}")
print(f"Min Temp: {all_temps.min()}°C")
print(f"Max Temp: {all_temps.max()}°C")
print(f"Mean Temp: {all_temps.mean():.1f}°C")
print(f"Median Temp: {np.median(all_temps):.1f}°C")

# Define bins for a histogram
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
counts, _ = np.histogram(all_temps, bins=bins)

print("\nBin Counts:")
for i in range(len(bins)-1):
    print(f"  {bins[i]}-{bins[i+1]}°C: {counts[i]} sequences ({100*counts[i]/len(all_temps):.2f}%)")

# Check specifically for extreme thermophiles
extreme = np.sum(all_temps >= 80)
print(f"\nExtreme Thermophiles (>= 80°C): {extreme} sequences ({100*extreme/len(all_temps):.3f}%)")

# Check specifically for psychrophiles
cold = np.sum(all_temps <= 15)
print(f"Psychrophiles (<= 15°C): {cold} sequences ({100*cold/len(all_temps):.3f}%)")
