import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_velocities(data):
    """Calculate velocities from joint angle data"""
    velocities = {}
    timestamps = data['timestamp'].values
    dt = 0.1  # 10 Hz sampling rate
    
    for column in data.columns:
        if column != 'timestamp':
            angles = data[column].values
            # Calculate velocity as difference between consecutive angles divided by time step
            velocity = np.diff(angles) / dt
            velocities[column] = velocity
            
    return velocities, timestamps[1:]  # Return velocities and corresponding timestamps

def plot_velocities(velocities, timestamps):
    """Plot joint velocities over time"""
    plt.figure(figsize=(15, 10))
    
    # Plot left arm velocities
    plt.subplot(2, 1, 1)
    for joint in velocities:
        if 'left' in joint:
            plt.plot(timestamps, velocities[joint], label=joint)
    plt.title('Left Arm Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot right arm velocities
    plt.subplot(2, 1, 2)
    for joint in velocities:
        if 'right' in joint:
            plt.plot(timestamps, velocities[joint], label=joint)
    plt.title('Right Arm Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Read the CSV file
    data = pd.read_csv('test.csv')
    
    # Calculate velocities
    velocities, timestamps = calculate_velocities(data)
    
    # Plot velocities
    plot_velocities(velocities, timestamps)
    
    # Print maximum velocities for each joint
    print("\nMaximum velocities (rad/s):")
    for joint in velocities:
        print(f"{joint}: {np.max(np.abs(velocities[joint])):.2f}")

if __name__ == "__main__":
    main() 