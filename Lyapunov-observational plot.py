"""
Created on Sat Nov 16 17:09:23 2024

@author: Harry 

Code for Plotting the observational data pulled out of an ephemeris dataset 
extracted from the NASA JPL Horizons System website, compared to the peturbed trajectory 
made from incorporating lyapunov's exponent.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define Mercury's orbit equations in terms of position and velocity
def mercury_orbit(t, state):
    x, vx, y, vy = state
    r = np.sqrt(x**2 + y**2)
    # Gravitational constant times mass of the Sun (in AU^3/yr^2)
    GM = 4 * np.pi**2
    # Equations of motion
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    return [vx, ax, vy, ay]

# Initial conditions for Mercury's orbit (position in AU, velocity in AU/year)
initial_state = [0.39, 0, 0, 9.96]  # Mercury's average orbital radius and velocity

# Time span for one full orbit (88 days = 0.2407 years)
time_span = (0, 0.2407)  # 88 days in years
time_eval = np.linspace(0, 0.2407, 1000)  # Time evaluation points

# Solving the orbit for Mercury's trajectory (numerical integration)
solution_mercury = solve_ivp(mercury_orbit, time_span, initial_state, t_eval=time_eval)

# Example file_path - Replace with your actual file path
file_path = 'horizons_results.txt'

# Initialize lists to hold the position (x) and velocity (vx) values from the observational data
x_values = []
vx_values = []

# Open the file and parse the data
with open(file_path, 'r') as file:
    for line in file:
        # Skip lines that are empty
        if line.strip() == '':
            continue
        
        # Split the line by commas (since the data is comma-separated)
        line_parts = line.split(',')
        
        # Ensure the line has at least 6 values (to safely extract x and vx)
        if len(line_parts) >= 6:
            try:
                # Extract x (3rd element) and vx (6th element)
                x = float(line_parts[2])  # 3rd value in the line
                vx = float(line_parts[5])  # 6th value in the line
                
                # Append the extracted values to the lists
                x_values.append(x)
                vx_values.append(vx)
            except ValueError:
                # Skip lines with invalid or missing data
                continue

# Check if valid data was collected
if len(x_values) == 0 or len(vx_values) == 0:
    print("Error: No valid data found to plot.")
else:
    # Convert the simulated Mercury's velocities from AU/year to AU/day (divide by 365.25)
    mercury_vx_values = solution_mercury.y[1] / 365.25  # Velocity in AU/day

    # Plotting both the observational data and Mercury's orbit
    plt.figure(figsize=(10, 6))

    # Plot observational data (X position vs X velocity)
    plt.scatter(x_values, vx_values, color='red', label='Observational Data', marker='o', s=10)

    # Plot Mercury's orbit (X position vs X velocity from the numerical solution)
    plt.plot(solution_mercury.y[0], mercury_vx_values, label="Mercury Orbit (Model)", color="green", linestyle='-', linewidth=2)

    # Labels and title
    plt.title('X Position vs X Velocity: Observational Data vs. Mercury Orbit')
    plt.xlabel('X Position (AU)')
    plt.ylabel('X Velocity (AU/day)')

    # Display the legend
    plt.legend()

    # Show the grid
    plt.grid(True)

    # Show the plot
    plt.show()
