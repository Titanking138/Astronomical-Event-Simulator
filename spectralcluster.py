from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
# Define gravitational similarity matrix

solar_extraVel = [0, 0, 0]  # Gives the solar system an extra velocity. (Au / yr)
N = 10  # Number of bodies
rand_massRange = [1, 25]  # Range of random bodies' mass (solar masses)
rand_posRange = [5, 5, 5]  # Range of random bodies' positions in AU (+-x, +-y, +-z)
rand_velRange = [1, 1, 1]  # Range of random bodies' velocities in AU / yr (+-x, +-y, +-z)

time_step = 0.01  # in years
time = 0  # keeps track of time
G = 4 * (math.pi**2)  # au^3 sm^-1 yr^-2
max_force = 10  # Stops bodies from flying away, mainly to make it look nice.


class planet():
    def __init__(self, name, mass, pos, vel):
        self.mass = mass
        self.dotSize = linear_map(self.mass, 0, 200, 10, 750)
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.pos_hist = []
        self.name = name

    def update_pos(self):
        self.pos_hist.append([self.pos[0], self.pos[1], self.pos[2]])
        self.pos += (self.vel * time_step)

def linear_map(value, minIn, maxIn, minOut, maxOut):
    if value >= maxIn:
        return(maxOut)
    elif value <= minIn:
        return(minOut)
    else:
        spanIn = maxIn - minIn
        spanOut = maxOut - minOut
        scale = (value - minIn) / spanIn
        output = minOut + (scale * spanOut)
        return(output)
    
def main():
    G = 6.67430e-11  # Gravitational constant
    data = pd.read_csv("star_systems.csv")
    positions = data[['Position_x', 'Position_y', 'Position_z']].values
    masses = data['Mass'].values

    # Compute pairwise distances
    distances = pdist(positions)
    distances = squareform(distances)  # Convert to matrix form

    # Avoid division by zero
    distances[distances == 0] = np.inf

    # Compute gravitational influence
    gravity_matrix = np.zeros_like(distances)
    for i in range(len(masses)):
        for j in range(len(masses)):
            gravity_matrix[i, j] = G * masses[i] * masses[j] / (distances[i, j]**2)

    # Normalize the matrix for clustering
    gravity_matrix /= gravity_matrix.max()

    # Apply Spectral Clustering
    spectral = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
    data['cluster'] = spectral.fit_predict(gravity_matrix)
    print(data['cluster'])
    clusters = []

    for j in range(14):
        temp = []
        for _, row in data.iterrows():  # Iterate over rows
            if row['cluster'] == j:  # Access the 'cluster' column in the row
                p = planet(
                    row['Name'], 
                    row['Mass'], 
                    [row['Position_x'], row['Position_y'], row['Position_z']], 
                    [row['Velocity_x'], row['Velocity_y'], row['Velocity_z']]
                )
                temp.append(p)
        clusters.append(temp)
    return clusters


    # # Visualization
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(data['Position_x'], data['Position_y'], data['Position_z'], c=data['cluster'], cmap='viridis', s=50)
    # ax.set_xlabel('Position X')
    # ax.set_ylabel('Position Y')
    # ax.set_zlabel('Position Z')
    # plt.colorbar(scatter, label='Cluster')
    # plt.title('Spectral Clustering with Gravitational Influence')
    # plt.show()
