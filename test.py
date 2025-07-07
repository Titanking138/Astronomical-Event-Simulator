import numpy as np
from typing import List
import pandas as pd
import math 
import exp
import planet
import matplotlib.pyplot  as plt

### SETTINGS ###
solar_extraVel = [0, 0, 0]  # Gives the solar system an extra velocity. (Au / yr)
N = 10  # Number of bodies
rand_massRange = [1, 25]  # Range of random bodies' mass (solar masses)
rand_posRange = [5, 5, 5]  # Range of random bodies' positions in AU (+-x, +-y, +-z)
rand_velRange = [1, 1, 1]  # Range of random bodies' velocities in AU / yr (+-x, +-y, +-z)

time_step = 0.1  # in years
time = 0  # keeps track of time
G = 4 * (math.pi**2)  # au^3 sm^-1 yr^-2
max_force = 10  # Stops bodies from flying away, mainly to make it look nice.



def csv_to_planet_objects(csv_file_path):
    """
    Load a CSV file and convert rows into a list of `planet` objects.

    :param csv_file_path: Path to the CSV file containing planet data.
    :return: List of `planet` objects.
    """
    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

    # Dynamically identify column names
    mass_col = [col for col in df.columns if 'mass' in col.lower()][0]
    pos_cols = [col for col in df.columns if 'position' in col.lower()]
    vel_cols = [col for col in df.columns if 'velocity' in col.lower()]

    # Check if required columns are present
    if not mass_col or len(pos_cols) != 3 or len(vel_cols) != 3:
        print("Error: Missing required columns in the CSV.")
        return []

    # Convert each row into a `planet` object
    planet_objects = []
    for index, row in df.iterrows():
        try:
            name = f"Planet_{index}"  # Assign a default name
            mass = row[mass_col]
            pos = row[pos_cols].values  # Extract position as [x, y, z]
            vel = row[vel_cols].values  # Extract velocity as [vx, vy, vz]

            # Create a `planet` object
            new_planet = planet.planet(name=name, mass=mass, pos=pos, vel=vel,acc = 0)
            planet_objects.append(new_planet)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    return planet_objects



class SpaceMassClustering:
    def __init__(self, gravitational_constant: float = 6.67430e-11):
        """
        Initialize the space mass clustering algorithm.

        :param gravitational_constant: Gravitational constant (default is standard SI value)
        """
        self.gravitational_constant = gravitational_constant

    def calculate_gravity_force(self, mass1: float, mass2: float, distance: float) -> float:
        """
        Calculate gravitational force between two masses with safe handling.

        :param mass1: Mass of first object
        :param mass2: Mass of second object
        :param distance: Distance between objects
        :return: Gravitational force
        """
        safe_distance = max(distance, 1e-10)  # Add a small epsilon to prevent divide by zero
        return self.gravitational_constant * mass1 * mass2 / (safe_distance ** 2)

    def cluster_masses(self, bodies: List['planet'], max_iterations: int = 50) -> List[List[int]]:
        """
        Cluster masses based on gravitational attraction.

        :param bodies: List of planet objects
        :param max_iterations: Maximum number of clustering iterations
        :return: List of cluster assignments (indices corresponding to the input list)
        """
        # Extract positions, masses, and indices
        positions = np.array([body.pos for body in bodies])
        masses = np.array([body.mass for body in bodies])
        indices = np.arange(len(bodies))

        # Initial centroids (select largest masses)
        sorted_indices = np.argsort(masses)[::-1]
        num_centroids = min(5, len(masses))  # Limit number of clusters
        centroid_indices = sorted_indices[:num_centroids]
        centroids = positions[centroid_indices]

        # Clustering iterations
        for _ in range(max_iterations):
            # Assign points to nearest centroid based on gravitational force
            clusters = [[] for _ in range(num_centroids)]

            for i, pos in enumerate(positions):
                if i in centroid_indices:
                    # Centroid always belongs to its own cluster
                    cluster_index = list(centroid_indices).index(i)
                    clusters[cluster_index].append(i)
                    continue

                # Calculate gravitational forces to each centroid
                forces = []
                for j, centroid in enumerate(centroids):
                    dist = np.linalg.norm(pos - centroid)
                    force = self.calculate_gravity_force(masses[centroid_indices[j]], masses[i], dist)
                    forces.append(force)

                # Assign to centroid with maximum gravitational force
                best_cluster = np.argmax(forces)
                clusters[best_cluster].append(i)

            # Recalculate centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroid = np.mean(positions[cluster], axis=0)
                    new_centroids.append(new_centroid)
                else:
                    # If cluster becomes empty, keep previous centroid
                    new_centroids.append(centroids[len(new_centroids)])

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = np.array(new_centroids)

        return clusters

# Example Usage
def main():
    # Example: List of `planet` objects
    # Path to your CSV file
    csv_file_path = "planets.csv"

    # Convert CSV data into a list of `planet` objects
    # planet_list = csv_to_planet_objects(csv_file_path)
    planet_list = exp.main()

    # Check the results
    for planet_obj in planet_list[:5]:  # Display the first 5 objects
        print(f"Name: {planet_obj.name}, Mass: {planet_obj.mass}, Pos: {planet_obj.pos}, Vel: {planet_obj.vel}")


    # Instantiate clustering algorithm
    clusterer = SpaceMassClustering()

    # Perform clustering
    clusters = clusterer.cluster_masses(planet_list)

    print("Clusters:", len(clusters))  # Output: Cluster assignments as lists of indices
    galaxy = []

    for j in range(len(clusters)):
        temp = []
        for i in clusters[j]:
                temp.append(planet_list[i])
        galaxy.append(temp)
    return galaxy

if __name__ == "__main__":
    main()
