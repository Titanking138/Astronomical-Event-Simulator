import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from planet import planet
import random
import time



def simulate_big_bang(
    initial_mass, num_fragments, explosion_energy, max_time, time_steps, accel_duration, evaporation_radius
):
    directions = np.random.randn(num_fragments, 3)
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    velocities = np.sqrt(2 * explosion_energy / initial_mass) * np.random.rand(num_fragments)
    accelerations = velocities / accel_duration  # Acceleration based on duration

    # Fragment masses with variability
    fragment_masses = np.random.normal(loc=initial_mass / num_fragments, scale=initial_mass / (10 * num_fragments), size=num_fragments)
    fragment_masses = np.maximum(fragment_masses, 0.1)  # Prevent negative masses

    planets = []
    for i in range(num_fragments):
        position = [0, 0, 0]
        velocity = np.zeros(3)
        acceleration = accelerations[i] * directions[i]
        planet_obj = planet(
            f"Fragment_{i+1}",
            fragment_masses[i],
            position,
            velocity,
            acceleration,
 
        )
        planets.append(planet_obj)

    times = np.linspace(0, max_time, time_steps)
    planets_over_time = []

    for t_idx, t in enumerate(times):
        apply_acceleration = t <= accel_duration
        current_positions = []
        for planet_inst in planets:
            planet_inst.update_position_and_velocity(times[1] - times[0], apply_acceleration)
            current_positions.append(planet_inst.pos)
        planets_over_time.append(np.array(current_positions))


    return times, planets_over_time, planets


def main():
    # Parameters for Big Bang
    initial_mass = 1e12  # Arbitrary large mass
    num_fragments = random.randint(10,50)  # Vast number of fragments to simulate the Big Bang
    explosion_energy = 1e15  # Immense energy to represent the scale of the Big Bang
    max_time = 1000  # Simulate over a longer time span
    time_steps = 500  # Number of time steps (finer resolution)
    accel_duration = 50  # Acceleration phase lasts 50 seconds
    evaporation_radius = 1000

    # Run the Big Bang simulation
    times, positions_over_time, planets = simulate_big_bang(
        initial_mass, num_fragments, explosion_energy, max_time, time_steps, accel_duration, evaporation_radius
    )

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for t_idx, positions in enumerate(positions_over_time):
        ax.clear()
        ax.set_title(f"Big Bang Simulation at t = {times[t_idx]:.2f}year")
        max_distance = np.max(np.linalg.norm(positions, axis=1))
        limit = max_distance + 100
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        # Plot particle positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=300, label="Particles", color="blue")
        ax.scatter(0, 0, 0, s=1000, label="Origin", color="red")
        ax.legend()

        # Pause to visualize the animation smoothly
        plt.pause(0.01)

    plt.show()

    return planets


if __name__ == '__main__':
    main()