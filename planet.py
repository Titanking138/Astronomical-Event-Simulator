import numpy as np
import math

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

# Planet class
class planet():
    def __init__(self, name, mass, pos, vel,acc):
        self.mass = mass
        self.dotSize = linear_map(self.mass, 0, 200, 10, 750)
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.acc = np.array(acc, dtype=np.float64)
        self.pos_hist = []
        self.name = name

    def update_pos(self):
        self.pos_hist.append([self.pos[0], self.pos[1], self.pos[2]])
        self.pos += (self.vel * time_step)
    def update_position_and_velocity(self, time_step, apply_acceleration):
        if apply_acceleration:
            self.vel += self.acc * time_step  # Apply acceleration
        self.pos += self.vel * time_step  # Update position based on current velocity

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