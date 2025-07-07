import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math
from multiprocessing import Pool
import test
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

# Updates velocities and positions of bodies
def update_vels(body_list):
    for current in body_list:
        force = np.array([0.0, 0.0, 0.0])
        for other in body_list:
            if other != current:
                try:
                    radius = np.linalg.norm(current.pos - other.pos)
                    r_hat = (other.pos - current.pos) / np.linalg.norm(other.pos - current.pos)
                    force_norm = (G * other.mass) / ( radius**2 )
                    force += (r_hat * force_norm * time_step)

                    if np.linalg.norm(force) > max_force:
                        force /= np.linalg.norm(force)
                        force *= max_force
                except ZeroDivisionError:
                    pass
                    
        acc = force
        current.vel += acc

def update_poss(body_list):
    for current in body_list:
        current.update_pos()

def printInfo(list):
    for bdy in list:
        print('Body:', bdy.name)
        print('Body Mass: ', bdy.mass)
        print('Initial Position:', bdy.pos)
        print('Initial Velocity:', bdy.vel)
        print('=====')

# Initializes the bodies and returns the list
def init_bodies():
    bodies_list = []  # Initialize the list of bodies
    for n in range(N):
        mass = random.uniform(rand_massRange[0], rand_massRange[1])
        pos_x = random.uniform(-rand_posRange[0],rand_posRange[0])
        pos_y = random.uniform(-rand_posRange[1],rand_posRange[1])
        pos_z = random.uniform(-rand_posRange[2],rand_posRange[2])
        vel_x = random.uniform(-rand_velRange[0],rand_velRange[0])
        vel_y = random.uniform(-rand_velRange[1],rand_velRange[1])
        vel_z = random.uniform(-rand_velRange[2],rand_velRange[2])
        if n==0 :
            mass = 2000
        bodies_list.append(planet(str(n), mass, [pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z]))  
    return bodies_list

# PLOTTING AND ANIMATION
def animate(i, simulation_id, bodies_list):
    global time
    
    update_vels(bodies_list)
    update_poss(bodies_list)
    
    plt.cla()

    ax.set_title(f'Gravity Simulation for galaxy number {simulation_id} ! Time: ' +str(time)+' years.')
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(-10, 10)

    for bdy in bodies_list:
        x_live = [element[0] for element in bdy.pos_hist]
        y_live = [element[1] for element in bdy.pos_hist]
        z_live = [element[2] for element in bdy.pos_hist]
        x_last = bdy.pos[0]
        y_last = bdy.pos[1]
        z_last = bdy.pos[2]

        ax.plot3D(x_live, y_live, z_live)                # To see a trail for the path of the Planet
        ax.scatter3D(x_last, y_last, z_last, s = bdy.dotSize)
        ax.text(x_last, y_last, z_last, bdy.name)
                
    time = round(time + time_step, 2)

# Function to run the simulation
def run_simulation(simulation_id, bodies_list,total_steps):
    # Prepare figure and axes for 3D plotting
    fig = plt.figure()
    global ax
    ax = fig.add_subplot(111, projection='3d')
    printInfo(bodies_list)
    # print(bodies_list)
    ani = FuncAnimation(fig, animate, fargs=(simulation_id, bodies_list,), interval=1, frames=total_steps, repeat=False)
    plt.show()
    # ani.save("nbody.gif")

def run_simulation_wrapper(args):
    """Wrapper function to unpack arguments for run_simulation."""
    galaxy_index, bodies_list, total_steps = args
    run_simulation(galaxy_index, bodies_list, total_steps)
# Entry point for running the simulation as a process
# if __name__ == '__main__':
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     # if rank == 0:
#     #     bodies_list = init_bodies()
#     # else:
#     #     bodies_list = None
#     # bodies_list = comm.bcast(bodies_list, root=0)
#     bodies_list = init_bodies()  # Initialize the bodies and get the list
#     total_steps = 200
#     num_galaxies = 4
#     for i in range(0,num_galaxies):
#         if (i%size==0):

#             run_simulation(i,bodies_list,total_steps)  # Run the simulation by passing bodies_list and total_steps


if __name__ == '__main__':
    total_steps = 200
    # Initialize the bodies and get the list
    clusters = test.main()
    num_galaxies = len(clusters)
    # num_galaxies = 1
    print(f"number of clusters are: {num_galaxies}")


    # # clusters[0][0].mass*=10e7
    # # clusters[0][0].vel = [0,0,1]

    # Create a list of arguments for each galaxy simulation
    tasks = [(i, clusters[i][0:10], total_steps) for i in range(num_galaxies)]
    # tasks = [(i, bodies_list, total_steps) for i in range(1)]



    # Use Pool to run simulations in parallel
    with Pool(processes=num_galaxies) as pool:
        pool.map(run_simulation_wrapper, tasks)

    print("Simulations completed.")