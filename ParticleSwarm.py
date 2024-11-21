import numpy as np
import random

grid_size = (20, 20)
obstacle_map = np.zeros(grid_size)
obstacle_map[5:10, 5:10] = 1  

num_particles = 30
num_iterations = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5

start_pos = np.array([0, 0])
target_pos = np.array([grid_size[0] - 1, grid_size[1] - 1])

def fitness(position):
    if obstacle_map[int(position[0]), int(position[1])] == 1:
        return float('inf')  
    return np.linalg.norm(position - target_pos)  

particles = [np.array([random.uniform(0, grid_size[0]-1), random.uniform(0, grid_size[1]-1)]) for _ in range(num_particles)]
velocities = [np.random.uniform(-1, 1, 2) for _ in range(num_particles)]
personal_best_positions = particles[:]
personal_best_scores = [fitness(p) for p in particles]

global_best_position = min(personal_best_positions, key=fitness)
global_best_score = fitness(global_best_position)

for iteration in range(num_iterations):
    for i, particle in enumerate(particles):
        inertia = inertia_weight * velocities[i]
        cognitive = cognitive_coeff * random.random() * (personal_best_positions[i] - particle)
        social = social_coeff * random.random() * (global_best_position - particle)
        velocities[i] = inertia + cognitive + social

        particles[i] = particles[i] + velocities[i]
   
        particles[i] = np.clip(particles[i], [0, 0], [grid_size[0] - 1, grid_size[1] - 1])

        current_fitness = fitness(particles[i])
     
        if current_fitness < personal_best_scores[i]:
            personal_best_positions[i] = particles[i]
            personal_best_scores[i] = current_fitness
     
        if current_fitness < global_best_score:
            global_best_position = particles[i]
            global_best_score = current_fitness

    print(f"Iteration {iteration+1}: Best Score = {global_best_score}, Best Position = {global_best_position}")

print("\nBest Path Solution Found:")
print(f"Position = {global_best_position}, Distance to Target = {global_best_score}")