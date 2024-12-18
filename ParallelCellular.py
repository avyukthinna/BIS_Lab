import numpy as np

num_servers = 10  
num_tasks = 100  
max_iterations = 50  
neighborhood_size = 2  

server_loads = np.random.randint(1, 10, size=num_servers)

total_load = sum(server_loads)

while total_load < num_tasks:
    idx = np.random.randint(0, num_servers)
    server_loads[idx] += 1
    total_load += 1

print("Initial server loads:", server_loads)

def fitness(loads):
    return np.max(loads) - np.min(loads)  

def update_loads(server_loads, neighborhood_size):
    new_loads = server_loads.copy()
    for i in range(len(server_loads)):
    
        neighbors = [
            server_loads[(i + offset) % len(server_loads)]
            for offset in range(-neighborhood_size, neighborhood_size + 1)
            if offset != 0
        ]
      
        avg_neighbor_load = sum(neighbors) // len(neighbors)
        
        if server_loads[i] > avg_neighbor_load:
            new_loads[i] -= 1
        elif server_loads[i] < avg_neighbor_load:
            new_loads[i] += 1
        
        new_loads[i] = max(0, new_loads[i])
    
    return new_loads

best_solution = server_loads
best_fitness = fitness(server_loads)

for iteration in range(max_iterations):
    server_loads = update_loads(server_loads, neighborhood_size)
    
    current_fitness = fitness(server_loads)
   
    if current_fitness < best_fitness:
        best_solution = server_loads
        best_fitness = current_fitness
    
    print(f"Iteration {iteration+1}: Server Loads = {server_loads}, Fitness = {current_fitness}")

print("\nFinal optimized server loads:", best_solution)
print("Final load imbalance (fitness):", best_fitness)

print("Avyukth Inna - 1BM22CS060")