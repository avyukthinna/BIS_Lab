import random
import numpy as np

def objective_function(x):
    return x * np.sin(x)

population_size = 20   
generations = 100      
mutation_rate = 0.01    
crossover_rate = 0.7   
search_range = (-10, 10) 

def initialize_population():
    return [random.uniform(search_range[0], search_range[1]) for _ in range(population_size)]

def evaluate_fitness(population):
    return [objective_function(individual) for individual in population]

def select_parents(population, fitness):
    min_fitness = min(fitness)
    if min_fitness < 0:
        fitness = [f - min_fitness for f in fitness]  

    total_fitness = sum(fitness)
    selection_probs = [f / total_fitness for f in fitness]
   
    parents = np.random.choice(population, size=population_size, p=selection_probs)
    return parents


def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        alpha = random.random()
        return alpha * parent1 + (1 - alpha) * parent2
    return parent1 if random.random() < 0.5 else parent2

def mutate(individual):
    if random.random() < mutation_rate:
        mutation_value = random.uniform(-1, 1)
        individual = individual + mutation_value
        individual = max(min(individual, search_range[1]), search_range[0])
    return individual

def genetic_algorithm():
    population = initialize_population()

    for generation in range(generations):
        fitness = evaluate_fitness(population)
        parents = select_parents(population, fitness)
  
        next_generation = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
          
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            
            next_generation.extend([offspring1, offspring2])

        population = next_generation

        best_solution = max(population, key=objective_function)
        best_fitness = objective_function(best_solution)
        print(f"Generation {generation + 1}: Best Solution = {best_solution:.4f}, Best Fitness = {best_fitness:.4f}")
   
    final_solution = max(population, key=objective_function)
    print("\nBest Solution Found:")
    print(f"x = {final_solution:.4f}, f(x) = {objective_function(final_solution):.4f}")

genetic_algorithm()