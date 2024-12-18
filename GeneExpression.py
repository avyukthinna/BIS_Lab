import numpy as np

class GeneExpressionClustering:
    def __init__(self, num_genes, num_clusters, num_generations, population_size, mutation_rate):
        self.num_genes = num_genes
        self.num_clusters = num_clusters
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def initialize_population(self, data):
        population = []
        for _ in range(self.population_size):
            genes = np.random.rand(self.num_genes, data.shape[1]) * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)
            population.append(genes)
        return np.array(population)

    def evaluate_fitness(self, data, population):
        fitness_scores = []
        for individual in population:
            labels = self.assign_clusters(data, individual)
            intra_cluster_distance = self.calculate_intra_cluster_distance(data, labels, individual)
            fitness = 1 / (1 + intra_cluster_distance)  # Minimize intra-cluster distance
            fitness_scores.append(fitness)
        return np.array(fitness_scores)

    def assign_clusters(self, data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_intra_cluster_distance(self, data, labels, centroids):
        total_distance = 0
        for i in range(len(centroids)):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                total_distance += np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1))
        return total_distance

    def mutate(self, individual):
        for gene in range(individual.shape[0]):
            if np.random.rand() < self.mutation_rate:
                individual[gene] += np.random.normal(0, 0.1, size=individual[gene].shape)
        return individual

    def evolve(self, data):
        population = self.initialize_population(data)
        for generation in range(self.num_generations):
            fitness_scores = self.evaluate_fitness(data, population)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = population[sorted_indices]
            fittest_individual = population[0]
            print(f"Generation {generation + 1}: Best Fitness = {fitness_scores[sorted_indices[0]]:.4f}")
            
            next_population = population[: self.population_size // 2]

            while len(next_population) < self.population_size:
                parent_indices = np.random.choice(len(next_population), size=2, replace=False)
                parent1, parent2 = next_population[parent_indices[0]], next_population[parent_indices[1]]
                child = (parent1 + parent2) / 2
                next_population = np.vstack([next_population, child[np.newaxis]])
                next_population = np.array([self.mutate(ind) for ind in next_population])

            population = next_population

        return fittest_individual

if __name__ == "__main__":   
    np.random.seed(42)
    data = np.random.rand(100, 2) * 10  

    num_clusters = 3
    num_genes = num_clusters
    num_generations = 20
    population_size = 10
    mutation_rate = 0.1

    gep = GeneExpressionClustering(num_genes, num_clusters, num_generations, population_size, mutation_rate)
    best_centroids = gep.evolve(data)
    final_labels = gep.assign_clusters(data, best_centroids)
    print("Best centroids:", best_centroids)
    print("Avyukth Inna - 1BM22CS060")
