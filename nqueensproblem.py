from collections import deque
import time
import random
import numpy as np
from typing import List

# N-Queens Solver using Depth-First Search
class NQueensDFS:
    def __init__(self, n):
        self.n = n
        self.solutions = []

    def solve(self):
        self.dfs([])
        return self.solutions

    def dfs(self, current_solution):
        row = len(current_solution)
        if row == self.n:
            self.solutions.append(current_solution)
            return
        for col in range(self.n):
            if not self.is_conflict(current_solution, row, col):
                self.dfs(current_solution + [col])

    def is_conflict(self, current_solution, row, col):
        for r in range(row):
            c = current_solution[r]
            if c == col or abs(c - col) == abs(r - row):
                return True
        return False

    def find_solution(self):
        self.solve()
        return self.solutions[0] if self.solutions else None


# Exhaustive DFS algorithm 
# Function to check if placing a queen at (row, col) conflicts with existing queens
def is_conflict(current_solution, row, col):
    for r in range(row):
        c = current_solution[r]
        # Check if the column or diagonals are attacked
        if c == col or abs(c - col) == abs(r - row):
            return True
    return False

# N-Queens Solver using Breadth-First Search
def solve_n_queens_bfs(N):
    solutions = []
    queue = deque([([], 0)])  # Queue stores pairs (partial solution, current row)

    while queue:
        current_solution, row = queue.popleft()

        if row == N:
            solutions.append(current_solution)
        else:
            for col in range(N):
                if not is_conflict(current_solution, row, col):
                    queue.append((current_solution + [col], row + 1))
    return solutions

#n is 10
N = 50

start_time_bfs = time.time()
solutions_bfs = solve_n_queens_bfs(N)
time_bfs = time.time() - start_time_bfs

# Results
num_solutions_bfs = len(solutions_bfs)

print(f"Time taken (BFS): {time_bfs} seconds")
print(f"Number of solutions (BFS): {num_solutions_bfs}")


# Genetic Algorithm

class NQueensGA:
    def __init__(self, n_queens, population_size=100, mutation_rate=0.1):
        self.n_queens = n_queens
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        return [random.sample(range(self.n_queens), self.n_queens) 
                for _ in range(self.population_size)]

    def fitness(self, chromosome):
        conflicts = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if chromosome[i] == chromosome[j]:  # Same row
                    conflicts += 1
                offset = j - i
                if chromosome[i] == chromosome[j] - offset or chromosome[i] == chromosome[j] + offset:  # Diagonal
                    conflicts += 1
        return 1 / (conflicts + 1)

    def select_parent(self, total_fitness):
        r = random.uniform(0, total_fitness)
        current = 0
        for chromosome in self.population:
            current += self.fitness(chromosome)
            if current > r:
                return chromosome
        return self.population[-1]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.n_queens - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            pos1, pos2 = random.sample(range(self.n_queens), 2)
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome

    def evolve(self, generations=1000):
        for _ in range(generations):
            new_population = []
            total_fitness = sum(self.fitness(chrom) for chrom in self.population)
            
            while len(new_population) < self.population_size:
                parent1 = self.select_parent(total_fitness)
                parent2 = self.select_parent(total_fitness)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            best_solution = max(self.population, key=self.fitness)
            if self.fitness(best_solution) == 1.0:
                return best_solution
        
        return max(self.population, key=self.fitness)

    def display_solution(self, solution):
        board = np.zeros((self.n_queens, self.n_queens))
        for i, pos in enumerate(solution):
            board[pos][i] = 1
        print("\nSolution found:")
        print(board)
        return board

# Run the algorithm
n_queens = 50
ga = NQueensGA(n_queens)
solution = ga.evolve()
ga.display_solution(solution)
print("\nQueen positions (row indices):", solution)
print("Fitness:", ga.fitness(solution))



#Comparison of the two algorithms

# Comparison Function
def compare_algorithms(n: int, num_runs: int = 5):
    print(f"\nComparing algorithms for {n}-Queens problem:")
    
    # DFS timing
    dfs_times = []
    for _ in range(num_runs):
        dfs = NQueensDFS(n)
        start_time = time.time()
        solution = dfs.find_solution()
        end_time = time.time()
        dfs_times.append(end_time - start_time)
    
    # GA timing
    ga_times = []
    for _ in range(num_runs):
        ga = NQueensGA(n)
        start_time = time.time()
        solution = ga.evolve(generations=1000)
        end_time = time.time()
        ga_times.append(end_time - start_time)
    
    print(f"\nResults for n={n}:")
    print(f"DFS Average Time: {sum(dfs_times)/len(dfs_times):.4f} seconds")
    print(f"GA Average Time: {sum(ga_times)/len(ga_times):.4f} seconds")

# Run comparison
n_values = [50]
for n in n_values:
    compare_algorithms(n)