from itertools import product

class MultipleKnapsackSolver:
    def __init__(self, values, weights, capacities):
        self.values = values
        self.weights = weights
        self.capacities = capacities
        self.num_items = len(values)
        self.num_knapsacks = len(capacities)
    
    def solve_knapsack_brute_force(self):
        best_value = 0
        best_combination = None
        
        for combination in product(range(self.num_knapsacks), repeat=self.num_items):
            knapsack_weights = [0] * self.num_knapsacks
            knapsack_values = [0] * self.num_knapsacks
            for item, knapsack in enumerate(combination):
                # knapsack_weights[knapsack] += self.weights[item] restrição
                knapsack_values[knapsack] += self.values[item]
            
            if all(knapsack_weight <= self.capacities[knapsack] for knapsack, knapsack_weight in enumerate(knapsack_weights)):
                total_value = sum(knapsack_values)
                if total_value > best_value:
                    best_value = total_value
                    best_combination = combination
        
        if best_combination is None:
            best_combination = []  # Retorna uma lista vazia se não houver combinação válida
        
        knapsacks = [[] for _ in range(self.num_knapsacks)]
        for item, knapsack in enumerate(best_combination):
            knapsacks[knapsack].append(item)
        
        return knapsacks, best_value


# train_values = [132, 94, 110, 190, 120, 175, 90, 80, 115, 160, 96, 76, 90, 160]
# train_weights = [2.6, 1.8, 2.0, 3.5, 2.3, 3.0, 1.6, 1.4, 2.2, 2.7, 1.8, 1.2, 1.7, 2.8]
# train_capacities = [10, 6] 

train_values = [ 1898, 440, 22507, 270, 14148, 3100,  4650, 30800, 615, 4975,
                1160, 4225, 510,  11880, 479, 440, 490, 330, 110, 560,
                24355, 2885, 11748, 4550, 750,  3720, 1950, 10500]

train_weights = [45,  0, 85, 150, 65, 95, 30,  0, 170, 0,
                40, 25, 20, 0, 0, 25,  0,  0, 25, 0,
                165,  0, 85	, 0, 0, 0, 0, 100, 	
                30, 20, 125, 5, 80, 25, 35, 73, 12,  15,
                15, 40, 5, 10, 10, 12, 10,  9,  0,  20,
                60, 40, 50, 36, 49, 40, 19, 150] 

train_capacities = [600, 600]

solver = MultipleKnapsackSolver(train_values, train_weights, train_capacities)
knapsacks, best_value = solver.solve_knapsack_brute_force()

print(f"Best value: {best_value}")
for i, knapsack in enumerate(knapsacks):
    print(f"Knapsack {i+1}: {knapsack}")
