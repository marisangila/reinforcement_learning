import numpy as np
from gurobipy import Model, GRB, quicksum

class MultipleKnapsackRL:
    def __init__(self, values, weights, capacities):
        self.values = values
        self.weights = weights
        self.capacities = capacities
        self.num_items = len(values)
        self.num_knapsacks = len(capacities)

    def solve_knapsack_exact(self, values, weights, capacities):
     
        model = Model('MultipleKnapsack')
        
        x = {}
        for i in range(self.num_items):
            for k in range(self.num_knapsacks):
                x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x[{i},{k}]')
        

        model.setObjective(quicksum(values[i] * x[i, k] for i in range(self.num_items) for k in range(self.num_knapsacks)), GRB.MAXIMIZE)
        
        for k in range(self.num_knapsacks):
            model.addConstr(quicksum(weights[i] * x[i, k] for i in range(self.num_items)) <= capacities[k], f'capacity_{k}')
        
        for i in range(self.num_items):
            model.addConstr(quicksum(x[i, k] for k in range(self.num_knapsacks)) == 1, f'item_placement_{i}')

        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            total_value = model.objVal
            knapsacks = [[] for _ in range(self.num_knapsacks)]
            for i in range(self.num_items):
                for k in range(self.num_knapsacks):
                    if x[i, k].x == 1:
                        knapsacks[k].append(i)
            return total_value, knapsacks
        else:
            raise RuntimeError('Gurobi optimization did not converge to an optimal solution.')

    def train(self):
        train_values = self.values
        train_weights = self.weights
        train_capacities = self.capacities

        train_max_value, train_best_combination = self.solve_knapsack_exact(train_values, train_weights, train_capacities)

        print(f"Maximum value: {train_max_value}")
        for i, knapsack in enumerate(train_best_combination):
            print(f"Best combination for knapsack {i + 1}: {knapsack}")

        return train_max_value, train_best_combination


if __name__ == '__main__':

    # train_values = [132, 94, 110, 190, 120, 175, 90, 80, 115, 160, 96, 76, 90, 160]
    # train_weights = [2.6, 1.8, 2.0, 3.5, 2.3, 3.0, 1.6, 1.4, 2.2, 2.7, 1.8, 1.2, 1.7, 2.8]
    # train_capacities = [10, 6] 
    
    # train_values = [135, 105, 176, 158, 167, 140]
    # train_weights = [13, 12, 17, 16, 15, 6]
    # train_capacities = [60] 
    
    train_values = [ 1898, 440, 22507, 270, 14148, 3100,  4650, 30800, 615, 4975,
                1160, 4225, 510,  11880, 479, 440, 490, 330, 110, 560,
                24355, 2885, 11748, 4550, 750,  3720, 1950, 10500]

    train_weights = [45,  0, 85, 150, 65, 95, 30,  0, 170, 0,
                40, 25, 20, 0, 0, 25,  0,  0, 25, 0,
                165,  0, 85	, 0, 0, 0, 0, 100, 	
                30, 20, 125, 5, 80, 25, 35, 73, 12,  15,
                15, 40, 5, 10, 10, 12, 10,  9,  0,  20,
                60, 40, 50, 36, 49, 40, 19, 150] 

    train_capacities = [500, 500]
    
    # train_values = [1,1,1,1,1,1,1,1,1,1]
    # train_weights = [2,2,4,4,8,8,16,16,32,32]
    # train_capacities = [62,62] 

    knapsack_rl = MultipleKnapsackRL(train_values, train_weights, train_capacities)
    train_max_value, train_best_combination = knapsack_rl.train()
