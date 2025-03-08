import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from itertools import combinations

# Load categorized warehouse data
df = pd.read_csv("categorized_warehouse_data.csv")

# Load frequent itemsets from Apriori results
rules = pd.read_csv("frequent_itemsets.csv")

# Define warehouse storage structure
aisles = 6
columns = 5
racks_per_column = 5
total_racks = aisles * columns * racks_per_column  # 6 * 5 * 5 = 150 racks
rack_capacity = 100  # Each rack can hold 100 volume units

# Prepare items (product_name, profit, volume, category)
items = [(row['product_name'], row['profit'], row['volume'], row['category']) for _, row in df.iterrows()]

# Create storage racks
racks = [{"id": f"A{a+1}-C{c+1}-R{r+1}", "capacity": rack_capacity, "used": 0} 
         for a in range(aisles) for c in range(columns) for r in range(racks_per_column)]

# Define aisle, column, rack priorities (lower number = higher priority)
rack_priority = {
    "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6,  # High Demand → Low Demand
    "C2": 1, "C3": 2, "C4": 3, "C1": 4, "C5": 5,  # Middle Columns are best
    "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5   # Lower Racks are best
}

def rack_score(rack_id):
    """Calculate accessibility score for each rack."""
    aisle, column, rack = rack_id.split('-')
    return rack_priority[aisle] + rack_priority[column] + rack_priority[rack]

# **🔹 Step 1: Create a mapping for frequently bought-together products**
frequent_pairs = {}
for _, row in rules.iterrows():
    pair = tuple(sorted(list(eval(row['antecedents'])) + list(eval(row['consequents']))))  
    frequent_pairs[pair] = row['support']  # Store frequency score

def get_adjacent_rack(rack_id):
    """Find an adjacent rack for frequently bought-together items."""
    aisle, column, rack = rack_id.split('-')
    rack_number = int(rack.replace("R", ""))  # Remove 'R' and convert to integer
    
    if rack_number < 5:
        adj_rack = f"{aisle}-{column}-R{rack_number + 1}"  # Move up
    else:
        adj_rack = f"{aisle}-{column}-R{rack_number - 1}"  # Move down
    
    return adj_rack if adj_rack in [r["id"] for r in racks] else rack_id  # Ensure rack exists

# Define Genetic Algorithm (GA) Optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize profit
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_rack", np.random.randint, 0, total_racks)  # Assign to a random rack
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rack, n=len(items))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalStorage(individual):
    """Evaluate storage allocation based on profit, accessibility, and product relationships."""
    total_profit = 0
    rack_usage = {rack["id"]: 0 for rack in racks}  # Track used space per rack
    product_rack_map = {}

    # Assign products to racks
    for (product, profit, volume, category), rack_index in zip(items, individual):
        rack_id = racks[rack_index]["id"]

        if rack_usage[rack_id] + volume <= rack_capacity:  # Check space availability
            rack_usage[rack_id] += volume
            total_profit += profit * (1 / rack_score(rack_id))  # Prioritize accessible racks
            product_rack_map[product] = rack_id  # Store placement

    # **🔹 Step 2: Apply Frequent Itemset Rules**
    for (prod1, prod2), support in frequent_pairs.items():
        if prod1 in product_rack_map and prod2 in product_rack_map:
            rack1 = product_rack_map[prod1]
            rack2 = product_rack_map[prod2]

            # If products are not stored near each other, adjust
            if rack1 != rack2 and rack2 != get_adjacent_rack(rack1):
                adjusted_rack = get_adjacent_rack(rack1)
                if rack_usage[adjusted_rack] + df.loc[df['product_name'] == prod2, 'volume'].values[0] <= rack_capacity:
                    product_rack_map[prod2] = adjusted_rack  # Move product closer
                    total_profit += support * 100  # Reward keeping pairs together

    return (total_profit,)

toolbox.register("evaluate", evalStorage)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=total_racks - 1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run Genetic Algorithm
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

# Get best storage allocation solution
best_solution = tools.selBest(population, 1)[0]

# Assign optimized storage locations
df['allocated_rack'] = [racks[idx]["id"] for idx in best_solution]
df.to_csv("optimized_storage_allocation.csv", index=False)

print("Storage Optimization Completed! Rack assignments saved in 'optimized_storage_allocation.csv'.")
