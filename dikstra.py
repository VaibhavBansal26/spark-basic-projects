from pyspark import SparkContext, SparkConf
import findspark
findspark.init()
from pyspark.sql.functions import explode, split, col, desc
from pyspark.sql import SparkSession

# Initialize SparkContext
spark = SparkSession.builder.appName("Djikstra").master("local[*]").getOrCreate()
sc = spark.sparkContext


# Read the data files
edges_rdd = sc.textFile('question2_1.txt').union(sc.textFile('question2_2.txt'))

# Parse the edges into (source_node, destination_node, weight)
edges = edges_rdd.map(lambda line: line.strip().split(',')) \
                 .map(lambda parts: (int(parts[0]), int(parts[1]), float(parts[2])))

# Create an adjacency list RDD: (node, list of (neighbor, weight))
adjacency_list = edges.map(lambda x: (x[0], (x[1], x[2]))) \
                      .groupByKey() \
                      .mapValues(list) \
                      .cache()

# Get all nodes
nodes_from = edges.map(lambda x: x[0])
nodes_to = edges.map(lambda x: x[1])
all_nodes = nodes_from.union(nodes_to).distinct().cache()

# Initialize distances: (node, distance)
start_node = 0  # Assuming the first node is 0
infinity = float('inf')
distances = all_nodes.map(lambda node: (node, infinity))
distances = distances.map(lambda x: (x[0], 0.0) if x[0] == start_node else x)
distances = distances.cache()

# Iterative update of distances
updated = True
iteration = 0
max_iterations = all_nodes.count() - 1  # Maximum possible iterations

while updated and iteration < max_iterations:
    iteration += 1
    # Join distances with adjacency list
    joined = distances.join(adjacency_list, numPartitions=8)
    
    # Compute tentative distances
    tentative_distances = joined.flatMap(lambda x: [ 
        (neighbor[0], x[1][0] + neighbor[1]) for neighbor in x[1][1]
    ])
    
    # Combine the new distances with the existing ones
    new_distances = distances.union(tentative_distances) \
                             .reduceByKey(lambda x, y: min(x, y))
    
    # Check if distances have changed
    changes = new_distances.join(distances).filter(lambda x: x[1][0] != x[1][1])
    updated = not changes.isEmpty()
    
    # Update distances for the next iteration
    distances = new_distances
    distances = distances.cache()

# Collect final distances
final_distances = distances.collectAsMap()

# Write distances to output file
with open('output_2.txt', 'w') as f:
    for node in sorted(final_distances.keys()):
        dist = final_distances[node]
        if dist == infinity:
            f.write(f"{node} unreachable\n")
        else:
            f.write(f"{node} {dist}\n")

# Find nodes with greatest and least distances (excluding infinity and the start node)
reachable_nodes = {node: dist for node, dist in final_distances.items() if dist != infinity and node != start_node}
if reachable_nodes:
    # Find the maximum and minimum distances
    max_distance = max(reachable_nodes.values())
    min_distance = min(reachable_nodes.values())
    
    # Find all nodes that have the maximum and minimum distances
    max_nodes = [node for node, dist in reachable_nodes.items() if dist == max_distance]
    min_nodes = [node for node, dist in reachable_nodes.items() if dist == min_distance]
    
    # Print nodes with greatest distance
    print(f"Nodes with greatest distance from {start_node} (Distance: {max_distance}): {max_nodes}")
    
    # Print nodes with least distance
    print(f"Nodes with least distance from {start_node} (Distance: {min_distance}): {min_nodes}")
else:
    print("No reachable nodes from the starting node.")


