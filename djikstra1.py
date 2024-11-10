from pyspark.sql import SparkSession
import sys

# Initialize Spark session
spark = SparkSession.builder.appName("DijkstraShortestPath").master("local[*]").getOrCreate()
sc = spark.sparkContext

# Define a large number to represent infinity
INFINITY = float('inf')

# Load graph edges from both files
edges1 = sc.textFile("question2_1.txt")
edges2 = sc.textFile("question2_2.txt")

# Parse each line into (source, destination, weight)
edges = edges1.union(edges2).map(lambda line: line.split()).map(lambda parts: (parts[0], parts[1], float(parts[2])))

# Specify the starting node
start_node = 'A'  # Replace 'A' with the desired starting node

# Initialize distances RDD with start node distance 0, others as infinity
distances = edges.flatMap(lambda x: [(x[0], INFINITY), (x[1], INFINITY)]) \
                 .distinct() \
                 .map(lambda x: (x[0], 0 if x[0] == start_node else INFINITY))

# Initialize edges RDD for graph representation
graph = edges.map(lambda x: (x[0], (x[1], x[2])))

# Iteratively update distances using Dijkstra's algorithm
def dijkstra_update(distances, graph):
    updated_distances = distances.join(graph) \
        .flatMap(lambda x: [(x[0], x[1][0]), (x[1][1][0], x[1][0] + x[1][1][1])]) \
        .reduceByKey(lambda a, b: min(a, b))
    return updated_distances

for _ in range(10):  # Limit iterations for convergence; adjust as necessary
    distances = dijkstra_update(distances, graph)

# Collect and save results to output_2.txt
shortest_paths = distances.collect()
with open("output_3.txt", "w") as f:
    for node, dist in shortest_paths:
        f.write(f"{node}: {dist}\n")

# Find nodes with the greatest and least distance
max_node = max(shortest_paths, key=lambda x: x[1])
min_node = min(shortest_paths, key=lambda x: x[1] if x[1] > 0 else sys.maxsize)

print(f"Node with greatest distance: {max_node}")
print(f"Node with least distance: {min_node}")

# Stop Spark session

