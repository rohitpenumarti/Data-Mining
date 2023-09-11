import sys
from pyspark import SparkContext, StorageLevel
from collections import defaultdict
from itertools import combinations
from time import perf_counter

def read_dataset(sc, filepath):
    return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

def create_user_business_matrix(rdd):
    return rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)

def generate_graph(user_business_matrix, filter_threshold):
    graph_rdd = user_business_matrix.cartesian(user_business_matrix).filter(lambda x: True if x[0][0] != x[1][0] else False) \
        .map(lambda x: ((x[0][0], x[1][0]), x[0][1].intersection(x[1][1]))) \
            .filter(lambda x: True if len(x[1]) >= filter_threshold else False).map(lambda x: (x[0][0], x[0][1]))
    return graph_rdd

def bfs(root_node, graph):
    visited = {}
    queue = []
    shortest_dist = {key: 2**32-1 for key in graph.keys()}
    num_shorted_dists = {key: (0, 0, set()) for key in graph.keys()}

    queue.append(root_node)
    queue.append(None)
    visited[root_node] = True
    shortest_dist[root_node] = 0
    i = 0
    num_shorted_dists[root_node] = (1, i, set())
    i += 1
    while queue:
        cur = queue[0]
        if cur is None:
            i += 1
            queue.append(None)
            if queue[1] is None:
                break
            else:
                queue.pop(0)
                cur = queue[0]
                queue.pop(0)
        else:
            queue.pop(0)
        
        for n in graph[cur]:
            if n not in visited:
                visited[n] = True
                queue.append(n)
                num_shorted_dists[n] = (num_shorted_dists[n][0], i, num_shorted_dists[n][2])
            
            if shortest_dist[n] > shortest_dist[cur] + 1:
                shortest_dist[n] = shortest_dist[cur] + 1
                num_shorted_dists[n][2].add(cur)
                num_shorted_dists[n] = (num_shorted_dists[cur][0], num_shorted_dists[n][1], num_shorted_dists[n][2])
            elif shortest_dist[n] == shortest_dist[cur] + 1:
                num_shorted_dists[n][2].add(cur)
                num_shorted_dists[n] = (num_shorted_dists[n][0] + num_shorted_dists[cur][0], num_shorted_dists[n][1], num_shorted_dists[n][2])
    
    return num_shorted_dists

def credit_calculations(num_shortest_dists):
    sorted_shortest_distances = sorted(num_shortest_dists.items(), key=lambda x: -x[1][1])
    sorted_shortest_distance_map = dict(sorted_shortest_distances)
    node_val = defaultdict(lambda: 1)
    res = {}

    for node, vals in sorted_shortest_distances:
        total_parents_shortest_paths = sum([sorted_shortest_distance_map[parent][0] for parent in vals[2]])
        for parent in vals[2]:
            if sorted_shortest_distance_map[parent][1] < sorted_shortest_distance_map[node][1]:
                res[(node, parent)] = \
                    (node_val[node]*sorted_shortest_distance_map[parent][0])/total_parents_shortest_paths
                node_val[parent] += res[(node, parent)]
    
    return res

def compute_modularity(sub_graphs, graph, original_graph, m):
    outer_sum = 0
    for sub_graph_nodes in sub_graphs.values():
        inner_sum = 0
        for node1, node2 in combinations(sub_graph_nodes, 2):
            edges1 = graph[node1]
            ki = len(edges1)
            edges2 = graph[node2]
            kj = len(edges2)

            Aij = 0
            if node2 in original_graph[node1]:
                Aij = 1
            inner_sum += (Aij - ((ki*kj)/(2*m)))
        
        outer_sum += inner_sum
    return outer_sum/(2*m)

def get_natural_communities(root_node, graph):
    remaining_nodes = set(graph.keys())
    visited = {}
    queue = []
    communities = defaultdict(list)

    queue.append(root_node)
    visited[root_node] = True
    comm_num = 0
    communities[comm_num].append(root_node)
    next_node = root_node
    prev_num = -1
    while len(remaining_nodes) > 0:
        while queue:
            cur = queue[0]
            queue.pop(0)
            if prev_num != comm_num:
                remaining_nodes.remove(next_node)
                prev_num = comm_num
            
            for n in graph[cur]:
                if n not in visited:
                    visited[n] = True
                    queue.append(n)
                    remaining_nodes.remove(n)
                    communities[comm_num].append(n)
        
        if remaining_nodes:
            comm_num += 1
            next_node = list(remaining_nodes)[0]
            queue.append(next_node)
            visited[next_node] = True
            communities[comm_num].append(next_node)

    return dict(communities)

def remove_edge(edges, node):
    s = set([])
    s.add(node)
    return edges.difference(s)

def determine_optimal_communities(graph, betweenness_vals):
    graph_map = graph.collectAsMap()
    m = sum([len(v) for k, v in graph_map.items()])/2
    root_node = list(graph_map.keys())[0]
    curr_communities = get_natural_communities(root_node, graph_map)
    
    Q_new = compute_modularity(curr_communities, graph_map, graph_map, m)
    Q_curr = -1e10
    
    max_betweenness_edge = betweenness_vals[0][0]
    new_graph_rdd = graph
    final_communities = curr_communities

    while Q_new > Q_curr:
        Q_curr = Q_new
        new_graph_map = new_graph_rdd.collectAsMap()
        new_graph_map[max_betweenness_edge[0]] = remove_edge(new_graph_map[max_betweenness_edge[0]], max_betweenness_edge[1])
        new_graph_map[max_betweenness_edge[1]] = remove_edge(new_graph_map[max_betweenness_edge[1]], max_betweenness_edge[0])
        new_graph_rdd = sc.parallelize(list(new_graph_map.items()))

        root_node_new = list(new_graph_map.keys())[0]
        new_communities = get_natural_communities(root_node_new, new_graph_map)
        Q_new = compute_modularity(new_communities, new_graph_map, graph_map, m)

        shortest_distances_rdd = new_graph_rdd.map(lambda x: (x[0], bfs(x[0], new_graph_map)))
        shortest_distances_map = shortest_distances_rdd.collectAsMap()
        credit_map = dict(map(lambda x: (x[0], credit_calculations(shortest_distances_map[x[0]])), shortest_distances_map.items()))
        credit_rdd = sc.parallelize(list(credit_map.items()))
        betweenness_rdd = credit_rdd.flatMap(lambda x: x[1].items()).filter(lambda x: True if x[1] != 0 else False) \
            .map(lambda x: (tuple(sorted(x[0])), x[1])).reduceByKey(lambda a, b: a+b).mapValues(lambda x: x/2)
        sorted_betweenness = betweenness_rdd.sortBy(lambda x: [-x[1], x[0][0]]).collect()

        if sorted_betweenness:
            max_betweenness_edge = sorted_betweenness[0][0]
        
        if Q_curr < Q_new:
            final_communities = new_communities

    return final_communities

if __name__ == "__main__":
    start = perf_counter()
    filter_threshold = int(sys.argv[1])
    input_file_name = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('ERROR')

    rdd = read_dataset(sc, input_file_name)
    user_business_matrix = create_user_business_matrix(rdd)
    edge_rdd = generate_graph(user_business_matrix, filter_threshold)
    graph_rdd = edge_rdd.groupByKey().mapValues(set)
    graph_map = graph_rdd.collectAsMap()

    ##### Part 1 #####

    shortest_distances_rdd = graph_rdd.map(lambda x: (x[0], bfs(x[0], graph_map)))
    shortest_distances_map = shortest_distances_rdd.collectAsMap()
    credit_map = dict(map(lambda x: (x[0], credit_calculations(shortest_distances_map[x[0]])), shortest_distances_map.items()))
    credit_rdd = sc.parallelize(list(credit_map.items()))
    betweenness_rdd = credit_rdd.flatMap(lambda x: x[1].items()).filter(lambda x: True if x[1] != 0 else False) \
        .map(lambda x: (tuple(sorted(x[0])), x[1])).reduceByKey(lambda a, b: a+b).mapValues(lambda x: x/2)
    sorted_betweenness = betweenness_rdd.sortBy(lambda x: [-x[1], x[0][0]]).collect()
    
    with open(betweenness_output_file_path, 'w') as file:
        for i, edge in enumerate(sorted_betweenness):
            if i == len(sorted_betweenness)-1:
                file.write(f'{edge[0]},{edge[1]}')
            else:
                file.write(f'{edge[0]},{round(edge[1], 5)}\n')

    ##### Part 2 #####

    final_communities = determine_optimal_communities(graph_rdd, sorted_betweenness)
    communities_sorted = {k: sorted(v) for k, v in final_communities.items()}
    sorted_final_communities = sorted(communities_sorted.items(), key=lambda x: [len(x[1]), x[1][0]])

    with open(community_output_file_path, 'w') as file:
        for i, community in enumerate(sorted_final_communities):
            for j, user in enumerate(community[1]):
                if j == len(community[1])-1:
                    if i == len(sorted_final_communities)-1:
                        file.write(f'\'{user}\'')
                    else:
                        file.write(f'\'{user}\'\n')
                else:
                    file.write(f'\'{user}\', ')

    end = perf_counter()
    print(f'Time elapsed: {end-start}')