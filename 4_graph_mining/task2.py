import itertools
from queue import Queue
from operator import add
import sys
import time
import random
from pyspark import SparkConf, SparkContext
# to build this graph by bfs, we need to define a node/vertex class to use it to represet vertices
class Vertex:
    def __init__(self, index):
        self.index = index
        self.level = 0
        self.parent = set()
        self.credit = 0
        self.betweenness = 1
    def addparent(self, parentID):
        self.parent.add(parentID)
    def changelv(self, newlv):
        self.level = newlv
    def changecredit(self, newcd):
        self.credit = self.credit+newcd
    def changebet(self, newbet):
        self.betweenness = self.betweenness+newbet

def bfs(x):
    visited_vertecies = set()
    queue = Queue()
    root = Vertex(x)
    visited_vertecies.add(x)
    queue.put(root)
    vertices = {}
    vertices[x] = root
    while queue.empty() != True:
        pointer = queue.get()
        for v in edges_dict[pointer.index]:
            children = Vertex(v)
            if v not in visited_vertecies:
                visited_vertecies.add(children.index)
                queue.put(children)
                vertices[children.index] = children
    return visited_vertecies

def calculateBetweeness(x):
    visited_vertecies = set()
    queue = Queue()
    root = Vertex(x)
    visited_vertecies.add(x)
    queue.put(root)
    root.changecredit(1)

    vertices = {}
    vertices[x] = root

    while queue.empty() != True:
        pointer = queue.get()
        for v in edges_dict[pointer.index]:
            children = Vertex(v)
            if v in visited_vertecies:
                children = vertices[v]
                if children.level > pointer.level:
                    children.addparent(pointer.index)
                    children.changecredit(pointer.credit)
            else:
                visited_vertecies.add(children.index)
                children.addparent(pointer.index)
                children.changelv(pointer.level+1)
                children.changecredit(pointer.credit)
                queue.put(children)
                vertices[children.index] = children

    result = []
    hierarchical_dict = {}
    for v in vertices.values():
        if v.level in hierarchical_dict:
            hierarchical_dict[v.level].append(v.index)            
        else: # add a new level
            hierarchical_dict[v.level] = [v.index]

    height = max(hierarchical_dict.keys())
    for i in range(height, 0, -1):
        for k in hierarchical_dict[i]:
            v = vertices[k]
            parent_credit = []
            parent_credit_sum = 0
            for parent in list(v.parent):
                parent_credit.append(vertices[parent].credit)
                parent_credit_sum += vertices[parent].credit
            weight_list = []
            for p in parent_credit:
                weight_list.append(p/parent_credit_sum)
            for m in range(len(weight_list)):
                newbet = float(v.betweenness*weight_list[m])
                vertices[list(v.parent)[m]].changebet(newbet)
                result.append((tuple(sorted((v.index, list(v.parent)[m]))), newbet))
    return result     

if __name__ == '__main__':
    start_time = time.time()
    # define input variables
    filter_threshold = "7"
    input_file_path = "data/ub_sample_data.csv"
    betweenness_file_path = "task2_bet.txt"
    output_file_path = "task2_com.txt"
    # filter_threshold = sys.argv[1]
    # input_file_path = sys.argv[2]
    # betweenness_file_path = sys.argv[3]
    # output_file_path = sys.argv[4]
    configuration = SparkConf().set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("WARN")
    input_lines = sc.textFile(input_file_path).map(lambda x : x.split(',')).map(lambda x:(x[0], x[1])).filter(lambda x: x[0]!= "user_id").groupByKey().mapValues(lambda x: list(x))
    ub_dict = input_lines.collectAsMap()
   
    edges = []
    points = set()
    for x in list(itertools.combinations(ub_dict.keys(), 2)):
        if len(set(ub_dict[x[0]]).intersection(set(ub_dict[x[1]]))) >= int(filter_threshold):
            edges.append(x)
            edges.append((x[1],x[0]))
            points.add(x[0])
            points.add(x[1])
    # create a tree using edges
    points = sc.parallelize(sorted(list(points))).collect()
    points_dict = {}
    for i in range(len(points)):
        points_dict[points[i]] = i
    points_dict2 = {}
    for i in range(len(points)):
        points_dict2[i] = points[i]
    edges= sc.parallelize(edges).groupByKey().mapValues(lambda x: sorted(list(set(x)))).map(lambda x: (points_dict[x[0]], [points_dict[y] for y in x[1]]))
    edges_dict = {}
    for item in edges.collect():
        edges_dict[item[0]] = item[1]
    betweenness_temp = edges.flatMap(lambda x: calculateBetweeness(x[0]))
    betweenness = betweenness_temp.reduceByKey(add).mapValues(lambda x: x/2).sortBy(lambda x: (-x[1], x[0]))
    betweenness1 = betweenness.map(lambda x: ((points_dict2[x[0][0]], points_dict2[x[0][1]]), x[1]))  
    #output 2.1
    result = betweenness1.collect()
    with open(betweenness_file_path, 'w+') as output_file:
        for line in result:
            output_file.writelines(str(line)[1:-1] + "\n")
        output_file.close()

    # first, we use betweeness to find all clusters
    def clusters(current_graph):
        visited = set()
        clusters = {}
        idx = 0
        candidate_vertices = list(current_graph.keys())
        for v in range(len(candidate_vertices)):
            if candidate_vertices[v] not in visited:
                possible_cluster = bfs(candidate_vertices[v])
                clusters[idx] = possible_cluster
                idx += 1
                visited = visited.union(set(possible_cluster))
        return clusters
    # calculate modularity for removing the highest ranked edge
    m = betweenness.count()
    A = set(betweenness.keys().collect())
    K = betweenness.keys().flatMap(lambda x: ((x[0],1),(x[1],1))).reduceByKey(add).sortByKey().collectAsMap()
    high_ranked_edge = betweenness.sortBy(lambda x: -x[1]).map(lambda x: (x[0][0], x[0][1])).first()
    edges_dict[high_ranked_edge[0]].remove(high_ranked_edge[1])
    edges_dict[high_ranked_edge[1]].remove(high_ranked_edge[0])
    def modularity(current_clusters, m, A, K):
        result = 0
        for c in current_clusters.keys():
            community = list(current_clusters[c])
            for i in range(len(community)-1):
                for j in range(i+1, len(community)):
                    candidate_adj_vertex = tuple(sorted((community[i], community[j])))
                    if candidate_adj_vertex in A:
                        a_ij = 1
                    else:
                        a_ij = 0
                    result += a_ij - (K[community[i]]*K[community[j]])/(2*m)
        return result
    Q = modularity(clusters(edges_dict), m, A, K)
    # keep removing edges and find all communities
    threshold_q = Q
    while Q >= threshold_q:
        betweenness2 = sc.parallelize(edges_dict.items()).flatMap(lambda x: calculateBetweeness(x[0])).reduceByKey(add).mapValues(lambda x: x/2).sortBy(lambda x: (-x[1], x[0]))
        high_ranked_edge = betweenness2.map(lambda x: (x[0][0], x[0][1])).first()
        edges_dict[high_ranked_edge[0]].remove(high_ranked_edge[1])
        edges_dict[high_ranked_edge[1]].remove(high_ranked_edge[0])
        Q = modularity(clusters(edges_dict), m, A, K)
        if Q >= threshold_q:
            threshold_q = Q
        else:
            if high_ranked_edge[0] in edges_dict:
                edges_dict[high_ranked_edge[0]].append(high_ranked_edge[1])            
            else: # add a new level
                edges_dict[high_ranked_edge[0]] = high_ranked_edge[1]
            if high_ranked_edge[1] in edges_dict:
                edges_dict[high_ranked_edge[1]].append(high_ranked_edge[0])            
            else: # add a new level
                edges_dict[high_ranked_edge[1]] = high_ranked_edge[0]
            break      
    #output 2.2
    result = edges_dict.items()
    while edges_dict:
        vertices = list(edges_dict.keys())
        visited_vertecies = set()
        visited_vertecies.add(vertices[0])
        community = [vertices[0]]
        for i in range(len(community)):
            for v in edges_dict[community[i]]:
                if v not in visited_vertecies:
                    visited_vertecies.add(v)
                    community.append(v)
            del edges_dict[community[i]]
        result.append(sorted(community))
    result = sc.parallelize(result).map(lambda x: [points_dict2[y] for y in x[1]]).collect()
    with open(output_file_path, 'w+') as output_file:
        for line in result:
            output_file.writelines(str(line)[1:-1] + "\n")
        output_file.close()
    print('Duration:', (time.time()-start_time))