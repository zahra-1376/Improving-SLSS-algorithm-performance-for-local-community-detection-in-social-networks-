# Apply patch for grakel compatibility with newer numpy
import patch_grakel

import Function
import datetime
import numpy as np
import multiprocessing
import networkx as nx
import math
import copy
import warnings
import os
import csv
from tqdm import tqdm
import time
warnings.simplefilter("ignore")

np.seterr(divide='ignore',invalid='ignore')

def graph_object_to_networkx(graph_obj):
    """Convert a graph object (either NetworkX or Grakel) to a NetworkX graph"""
    if hasattr(graph_obj, 'nodes') and hasattr(graph_obj, 'edges') and hasattr(graph_obj, 'degree'):
        # It's already a NetworkX graph
        return graph_obj
    elif hasattr(graph_obj, 'get_adjacency_matrix'):
        # It's a Grakel graph object - convert it to NetworkX
        adj_matrix = graph_obj.get_adjacency_matrix()
        nx_graph = nx.from_numpy_array(adj_matrix)
        return nx_graph
    elif hasattr(graph_obj, 'adj_matrix'):
        # It's a Grakel graph with adj_matrix attribute
        nx_graph = nx.from_numpy_array(graph_obj.adj_matrix)
        return nx_graph
    else:
        # Try to create NetworkX graph from adjacency matrix if possible
        try:
            # If it has a method to get adjacency matrix
            if hasattr(graph_obj, '__dict__'):
                # Look for adjacency matrix in the object
                for attr_name in ['adjacency_matrix', 'adj', 'A']:
                    if hasattr(graph_obj, attr_name):
                        adj = getattr(graph_obj, attr_name)
                        if hasattr(adj, 'toarray'):
                            adj = adj.toarray()
                        return nx.from_numpy_array(adj)
            # If the object itself can be treated as an adjacency matrix
            try:
                return nx.from_numpy_array(np.array(graph_obj))
            except:
                pass
        except:
            pass
        # Return an empty graph if conversion fails
        return nx.Graph()


def compute_genwl_graph_representation(graph, iterations=5):
    """Calculate GenWL (Graph Weisfeiler-Lehman) representation for a graph"""
    try:
        # Convert the input graph to a NetworkX graph
        nx_graph = graph_object_to_networkx(graph)
        
        if nx_graph.number_of_nodes() == 0:
            return np.array([])
        
        # Initialize labels for each node (using degree as initial labels)
        node_labels = {}
        for node in nx_graph.nodes():
            degree = nx_graph.degree[node]
            node_labels[node] = degree
        
        # Use a dictionary to map unique label structures to integers to avoid collisions
        label_to_id = {}
        next_label_id = 0
        
        # Perform WL iterations
        all_labels = []
        for iteration in range(iterations):
            new_labels = {}
            for node in nx_graph.nodes():
                # Get current label of the node
                current_label = node_labels[node]
                
                # Get labels of neighbors
                neighbor_labels = sorted([node_labels[neighbor] for neighbor in nx_graph.neighbors(node)])
                
                # Create new label by combining current label and sorted neighbor labels
                new_label_tuple = (current_label, tuple(neighbor_labels))
                
                # Map the label tuple to a unique integer ID to avoid hash collisions
                if new_label_tuple not in label_to_id:
                    label_to_id[new_label_tuple] = next_label_id
                    next_label_id += 1
                
                new_labels[node] = label_to_id[new_label_tuple]
            
            # Update node labels for next iteration
            node_labels = new_labels
            
            # Count label frequencies for this iteration
            label_counts = {}
            for label in node_labels.values():
                label_counts[label] = label_counts.get(label, 0) + 1
            # Convert label_counts values to list for this iteration
            iteration_labels = list(label_counts.values())
            all_labels.append(iteration_labels)
        
        # Flatten all_labels to maintain compatibility with existing code
        flattened_labels = []
        for iteration_labels in all_labels:
            flattened_labels.extend(iteration_labels)
        
        return np.array(flattened_labels)
    except Exception as e:
        # Return a default representation in case of error
        return np.array([0.0])

def compute_genwl_similarity(graph1, graph2, iterations=5):
    """Calculate similarity between two graphs based on GenWL representations"""
    try:
        # Compute GenWL representations for both graphs
        repr1 = compute_genwl_graph_representation(graph1, iterations)
        repr2 = compute_genwl_graph_representation(graph2, iterations)
        
        # Pad shorter representation with zeros if they have different lengths
        max_len = max(len(repr1), len(repr2))
        if len(repr1) < max_len:
            repr1 = np.pad(repr1, (0, max_len - len(repr1)), 'constant')
        if len(repr2) < max_len:
            repr2 = np.pad(repr2, (0, max_len - len(repr2)), 'constant')
        
        # Calculate the Euclidean distance between the representations
        distance = np.linalg.norm(repr1 - repr2)
        
        # Convert distance to similarity (higher when distance is lower)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    except Exception as e:
        # In case of errors, return 0
        return 0.0

def  M(nodei,G):
    " M method to obtain the community where the seed node nodei is located"
    C = []
    N = []
    C.append(nodei)
    e_out = G.degree(nodei)
    e_c = 0
    M = e_c / e_out
    for node in G.neighbors(nodei):
        N.append(node)
    while True:
        Mmax = -1000
        vBest = 0
        e_cBest = 0
        e_outBest = 0
        for nodev in N:
            dv = G.degree(nodev)
            x = 0
            for node_n in G.neighbors(nodev):
                if node_n in C:
                    x += 1
            y = dv - x
            e_cCurrent = e_c + x
            e_outCurrent = e_out - x + y
            Mnew = round((e_cCurrent / (e_outCurrent+1e-9)),4)
            if Mnew > Mmax:
                Mmax = Mnew
                vBest = nodev
                e_cBest = e_cCurrent
                e_outBest = e_outCurrent
        if Mmax >= M:
            M = Mmax
            e_c = e_cBest
            e_out = e_outBest
            C.append(vBest)
            N.remove(vBest)
            for n in G.neighbors(vBest):
                if n not in C:
                    N.append(n)
            N = list(set(N))
        else:
            break
    return C


def M2(nodei, G):
    "Select the known community corresponding to node i from a given set of known communities"
    C = []
    N = []

    for node in list(G.neighbors(nodei)):
        N.append(node)
    C.append(nodei)
    edge = G.edges(nodei)
    e_c = 0
    e_out = []
    for i in edge:
        weight = list(G.get_edge_data(list(i)[0], list(i)[1]).values())[0]
        e_out.append(weight)
    M = e_c / sum(e_out)
    while True:
        Mmax = -1000
        vbest = 0
        candidate_node = []
        if len(N) == 0:
            break
        for nodev in N:
            candidate_node.append(nodev)
            candidate_com = list(set(C) | {nodev})
            inedge, outedge = Function.cal_inner_outer_C(candidate_com, G)
            com_inedge = []
            for ii in inedge:
                com_inedge1 = list(G.get_edge_data(list(ii)[0], list(ii)[1]).values())[0]
                com_inedge.append(com_inedge1)
            com_outedge = []
            for jj in outedge:
                com_outedge1 = list(G.get_edge_data(list(jj)[0], list(jj)[1]).values())[0]
                com_outedge.append(com_outedge1)
            Mnew = sum(com_inedge) / (sum(com_outedge) + 1e-9)
            if Mnew > Mmax:
                Mmax = Mnew
                vbest = nodev
        if Mmax >= M:
            M = Mmax
            C.append(vbest)
            N.remove(vbest)
            for n in G.neighbors(vbest):
                if n not in C:
                    N.append(n)
            N = list(set(N))
        else:
            break
    del C[0]

    return C,M

def  first_stage(nodei,G):
    "SLSS method first stage, select nodes with more common neighbors and higher degree"
    C=[nodei]
    N=list(G.neighbors(nodei))
    comm_nei=Function.cal_common_nei(N, G)
    max_length=max(comm_nei)
    node_index= [i for i,x in enumerate(comm_nei) if x==max_length]
    second_node=Function.pick_max_degree(node_index, N, G)
    C.append(second_node)
    N=Function.cal_N(C, G)
    return C,N

def community_similarity(C1,N1,local_com,G,global_com,maxsize,quanzhong):
    "Community similarity phase, when all candidate community similarities are greater than the average historical community similarity, select the one with higher modularity"
    C=C1
    N=N1
    global_used_similarity=[]#Store global historical similarity values
    local_used_similarity=[]#Store local historical similarity values
    
    local_subgraph_weight = quanzhong
    while True:
        if len(N)==0:
            break
        if len(C)>=maxsize:#Check if the found community exceeds the maximum community size
            break
        Mmax=-1000000
        candidate_local_similarity1=[]#Store node similarity of candidate communities
        candidate_node=[]#Store candidate nodes
        candidate_global_similarity=[]#Store community similarity of candidate communities
        candidate_modularity=[]#Store modularity of candidate communities
        record_global_node=[]#Community similarity phase, record nodes selected from N
        for nodev in N:
            candidate_node.append(nodev)
            candidate_com = list(set(C) | {nodev})
            com_inedge,com_outedge=Function.calculate_inedges_outedges(candidate_com, G)#Calculate inner and outer edges of candidate community
            Mnew=round((com_inedge/(com_outedge+1e-9)),4)#Calculate modularity of candidate community
            candidate_modularity.append(Mnew)
            
            # Calculate GenWL-based similarity with known communities
            candidate_com_graph=Function.spgraph(candidate_com, G)
            
            # Compute similarities with all known communities
            candidate_similarities = []
            for idx, known_com_graph in enumerate(global_com):
                # Calculate similarity between candidate and known community
                try:
                    candidate_similarity = compute_genwl_similarity(candidate_com_graph, known_com_graph)
                    candidate_similarities.append(candidate_similarity)
                except:
                    candidate_similarities.append(0.0)
            
            # Handle case where there are no similarities calculated
            if len(candidate_similarities) == 0:
                Snew = 0.0
            else:
                Snew = round((sum(candidate_similarities) / len(candidate_similarities)),4)#Take the average similarity with known communities as the candidate community similarity
            candidate_global_similarity.append(Snew)
            #When candidate community similarities are all greater than the average historical community similarity, select the node with the highest modularity as the candidate node to add to community C
            if (len(global_used_similarity) == 0):
                if Snew >= np.nan_to_num(
                        np.mean(global_used_similarity)) and Mnew >= Mmax:  # Initially the community historical similarity list is empty, taking the mean of an empty list results in nan
                    # np.nan_to_num(nan) means 0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            elif (len(global_used_similarity) <= 2 and len(global_used_similarity) >= 1):
                if Snew >= min(global_used_similarity) and Mnew >= Mmax:  # Initially the community historical similarity list is empty, taking the mean of an empty list results in nan
                    # np.nan_to_num(nan) means 0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            else:
                if Snew >= min(global_used_similarity[-3:]) and Mnew >= Mmax:  # Initially the community historical similarity list is empty, taking the mean of an empty list results in nan
                    # np.nan_to_num(nan) means 0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
        if  len(record_global_node)!=0:
            node=record_global_node[0]
        else:
            break
        node1=node
        index_node1=candidate_node.index(node1)
        #In the community similarity phase, after calculating the similarity of candidate nodes, we also need to calculate the corresponding node similarity.
        if len(candidate_local_similarity1)==0:
            # The subgraph of node1 consists of three parts: nodev, the secondnode corresponding to the maximum common neighbors of nodev, and N(secondnode) ∩ C
            local_com1=[]
            first_node = [node1]
            pick_second_node = list(set(list(G.neighbors(node1))) & set(C))
            common_nei_length = Function.cal_common_nei(pick_second_node, G)
            second_node = pick_second_node[common_nei_length.index(max(common_nei_length))]
            third_nodes = list(set(list(G.neighbors(second_node))) & set(C))
            local_com1=list(set(first_node).union(third_nodes).union([second_node]))
            local_subgraph1 = Function.spgraph(local_com1, G)
            
            # Calculate GenWL-based similarity for the local subgraph
            local_similarities = []
            for idx, known_local_graph in enumerate(local_com):
                try:
                    local_similarity = compute_genwl_similarity(local_subgraph1, known_local_graph)
                    local_similarities.append(local_similarity)
                except:
                    local_similarities.append(0.0)
            
            weight_s1 = np.multiply(np.array(local_subgraph_weight), np.array(local_similarities)).tolist()
            # Handle case where there are no similarities calculated or weights sum to zero
            if len(local_subgraph_weight) == 0 or sum(local_subgraph_weight) == 0:
                Similarity1 = 0.0
            else:
                Similarity1 = round((sum(weight_s1) / sum(local_subgraph_weight)), 4)
            local_used_similarity.append(Similarity1)
            global_used_similarity.append(candidate_global_similarity[index_node1])
        else:
           break
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C,N,local_used_similarity

def node_similarity(C1,N1,local_com,G,global_com,maxsize,quanzhong,localsmi):
    "Node similarity phase, when node similarity is greater than historical node similarity, select the node with maximum node similarity"
    C=C1
    N=N1
    local_used_similarity=localsmi#Record the node similarity corresponding to each node added to community C
    
    local_subgraph_weight = quanzhong  # Weight corresponding to node subgraphs in known communities
    while True:
        if len(N)==0:
            break
        if len(C)>=maxsize:#Check if the found community exceeds the maximum community size
            break
        candidate_local_similarity1=[]
        candidate_node=[]
        Smax=-10000
        record_local_node = []  # Record nodes selected from N in the node similarity phase
        for nodev in N:
            candidate_node.append(nodev)
            
            local_com1 = []
            first_node = [nodev]
            pick_second_node = list(set(list(G.neighbors(nodev))) & set(C))
            common_nei_length = Function.cal_common_nei(pick_second_node, G)
            second_node = pick_second_node[common_nei_length.index(max(common_nei_length))]
            third_nodes = list(set(list(G.neighbors(second_node))) & set(C))
            local_com1.extend(first_node)
            local_com1.append(second_node)
            local_com1.extend(third_nodes)
            local_com1 = list(set(local_com1))
            #The subgraph of nodev consists of three parts: nodev, the secondnode with maximum common neighbors, and N(secondnode) ∩ C
            local_subgraph1 = Function.spgraph(local_com1, G)#Convert the node subgraph of candidate community to graph representation
            
            # Calculate GenWL representation for the local subgraph
            local_similarities = []
            for idx, known_local_graph in enumerate(local_com):
                try:
                    local_similarity = compute_genwl_similarity(local_subgraph1, known_local_graph)
                    local_similarities.append(local_similarity)
                except:
                    local_similarities.append(0.0)  # Default similarity in case of error
            
            weight_s1 = np.multiply(np.array(local_subgraph_weight),
                                    np.array(local_similarities)).tolist()#Calculate similarity between candidate community's node subgraph and known community's node subgraph, multiplied by the corresponding node subgraph weight
            
            # Handle case where there are no similarities calculated or weights sum to zero
            if len(local_subgraph_weight) == 0 or sum(local_subgraph_weight) == 0:
                Similarity1 = 0.0
            else:
                Similarity1 = round((sum(weight_s1) / sum(local_subgraph_weight)), 4)#Calculate the average node similarity of the candidate node subgraph
            candidate_local_similarity1.append(Similarity1)
            # When the node similarity corresponding to candidate node subgraphs are all greater than the average historical node similarity, select the node with maximum node similarity value as the node to add to community C
            if Similarity1 >= np.mean(local_used_similarity) and Similarity1>=Smax:
                Smax=Similarity1
                if len(record_local_node) != 0:
                    record_local_node[0] = nodev
                else:
                    record_local_node.append(nodev)
        if len(record_local_node) != 0:
            node = record_local_node[0]
        else:
            break
        node1 = node
        index_node1=candidate_node.index(node1)
        local_used_similarity.append(candidate_local_similarity1[index_node1])
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C,N




def select_com1(C, filename1111,comcom11,G):
    detected_C_graph = Function.spgraph(C, G)
    genwl_graphs = Function.knowknow(comcom11, filename1111)  # Get known community graphs
    
    # Calculate GenWL similarities between detected community and known communities
    similarities = []
    for known_graph in genwl_graphs:
        similarity = compute_genwl_similarity(detected_C_graph, known_graph)
        similarities.append(similarity)
    
    similarities = np.nan_to_num(similarities).tolist()
    comSim = {}  # by nili, replace a1 with comSim, key is community ID, Value is similarity
    for iq1 in range(len(similarities)):
        comSim[iq1] = similarities[iq1]
    comSort = sorted(comSim.items(), key=lambda x: x[1], reverse=True)
    coms_id=[list(yy)[0] for yy in comSort]
    return coms_id

def simi_num(com101,fileedge):
    sp_graph = Function.subgraphsubgraph(com101, fileedge)
    
    # Calculate GenWL-based pairwise similarities
    n = len(sp_graph)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity[i][j] = compute_genwl_similarity(sp_graph[i], sp_graph[j])
    
    similarity = np.nan_to_num(similarity)
    # Ensure diagonal elements are large enough so each graph is most similar to itself
    for i in range(n):
        if similarity[i][i] == 0:
            similarity[i][i] = 1.0  # Set self-similarity to 1.0
    
    return similarity

def si45(nv0,com101,fileedge):
    #print("com101",len(com101))
    si=simi_num(com101,fileedge) #Calculate similarity between communities in pairs
    si=np.nan_to_num(si)
    edge_sim={}
    #print("si",si)
    for i in range(len(si)):
        for j in range(i+1,len(si)):
            edge_sim[(i,j)]=si[i][j]
    know_sim=sorted(edge_sim.items(),key=lambda x:x[1],reverse=True)#Sort the similarity values
    modu = []
    comid=[]
    n=0#len(si)#int(len(know_sim)*0.1)
    g1=nx.Graph()
    #print("nv0",nv0)
    while n < len(know_sim):
        #print(list(know_sim[n][0])[0], list(know_sim[n][0])[1])
        g1.add_edge(list(know_sim[n][0])[0], list(know_sim[n][0])[1], weight=know_sim[n][1])
        #print('look',n, len(si))
        if nv0 in g1.nodes and n>=(len(si)-1):

            C,M =M2(nv0,g1)
            #print("M",M, len(si), len(g1.nodes))
            if M>1000: #1000 means the community modularity is infinite
                if (len(C))==(len(si)-1): #Indicates that the detected community contains all nodes in the similarity graph, modularity is set to -1
                    modu.append(-1)
                    comid.append(C)
                else:#Indicates that the detected community modularity is infinite, and the detected community nodes do not contain all nodes in the similarity graph
                    modu.append(1000)
                    comid.append(C)
                break
            else: #Indicates the community modularity is not infinite, add the community modularity and community to lists modu and comid
                modu.append(M)
                comid.append(C)
        n+=1
    # print("modu,", modu)
    # print("comid,", comid)
    index = modu.index(max(modu))#Select the community ID with maximum modularity
    #print("comid[index]",len(comid[index]))
    return comid[index]



def Go(para):
    i = para[0]
    filename, knowcomcom, file,  maxsize1= para[1], para[2], para[3],para[4]
    f = open(file + str(i), "w", encoding="utf-8")
    G = Function.networkx(filename)
        
    C11 = M(i, G)  # Obtain the community where the seed node is located using the M method
    gw=nx.Graph()
    gw.add_nodes_from(list(G.neighbors(i)))
    gw.add_edges_from(G.subgraph(G.neighbors(i)).edges())
    componentsNodes = max(nx.connected_components(gw), key=len)
    extraNodes = list(componentsNodes.union(C11))
    extraNodes.append(i)
    extraNodes=list(set(extraNodes))
    extrasubG = G.subgraph(extraNodes)  # g22 changed to extrasubG
    temp = copy.deepcopy(extraNodes)#Get the local structure where the given node is located
    extraNodes.remove(i)
    for ii in extraNodes:
        if extrasubG.degree(ii)==1 and ii not in list(extrasubG.neighbors(i)): #Remove nodes with degree 1 that are not neighbors of node i in the local structure
            temp.remove(ii)
    com10 = copy.deepcopy(knowcomcom) #Deep copy to prevent multiprocessing errors
    nv0=len(com10)
    com10.append(temp)
    comid=si45(nv0,com10,filename)#Obtain community ID via algorithm 5
    com1010=[]
    for uy in comid:
        com1010.append(com10[uy])
    graph, subgraph, quanzhong = Function.graph_subgraph_weight(com1010, filename)
    # Extract community subgraphs, node subgraphs, and weights of known communities based on comcom
    local_com = Function.subgraphsubgraph(subgraph, filename)
    global_com = Function.subgraphsubgraph(graph, filename)  # Convert to shortest path representation
    C1, N1 = first_stage(i, G)  # Add the node with the most common neighbors to the community
    C2, N2, ss = community_similarity(C1, N1, local_com, G, global_com, maxsize1, quanzhong)  # Community similarity phase
    C, N = node_similarity(C2, N2, local_com, G, global_com, maxsize1, quanzhong, ss)  # Node similarity phase
    f.writelines(str(C) + "\n")
    f.close()




if __name__ == '__main__':
    seed = Function.big_dataset_seed_top()  # Dataset seed nodes
    knowcom = ['dataset/amazon_knowcom', 'dataset/dblp_knowcom','dataset/facebook_knowcom',  'dataset/twitter_knowcom', 'dataset/lj_knowcom']
    fileedge = ['amazon-1.90.ungraph.txt', 'dblp-1.90.ungraph.txt','facebook-1.90.ungraph.txt', 'twitter-1.90.ungraph.txt', 'lj-1.90.ungraph.txt']
    path = ['amazon', 'dblp', 'facebook', 'twitter', 'lj']
    datasets = list(zip(seed, knowcom, fileedge, path))
    for filename in datasets: #datasets[:1] means run Amazon dataset, datasets means run all datasets
        if filename[3] != 'lj':
            continue
        print("Dataset: ", filename[3])
        dataset_knowcom = Function.get_knowcom(filename[1])
        for i in range(1,6):  # five groups of known communities
            print("Group", i, "-", len(dataset_knowcom[i - 1]), "known communities")
            
            # Start timing the group processing
            start_time = time.time()
            
            knowcomcom = dataset_knowcom[i - 1]
            maxsize = max([len(jjj) for jjj in knowcomcom])
            filename_edge = "dataset/" + filename[2]  # Path to edges corresponding to dataset
            nodelist = filename[0]  # Get seed nodes of the dataset
            savefile = "SLSS_GWL/" + filename[3] + "/" + str(i) + "/"  # Write the communities found for each node to files
            if os.path.isdir(os.path.dirname(savefile)) == False:  # Create directory
                os.makedirs(os.path.dirname(savefile))
            plist = [[node, filename_edge, knowcomcom, savefile, maxsize] for node in
                     nodelist]
            pool = multiprocessing.Pool(processes=2)  # Number of processes to start (CPU cores)
            # Use tqdm to show progress as tasks are completed
            for _ in tqdm(pool.imap(Go, plist), total=len(plist), desc=f"Processing nodes in group {i}", unit="node"):
                pass
            pool.close()
            pool.join()
            
            # Calculate timing metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            num_nodes = len(nodelist)
            nodes_per_second = num_nodes / elapsed_time if elapsed_time > 0 else 0
            
            # Print timing metrics
            print(f"Group {i} processing completed:")
            print(f"  Total time: {elapsed_time:.2f} seconds")
            print(f"  Total nodes processed: {num_nodes}")
            print(f"  Average: {nodes_per_second:.2f} nodes/second")
            
            # Write timing metrics to a CSV file
            timing_file = f"SLSS_GWL/{filename[3]}/timing_metrics.csv"
            os.makedirs(os.path.dirname(timing_file), exist_ok=True)
            
            # Check if file exists to write header only once
            file_exists = os.path.isfile(timing_file)
            
            with open(timing_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["dataset", "group", "total_nodes", "elapsed_time_sec", "nodes_per_sec"])
                writer.writerow([filename[3], i, num_nodes, f"{elapsed_time:.2f}", f"{nodes_per_second:.2f}"])
            
            print(f"  Timing metrics saved to: {timing_file}")