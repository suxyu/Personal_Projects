import math
"""
author:xiaoyu sun
student id:32950683
note:
at the very last minte, i noticed that adj matrix and dfs could not meet the time complexity requirement. 
I should use adj list and bfs with cloest node(edmonds karp method)
however, due to the time constraint, i am not able to modify the code on time.

I only had time to modify the code using bfs instead of dfs, the time to finish the same amount of tests decrease from 2.4s to 0.077s, showing
that the modification indeed improve the speed

ideally, i should change from adj matrix to adj list as well, however, due to the time constrint, i am not able to do it on time

If i used adj list and bfs with cloest node(edmonds karp method), i should get the time complexity of O(D*C^2) for Q1
"""


#############   Q1   ####################
def bfs(graph, start, end, parent):
    """
    Function description: use breadth first search to find augmenting path

    :Input:
    argv1:graph, the input graph
    argv2:start, start node
    argv3:end, end node
    argv4:parent, a list containing the parent node of each node
    :Output
    argv:True if augmenting path is found, False if not
    :Time complexity: O(V^2), V is the total vertex number of graph g, if adj list is used, it could down to O(V+E)
    :Space complexity: O(V62), to store graph
    :Aux space complexity:O(c)
    """

    visited = [False] * len(graph)
    pri_queue = []
    pri_queue.append(start)
    visited[start] = True

    while pri_queue:#worst case takes O(V)
        current = pri_queue.pop(0)
        if current == end:# if current is end,
            return True

        for adj in range(len(graph)): # takes O(V)

            if visited[adj] == False and graph[current][adj] > 0:#if unvisited and there is still augmenting flow
                visited[adj] = True
                pri_queue.append(adj)

                parent[adj] = current # keep track of the parent node of the current node

                if adj == end:
                    return True

    return False


def ford_fulkerson_bfs(graph, source, sink):
    """
    Function description: use ford fulkerson to find out max flow

    :Input:
    argv1:graph, the input graph
    argv2:source, start node
    argv3:sink, end node

    :Output
    argv:max, the maximum flow of the network
    :Time complexity: O(EV^4), V is the total vertex number of graph g, if adj list is used, optimally it will take O(EV^2)
    :Space complexity: O(V^2), to store the graph
    :Aux space complexity:O(V^2) , residual graph
    """
    residual = [graph[i][:] for i in range(len(graph))]# initialize the residual graph same as original graph
    max = 0  # Initialize the maximum flow to 0


    parent = [-1] * len(graph) #initialize a list to keep track of parent node
    while bfs(residual, source, sink, parent):#bfs takes O(V^2), the while loop at most takes O(VE) according to the lecture node
        flow = math.inf
        child = sink
        while child != source:#take O(V)
            flow = min(flow, residual[parent[child]][child]) # find the bottleneck of the augmenting path
            child = parent[child] #update child node to its parent node


        child = sink
        while child != source:#take O(V)
            node = parent[child] #find the parent node of
            residual[node][child] =residual[node][child] - flow #update the residual graph
            residual[child][node] =residual[child][node] + flow #update the residual graph
            child = parent[child]#update the child node to its parent and repeat the proess


        # update the max flow
        max =max + flow

    return max

""" 
                           The following code was orginally developed using dfs


def DFS(g,u,t,bottleneck,visited):
    
    Function description: use depth first search to find augmenting path

    :Input:
    argv1:g, the input graph
    argv2:u, start node
    argv3:t, end node
    argv4:bottleneck, initialized at infinity, it represents the maximum flow each augmenting path could have
    argv5:visited, an list representing if each node has been visited. visited: True. unvisited: False
    :Output
    argv:augment, augment flow
    :Time complexity: O(n^2), n is the total node number of graph g
    :Aux space complexity:O(c)
    
    if u==t:# if go to the end of the graph
        return bottleneck
    visited[u]=True#set current node to be visited

    for i in range(len(g)):
        residual = g[u][i][1]-g[u][i][0] #residual is the capacity - the flow
        if residual >0 and not visited[i]: # if residual >0 and unvisited
            augment = DFS(g,i,t,min(bottleneck,residual),visited)
            if augment>0:# found the augmenting path, update flow
                g[u][i][0] +=augment
                g[i][u][0] -=augment
                return augment
    return 0 # if could not find augmenting path



def ford_fulkerson(graph,source,sink):
    
     Function description: use ford_fulkerson to find the maximum flow

     :Input:
     argv1:graph, the input graph
     argv2:source, start node
     argv3:sink, end node

     :Output
     argv1:max_flow, the maximum flow of the network
     argv2:residual_graph, the residual graph
     :Time complexity: O(n^2*|F|), n is total node number of the network and |F| is the maximum flow
     :Aux space complexity:O(n^2), n is total node number of the network
     
    residual_graph = [graph[i][:] for i in range(len(graph))] #copy the graph


    max_flow = 0#set max flow to 0

    #parent = [-1]*len(graph)

    while True:#while there is still augmenting path
        visited = [False for _ in range(len(graph))]#set unvisited for all nodes
        augment = DFS(residual_graph, source, sink, math.inf, visited)#find augmenting path
        max_flow = max_flow + augment#update the max flow
        if augment <=0:#break if there is no more augment flow
            break

    return max_flow,residual_graph


"""



class Graph:

    def __init(self):
        """
         Function description: initialize the graph


         :Time complexity: O(c)

         :Aux space complexity:O(c),self.graph
         """
        self.graph = []

    def construct_empty_adj_matrix(self, vertex_number):
        """
         Function description: construct empty adj matrix, each cell contains 0

         :Input:
         argv1:vertex_number, the number of vertex for the graph

         :Time complexity: O(n^2), n is total node number of the network

         :Aux space complexity:O(0),
         """
        self.graph = [[0 for _ in range(vertex_number)] for _ in range(vertex_number)]
        return self.graph

    def add_vertex(self):
        """
         Function description: add one empty row and column to the adj matrix

        output:
        argv1:self.graph, modified adj matrix graph

         :Time complexity: O(n),

         :Aux space complexity:O(n), n is total node number of the network, temp value
         """
        for row in self.graph:#add 0 to the end of each row
            row.append(0)

        temp = []#temp list to store the added row at the end
        for item in range(len(self.graph)):
            temp.append(0)

        self.graph.append(temp)

    def add_edge(self,fr,to,capacity):
        """
               Function description: add edge to the adj matrix

               Input:
               argv1:fr, the node edge is from
               argv2:to, the node edge is to
               argv3:capacity, the capacity of this edge
               output:
               argv1:self.graph, modified adj matrix graph



               :Time complexity: O(1),
               :Aux space complexity:O(0)
        """
        self.graph[fr][to]=capacity

# this block of code is for displaying the graph visually
#    def print_graph(self):
#        for i in range(len(self.graph)):
#            for j in range(len(self.graph)):
#                if self.graph[i][j] != [0,0]:
#                    print(i, " -> ", j, \
#                          " edge weight: ", self.graph[i][j])




def maxThroughput(connections,maxIn,maxOut,origin,targets):
    """
    Function description:
    Approach description :add one node before(A) and after(B) each data center node,
    the capacity from A to data center is the corresponding maxIn value, the capacity
    from data center to B is the corresponding maxOut value.
    add a super sink at the end, connect each data center node to the super sink,
    the capacity is the corresponding maxIn.
    Therefore, the total node would be 3*number_of_data_center +1
    and finally, apply ford-fulkerson to get the total max throughput

    In addition, in the resultant adjusted node, 3*i is the input node for data center i,
    3*i+1 is the data center, 3*i+2 is the output node for data center i.

    3*num_of_data_center is the super sink node

    :Input:
    argv1:connections, the connection list stated in the problem
    argv2:maxIn, the maxIn list stated in the problem
    argv3:maxOut, the maxOut list stated in the problem
    argv4:origin: the origin of the network
    argv5:targets: the list containing the sink for the network
    :Output, return or postcondition:
    argv1:max, the maximum flow which could flow to targets
    :Time complexity:O(D^4 * C), D is the number of data center, C is the number of connections. If adj list and bfs is
    used for ford fulkerson, it will achieve the desired complexity.
    Although after modification to the graph, the node vertax and edge number changes, but it differs with a constant multipler,
    so the overall complexity stills the same

    :Aux space complexity:O(D^2), network valuable
    """
    num_of_data_centre = len(maxIn)
    networkflow = Graph()

    ### construct an empty adj matrix for the adjusted resultant graph, all value in the matrix is 0, used to represent capacity later
    flowdiagram = networkflow.construct_empty_adj_matrix(num_of_data_centre*3+1)#the total node number of the adjusted graph is 3*num_of_data_center +1

    ##add edge from the added output node for  each "from" data center to the added input node of each "to" data center
    for item in connections:
        networkflow.add_edge(3*item[0]+2,3*item[1],item[2])#3*item[0]+2 is the added afterward node of 'from node', 3*item[0] is the added input node of 'to node'


    #add edge from the added input node to the data center , the capacity is the corresponding maxIn value
    for count,item in enumerate(maxIn):
        networkflow.add_edge(3*count,3*count+1,item)#3*count is the added input node for data center, 3*count+1 is the data center node

    # add edge from each data center to its added output node , the capacity is the corresponding maxOut value
    for count,item in enumerate(maxOut):
        networkflow.add_edge(3*count+1,3*count+2,item)#3*count +1 is the data center node, 3*count+2 is the added afterward output node for each data center

    #add a super sink connecting all nodes in targets, the coresponding capacity is the maxIn for each target data center node
    for item in targets:
        networkflow.add_edge(3*item+1,3*num_of_data_centre,maxIn[item])#3*item+1 is the data cemter node, 3*num_of_data_center is the super sink node

    #print(networkflow.graph)



    max= ford_fulkerson_bfs(networkflow.graph, 3*origin+1,3*num_of_data_centre) # although after modification to the graph, the node and edge changes, but it differs
    #by a constant multipler, so the complexity stay the same
    return max


""""                    test case for q1
import unittest

class TestA2(unittest.TestCase):
    def test_341490(self):
        connections = [(32, 77, 363), (13, 77, 452), (74, 82, 602), (51, 1, 867), (77, 38, 879), (17, 61, 776),
                       (5, 22, 410), (9, 7, 331), (16, 30, 829), (43, 13, 843), (44, 66, 513), (65, 85, 633),
                       (18, 82, 705), (26, 15, 469), (91, 30, 550), (62, 29, 110), (17, 0, 619), (1, 35, 124),
                       (69, 64, 945), (47, 16, 319), (44, 54, 523), (42, 50, 833), (65, 26, 403), (7, 44, 609),
                       (26, 31, 550), (11, 52, 570), (81, 55, 350), (36, 59, 93), (31, 63, 254), (27, 12, 93),
                       (31, 82, 749), (86, 7, 259), (43, 25, 883), (12, 59, 434), (52, 2, 339), (70, 32, 290),
                       (59, 71, 906), (32, 70, 270), (59, 11, 690), (47, 38, 617), (63, 18, 555), (59, 39, 291),
                       (0, 5, 106), (41, 23, 501), (46, 85, 166), (64, 14, 118), (1, 58, 902), (16, 41, 607),
                       (87, 93, 792), (23, 20, 800), (61, 55, 693), (35, 76, 198), (43, 45, 809), (38, 63, 110),
                       (60, 38, 753), (72, 45, 350), (80, 49, 557), (93, 30, 994), (46, 73, 722), (60, 51, 314),
                       (9, 22, 610), (57, 92, 462), (33, 26, 849), (2, 61, 580), (73, 59, 556), (34, 77, 365),
                       (51, 74, 467), (6, 57, 852), (25, 92, 94), (19, 37, 392), (4, 70, 944), (23, 13, 970),
                       (73, 2, 534), (53, 89, 515), (93, 52, 81), (93, 61, 547), (83, 23, 745), (6, 26, 946),
                       (2, 1, 735), (91, 93, 768), (79, 9, 753), (48, 25, 971), (64, 57, 947), (0, 23, 570),
                       (32, 10, 940), (13, 58, 246), (82, 86, 219), (5, 76, 847), (80, 1, 701), (53, 26, 580),
                       (61, 45, 894), (46, 15, 230), (42, 67, 419), (24, 26, 172), (1, 16, 909), (43, 81, 772),
                       (30, 66, 446), (66, 25, 206), (12, 83, 660), (68, 89, 786), (93, 10, 904), (93, 2, 573),
                       (27, 41, 208), (50, 60, 624), (40, 69, 902), (43, 73, 194), (58, 10, 740), (2, 53, 866),
                       (7, 46, 871), (64, 41, 230), (79, 61, 902), (20, 14, 533), (17, 80, 845), (29, 5, 455),
                       (47, 86, 483), (64, 31, 869), (10, 25, 991), (6, 0, 486), (38, 11, 829), (71, 6, 541),
                       (40, 58, 962), (88, 64, 651), (72, 55, 110), (22, 55, 119), (11, 22, 403), (39, 61, 491),
                       (84, 89, 437), (28, 7, 532), (29, 12, 478), (38, 49, 849), (19, 31, 757), (80, 23, 391),
                       (49, 20, 643), (3, 30, 704), (61, 7, 542), (55, 93, 757), (78, 76, 707), (64, 6, 241),
                       (86, 58, 379), (79, 86, 341), (80, 47, 229), (24, 3, 214), (20, 66, 292), (27, 10, 637),
                       (35, 49, 179), (59, 54, 891), (59, 14, 199), (14, 51, 709), (46, 61, 777), (68, 27, 677),
                       (80, 41, 560), (11, 30, 763), (2, 85, 847), (37, 22, 596), (73, 46, 246), (72, 82, 379),
                       (90, 10, 959), (86, 57, 748), (41, 5, 773), (25, 40, 296), (41, 92, 875), (24, 32, 871),
                       (43, 55, 112), (0, 30, 422), (36, 43, 484), (20, 3, 622), (45, 82, 797), (92, 48, 252),
                       (12, 8, 257), (82, 87, 763), (54, 9, 514), (5, 41, 342), (18, 73, 765), (17, 54, 790),
                       (36, 50, 77), (39, 0, 710), (85, 0, 913), (15, 79, 518), (8, 22, 781), (93, 26, 966),
                       (45, 66, 910), (67, 14, 875), (46, 41, 86), (54, 57, 621), (89, 3, 206), (75, 1, 986),
                       (54, 81, 441), (84, 58, 974), (92, 30, 548), (93, 82, 955), (75, 54, 174), (43, 61, 841),
                       (28, 13, 993), (23, 67, 590), (66, 33, 440), (32, 48, 479), (21, 87, 300), (22, 51, 475),
                       (93, 39, 157), (86, 72, 378), (14, 18, 273), (64, 27, 742), (29, 88, 308), (83, 25, 294),
                       (29, 22, 436), (54, 49, 573), (82, 39, 370), (27, 45, 353), (10, 17, 744), (63, 24, 843),
                       (43, 7, 680), (24, 72, 993), (84, 30, 377), (69, 45, 440), (32, 2, 585), (87, 1, 570),
                       (22, 74, 253), (21, 64, 346), (77, 50, 437), (45, 79, 923), (85, 62, 166), (86, 60, 516),
                       (38, 3, 847), (29, 24, 234), (3, 55, 381), (76, 89, 814), (85, 45, 943), (77, 78, 526),
                       (28, 68, 409), (13, 63, 565), (52, 29, 262), (43, 18, 371), (52, 23, 246), (56, 14, 703),
                       (45, 44, 337), (62, 41, 745), (78, 74, 638), (13, 76, 839), (74, 8, 285), (89, 30, 993),
                       (12, 46, 848), (39, 92, 790), (35, 58, 647), (15, 31, 374), (78, 5, 840), (23, 90, 825),
                       (66, 93, 123), (38, 67, 577), (10, 15, 715), (18, 81, 732), (42, 47, 769), (15, 33, 374),
                       (65, 88, 895), (62, 68, 463), (82, 89, 268), (28, 86, 529), (65, 63, 765), (69, 68, 611),
                       (7, 92, 471)]
        maxIn = [297, 1357, 1146, 516, 189, 1040, 92, 1123, 1369, 572, 532, 486, 571, 397, 540, 561, 1479, 294, 484,
                 772, 652, 672, 792, 240, 1437, 1135, 1251, 925, 1027, 1302, 723, 1417, 125, 256, 292, 941, 775, 564,
                 601, 59, 718, 1058, 369, 552, 564, 199, 1188, 979, 544, 1095, 1366, 672, 595, 535, 1304, 832, 889,
                 1309, 583, 117, 922, 156, 309, 620, 989, 1480, 447, 901, 1394, 395, 164, 273, 444, 404, 874, 1113, 302,
                 810, 1031, 1025, 552, 506, 518, 916, 1374, 1239, 470, 1021, 472, 1289, 1307, 467, 52, 1275]
        maxOut = [1145, 1437, 133, 1075, 521, 538, 1335, 1361, 481, 904, 801, 178, 1315, 293, 1474, 287, 704, 1300,
                  1445, 216, 350, 137, 363, 1076, 777, 1119, 812, 528, 633, 364, 1251, 1370, 1209, 197, 1377, 352, 1303,
                  748, 1023, 795, 617, 1453, 1331, 538, 1498, 202, 645, 647, 595, 434, 152, 464, 808, 956, 503, 1105,
                  731, 615, 213, 821, 294, 304, 413, 396, 880, 607, 615, 890, 773, 1316, 1479, 1245, 326, 941, 1019,
                  1403, 823, 671, 578, 710, 504, 436, 780, 932, 945, 110, 494, 157, 1293, 1051, 130, 509, 988, 1086]
        origin = 41
        targets = [78, 33, 34, 72, 43]
        # my res: 830
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 830)

    def test_480663(self):
        connections = [(5, 20, 794), (73, 75, 292), (62, 39, 884), (11, 9, 881), (66, 65, 517), (73, 38, 520),
                       (62, 37, 749), (68, 39, 561), (42, 56, 847), (18, 59, 987), (21, 5, 547), (49, 77, 691),
                       (73, 47, 843), (1, 74, 298), (67, 19, 290), (60, 59, 225), (7, 52, 849), (35, 41, 894),
                       (28, 5, 286), (27, 9, 113), (41, 44, 265), (75, 40, 714), (29, 67, 95), (10, 31, 480),
                       (34, 35, 635), (63, 50, 315), (59, 9, 876), (75, 10, 938), (60, 34, 451), (29, 46, 630),
                       (37, 39, 599), (22, 40, 220), (41, 45, 833), (21, 76, 865), (49, 69, 954), (6, 12, 515),
                       (24, 63, 828), (58, 10, 169), (32, 7, 537), (28, 75, 304), (65, 58, 354), (71, 9, 942),
                       (51, 58, 660), (34, 66, 945), (37, 46, 268), (16, 1, 555), (51, 15, 271), (61, 40, 658),
                       (31, 18, 617), (74, 36, 806), (61, 42, 146), (55, 20, 300), (13, 79, 538), (14, 57, 222),
                       (43, 36, 189), (46, 68, 353), (44, 5, 180), (32, 56, 404), (27, 12, 945), (70, 60, 234),
                       (26, 17, 802), (21, 30, 171), (40, 39, 264), (13, 25, 487), (43, 42, 375), (78, 21, 511),
                       (9, 21, 659), (34, 37, 218), (70, 13, 741), (69, 4, 436), (19, 54, 410), (72, 9, 294),
                       (9, 71, 626), (45, 17, 643), (40, 24, 583), (78, 20, 668), (48, 53, 843), (0, 36, 545),
                       (65, 6, 751), (50, 11, 482), (58, 44, 312), (59, 73, 125), (36, 13, 367), (78, 6, 95),
                       (67, 10, 515), (76, 41, 725), (31, 38, 407), (22, 54, 500), (7, 22, 987), (71, 49, 357),
                       (63, 27, 447), (42, 9, 633), (36, 17, 193), (68, 20, 469), (34, 6, 791), (35, 70, 798),
                       (31, 50, 399), (67, 3, 677), (44, 35, 646), (43, 15, 536), (79, 56, 521), (72, 51, 131),
                       (52, 27, 676), (16, 15, 872), (10, 18, 970), (36, 60, 804), (36, 25, 270), (16, 50, 646),
                       (57, 63, 490), (3, 67, 863), (2, 3, 785), (63, 62, 416), (73, 18, 811), (76, 65, 794),
                       (33, 8, 260), (46, 59, 750), (60, 22, 561), (21, 61, 876), (12, 66, 584), (44, 27, 912),
                       (16, 47, 668), (75, 1, 973), (14, 51, 944), (7, 36, 990), (17, 40, 832), (11, 4, 186),
                       (20, 13, 384), (45, 73, 191), (41, 61, 614), (8, 26, 301), (27, 6, 291), (55, 69, 432),
                       (46, 23, 210), (63, 3, 286), (41, 49, 850), (38, 1, 331), (55, 77, 607), (69, 41, 772),
                       (18, 27, 570), (41, 13, 364), (15, 39, 431), (67, 29, 562), (28, 31, 866), (32, 49, 818),
                       (69, 78, 797), (40, 0, 854), (68, 61, 101), (63, 29, 115), (76, 53, 888), (19, 58, 446),
                       (2, 79, 583), (72, 47, 341), (59, 76, 871), (22, 4, 450), (47, 64, 422), (23, 62, 455),
                       (72, 75, 241), (12, 49, 246), (15, 36, 382), (56, 71, 376), (18, 55, 533), (70, 28, 438),
                       (13, 76, 634), (79, 9, 873), (7, 39, 685), (1, 72, 720), (25, 66, 637), (56, 28, 826),
                       (48, 38, 764), (54, 29, 397), (64, 35, 930), (56, 29, 983), (77, 26, 798), (13, 30, 243),
                       (4, 33, 663), (65, 24, 541), (79, 48, 711), (78, 14, 866), (34, 27, 989), (26, 41, 781),
                       (1, 57, 984), (18, 0, 837), (78, 38, 796), (78, 16, 183), (47, 25, 505), (39, 29, 613),
                       (28, 61, 987), (24, 16, 211), (1, 52, 1000), (31, 5, 716), (47, 8, 958), (14, 16, 467),
                       (66, 67, 758), (15, 29, 921), (73, 49, 806), (30, 55, 949), (78, 69, 859), (23, 77, 514),
                       (15, 21, 367), (22, 58, 327), (64, 50, 840), (5, 55, 976), (75, 64, 912), (57, 65, 253),
                       (1, 4, 385), (22, 25, 351), (54, 53, 758), (40, 78, 416), (1, 6, 959), (11, 22, 678),
                       (74, 67, 675), (3, 70, 664), (79, 23, 576), (26, 22, 378), (32, 2, 77), (56, 50, 707),
                       (4, 58, 635), (24, 39, 128), (30, 6, 174), (53, 28, 304), (45, 33, 293), (50, 36, 693),
                       (30, 36, 785), (27, 44, 923), (76, 73, 420), (19, 57, 471), (15, 31, 111), (53, 65, 942),
                       (17, 48, 913), (15, 16, 559)]
        maxIn = [468, 974, 1230, 145, 1419, 177, 1334, 161, 1159, 1282, 1286, 390, 1431, 140, 1423, 924, 1452, 1217, 86,
                 346, 1125, 160, 585, 174, 905, 972, 983, 181, 175, 272, 663, 703, 1390, 956, 777, 843, 635, 732, 448,
                 1131, 282, 366, 1192, 419, 411, 1496, 326, 1463, 1059, 1446, 530, 155, 244, 1459, 771, 1183, 547, 611,
                 285, 52, 522, 1238, 1193, 1395, 84, 1199, 785, 1140, 679, 561, 779, 977, 1346, 1354, 95, 1287, 173,
                 399, 727, 871]
        maxOut = [893, 1351, 1042, 897, 709, 527, 217, 1495, 121, 1391, 840, 561, 153, 844, 337, 660, 270, 1042, 948,
                  1312, 1027, 627, 1277, 653, 1024, 139, 1051, 82, 226, 1085, 924, 1227, 434, 302, 923, 868, 806, 477,
                  680, 1035, 1212, 1250, 954, 578, 215, 200, 77, 1144, 915, 910, 132, 244, 121, 454, 720, 1098, 419, 55,
                  963, 1212, 371, 1170, 958, 774, 1461, 329, 736, 978, 1402, 785, 125, 838, 241, 881, 761, 1321, 242,
                  1455, 812, 390]
        origin = 22
        targets = [63, 12, 44]
        # my res: 1265
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 1265)

    def test(self):
        connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),
                       (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
        maxIn = [5000, 3000, 3000, 2000, 2000]
        maxOut = [5000, 3000, 3000, 2000, 1500]
        origin = 0
        targets = [4, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 4000)

    def test1(self):
        connections = [(0, 1, 10), (1, 2, 5), (1, 3, 7), (3, 4, 10)]
        maxIn = [math.inf, 10, 5, 7, math.inf]
        maxOut = [math.inf, 10, 5, 10, math.inf]
        origin = 0
        targets = [2, 4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets),10)


    def test2(self):
        connections = [(0,1,20), (1,3,30), (1,4,5), (0,2,10), (2, 4, 15)]
        maxIn = [math.inf, math.inf, math.inf, math.inf, math.inf]
        maxOut = [math.inf, math.inf, math.inf, math.inf, math.inf]
        origin = 0
        targets = [3,4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 30)

    def test3(self):
        connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),
        (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
        maxIn = [5000, 3000, 3000, 3000, 2000]
        maxOut = [5000, 3000, 3000, 2500, 1500]
        origin = 3
        targets = [4, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 2500)

    def test4(self):
        connections = [(0,1,30), (1,3,30), (1,4,30), (0,2,30), (2, 4,30)]
        maxIn = [5, 5, 5, 5, 5]
        maxOut = [1000, 1000, 1000, 1000, 1000]
        origin = 0
        targets = [3,4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 10)

    def test5(self):
        connections = [(0,1,30), (1,3,30), (1,4,30), (0,2,30), (2, 4,30)]
        maxIn = [1000, 1000, 1000, 1000, 1000]
        maxOut = [100, 5, 5, 5, 5]
        origin = 0
        targets = [3,4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 10)


    def test6(self):
        connections = [(0,1,30), (1,3,30), (1,4,30), (0,2,30), (2, 4,30)]
        maxIn = [1000, 1000, 1000, 1000, 1000]
        maxOut = [1, 1000, 1000, 1000, 1000]
        origin = 0
        targets = [3,4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 1)

    def test7(self):
        connections = [(0,1,30), (1,3,30), (1,4,30), (0,2,30), (2, 4,30)]
        maxIn = [1000, 1000, 1000, 1000, 1000]
        maxOut = [1000, 1000, 1000, 1000, 1000]
        origin = 0
        targets = [1,2,3,4]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 60)


    def test9(self):
        connections = [(0, 1, 3000), (0, 2, 2000)]
        maxIn = [5000, 0, 0]
        maxOut = [5000, 0, 0]
        origin = 0
        targets = [1, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 0)


    def test10(self):
        connections = [(0, 1, 1000), (0, 2, 2000), (2, 1, 3000)]
        maxIn = [5000, 4000, 3000]
        maxOut = [5000, 4000, 3000]
        origin = 0
        targets = [1]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 3000)


    def test11(self):
        connections = [(0, 1, 1000), (0, 2, 2000), (0, 3, 3000)]
        maxIn = [5000, 1000, 2000, 3000]
        maxOut = [10000, 1000, 2000, 3000]
        origin = 0
        targets = [1, 2, 3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 6000)

    def test12(self):
        connections = [(0, 1, 3000), (0, 2, 2000)]
        maxIn = [0, 5000, 5000]
        maxOut = [0, 5000, 5000]
        origin = 0
        targets = [1, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 0)

    def test14(self):
        connections = [(0, 1, 2000), (0, 2, 2000)]
        maxIn = [5000, 2000, 2000]
        maxOut = [5000, 2000, 2000]
        origin = 0
        targets = [1, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 4000)

    def test15(self):
        connections = [(0, 1, 3000)]
        maxIn = [5000, 3000]
        maxOut = [5000, 3000]
        origin = 0
        targets = [1]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 3000)

    def test16(self):
        connections = [(0, 1, 0), (0, 2, 0)]
        maxIn = [5000, 5000, 5000]
        maxOut = [5000, 5000, 5000]
        origin = 0
        targets = [1, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 0)

    def test17(self):
        connections = [(0, 1, 1000), (0, 2, 1000), (0, 3, 1000)]
        maxIn = [5000, 1000, 1000, 1000]
        maxOut = [5000, 1000, 1000, 1000]
        origin = 0
        targets = [1, 2, 3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 3000)

    def test_provided(self):
    
        connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),
                       (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
        maxIn = [5000, 3000, 3000, 3000, 2000]
        maxOut = [5000, 3000, 3000, 2500, 1500]
        origin = 0
        targets = [4, 2]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 4500)

    def test_0(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
                       (2, 3, 50)]
        maxIn = [1, 60, 50, 7]
        maxOut = [1, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 1)

    def test_1(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
                       (2, 3, 50)]
        maxIn = [20, 60, 50, 7]
        maxOut = [20, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 7)

    def test_2(self):
        connections = [(0, 1, 10), (0, 2, 100), (1, 3, 50),
                       (2, 3, 50)]
        maxIn = [1, 60, 50, 7]
        maxOut = [20, 60, 50, 7]
        origin = 0
        targets = [3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 7)

    def test_3(self):
        connections = [(0, 1, 20), (1, 2, 20)]
        maxIn = [20, 5, 20]
        maxOut = [20, 100, 20]
        origin = 0
        targets = [1]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 5)

    def test_4(self):
        connections = [(0, 1, 50), (1, 2, 20), (1, 3, 35)]
        maxIn = [60, 60, 10, 30]
        maxOut = [60, 50, 30, 2]
        origin = 0
        targets = [2, 3]
        self.assertEqual(maxThroughput(connections, maxIn, maxOut, origin, targets), 40)
"""

















############################   Q2    ###############################
import sys


class Catnode:
    """
    Node description:
    this is the node class declaration for all nodes in CatsTries
    the meaning of each value payload would be explained in the comments
    section bewlow

    :Input:
    argv1:char, the character of the node


    :Time complexity: O(c)
    :Space complexity: O(c)
    :Aux space complexity:O(1)
    """

    def __init__(self, char):
        self.char = char  # store the char in self.char
        self.children = [
                            None] * 27  # this is a list containing the children object if there is, none if children is not exist. child node with character 'a' stored in position 0,'b' at 1, ...'$' at position 26
        self.level = 0  # this indicates the level
        self.count = 0  # this indicates how many times current node has been visited

        self.parent = None  # this keeps track of the parent node
        self.highest_f_of_children = 0  # this keeps track of the highest number of the end node visited
        self.s_index_of_highest_f = None  # this is the position index of senetence list, from this valuable, we could quickly find the whole word through this index


# The CatsTrie class structure
class CatsTrie:

    def __init__(self, sentences):
        """
        Function description:
        Approach description (if main function):
        :Input:
        argv1:sentences, the sentences list stated in the problem

        :Output, return or postcondition:
        after this initialization, the input sentence list would be sorted using radix sort and the cattries is constructed
        :Time complexity:O(MN), the time taken by radix sort and construct tries
        :Space complexity:O(MN), containing sentence list and the trie
        :Aux space complexity: O(c)
        """

        sentences = CatsTrie.radix_sort(sentences, 26)  # use radix sort to sort the input sentence list, takes O(MN)

        self.root = Catnode("")  # initialize the root character to be ""
        self.sentences = []
        for item in sentences:  # add '$' at the end of each string in sorted sentence list
            item = item + '$'
            self.sentences.append(item)

        CatsTrie.construct_tries(self)  # construct the tries, takes O(MN)
        CatsTrie.add_level(self, self.root, 0)  # add level to each node in the tries, takes O(MN)

    def radix_sort(arr, base):
        """
        Function description:
        radix sort the arr

        :Input:
        argv1:base, indicate the base number, 26 in this case
        argv2:arr, input string list

        :Output, return or postcondition:sorted arr lexicographcially from least to largest

        :Time complexity:O(MN),
        :Space complexity:O(MN), to contain arr
        :Aux space complexity:O(c)
        """
        digits = 0
        for item in arr:  # find the longest string, string length is digits, takes O(N)
            if len(item) > digits:
                digits = len(item)

        ###the following block takes O(MN)
        for digit in range(digits - 1, -1,
                           -1):  # takes, O(M), sort the string list from last position to first position  get the sorted arr lexicographcially from least to largest
            new_arr = CatsTrie.radix_pass(arr, base, digit)  # takes O(N)
            arr = new_arr
        ###
        return arr

    def radix_pass(arr, base, digit):
        """
        Function description:
        ralated function to radix sort, sort the position of the string list based on position digit
        :Input:
        argv1:arr, the string list needs to be sorted
        argv2:base, in this case 27
        argv3:digit, the position of each string that count sort used to sort
        :Output, return or postcondition:
        :Time complexity:O(N),N is the number of string in input
        :Aux space complexity:O(MN), temp
        """

        counter = [0] * (base + 1)  # initialized counter arr
        n = len(arr)
        min_base = ord('a') - 1  # represent the " " in the string

        for i in range(n):  # update counter list, takes O(N)
            letter = (ord(arr[i][digit]) - min_base if digit < len(arr[i]) else 0)
            counter[letter] += 1

        position = [0] * (base + 1)  # initialize the position list
        position[0] = 1  # set first to be one

        for v in range(base):  # update the position list
            position[v + 1] = position[v] + counter[v]

        temp = [0] * n  # containing the sorted arr list

        for i in range(n):  # using counting sort to sort the arr and store the sorted arr in temp,takes O(N)
            d = (ord(arr[i][digit]) - min_base if digit < len(arr[i]) else 0)

            temp[position[d] - 1] = arr[i]
            position[d] += 1

        return temp

    def autoComplete(self, prompt):
        """
        Function description: autoComplete the promt
        Approach description: use radix sort to sort the sentences list, so that in the tries construction, always put the
        lexicographically smaller string first. In addidion, in the constuction, I always keep the highest frequency end node
        with smallest lexicographic value as well as the index of the sentences list coorespnding. Therefore, i could easily get
        the whole string word
        :Input:
        argv1:prompt, as stated in the problem

        :Output, return or postcondition: completed word or None if prompt does not match any
        :Time complexity:O(X), X is the length of the prompt
        :Aux space complexity:O(c)
        """
        node = self.root
        if prompt == "":  # if see "", output the word coresponding to highest frequency end node with smallest lexicographical value stored in the payload
            max_frequency = 0
            max_frequency_node = None
            for item in node.children:  # takes O(c), since letter size is fixed
                if item != None:
                    if item.highest_f_of_children > max_frequency:
                        max_frequency = item.highest_f_of_children
                        max_frequency_node = item
            return self.sentences[max_frequency_node.s_index_of_highest_f][:-1]

        else:
            for s in prompt:  # takes O(X), X is the length of prompt
                if node.children[ord(s) - ord('a')] == None:  # if prompt does not match, return none and end program
                    return None
                    sys.exit()
                else:
                    node = node.children[ord(s) - ord('a')]

            # if finish traversing the whole prompt, output the word coorespnding to the highest frequency with smallest lexicographical value stored in payload
            return self.sentences[node.s_index_of_highest_f][:-1]

    def construct_tries(self):
        """
        Function description:
        construct the tries

        :Output, return or postcondition:after the construction,
        each node will have a payload called count, indicating how many times the current node has been visited.

        each node will have a payload called highest_f_of_children,
        it indicates the highest count of the end node under the current node with lexicograhically smallest value,

        each node will have a payload called s_index_of_highest_f,
        it indicates the index of string in the sentences list cooreponding to the end node with highest frequency, from
        this index, it can be easily to get the whole string word from the sentence list


        :Time complexity:O(MN),
        :Space complexity:O(MN)
        :Aux space complexity:O(c)
        """

        for index, item in enumerate(
                self.sentences):  # go through all node to update the node count, it take O(N), N is the number of string in sentence

            node = self.root

            for c in item:  # it takes O(M), M is the longest string len, so the total time complexity is O(MN)
                if c != '$':
                    if node.children[ord(c) - ord('a')] == None:
                        new_node = Catnode(c)
                        new_node.parent = node

                        node.children[ord(c) - ord('a')] = new_node
                        node = new_node
                        node.count = node.count + 1
                    else:
                        node = node.children[ord(c) - ord('a')]
                        node.count = node.count + 1

                elif c == '$':  # see the end node simbol

                    if node.children[26] == None:
                        new_node = Catnode('$')
                        new_node.parent = node
                        node.children[26] = new_node
                        node = new_node
                        node.count = node.count + 1
                    elif node.children[26].char == '$':
                        node = node.children[26]
                        node.count = node.count + 1

                    # after seeing the end node symbol, traverse back to root, so it takes total of O(2MN)=O(MN)
                    # trverse from end node back to the root to update,
                    # add highest frequency of children end node to each node above, the beautiful thing here is :
                    # since i am updating the children list from left to right, so if within the same branch,
                    # two children end node have same highest frequency, always the lexicographically smaller children
                    # end node would come out
                    highest_f_of_children = node.count  # get the count value from the end node
                    node.highest_f_of_children = highest_f_of_children  # store the number of frequency to highest_f_of_children
                    sentences_index = index  # and the index in the for loop is actually the index of the positionn in the string list
                    node.s_index_of_highest_f = sentences_index  # store the index of the string list , the current end node refers to

                    while node != self.root:  # update all the above value if a bigger frequency node found to the root

                        new_node = node.parent

                        if node.highest_f_of_children > new_node.highest_f_of_children:  # update the highest frequency number if it is bigger than the current highest
                            new_node.highest_f_of_children = node.highest_f_of_children
                            new_node.s_index_of_highest_f = sentences_index
                        node = new_node

    def add_level(self, node, level):
        """
        Function description: add level to each node

        :Input:
        argv1:node, the current node
        argv2:level, the level of current node
        :Output, return or postcondition:node level is added

        :Time complexity:O(MN), since traverse the whole tries
        :Aux space complexity:O(c)
        """

        for item in node.children:
            if item != None:
                item.level = level + 1
                node = item
                self.add_level(node, node.level)


# this block of code is for testing, the function of it is to display the tries for better visualization

#    def display(self,node,level):

#        for item in node.children:
#            if item != None:

# print(item.char, item.level,item.count,item.highest_f_of_children,item.s_index_of_highest_f)
#                node =item
#                self.display(node,node.level)

"""                                     test cases for Q2 
import unittest


class TestA2(unittest.TestCase):
    def test_edgerunners_case_3(self):
        # Lyrics from I Really Want to Stay at Your House
        # Source: https://open.spotify.com/track/7mykoq6R3BArsSpNDjFQTm?si=ad96c59a91464936
        sentences = [
            'i', 'couldnt', 'wait', 'for', 'you', 'to', 'come', 'clear', 'the', 'cupboards',
            'but', 'now', 'youre', 'going', 'to', 'leave', 'with', 'nothing', 'but', 'a', 'sign',
            'another', 'evening', 'ill', 'be', 'sitting', 'reading', 'in', 'between', 'your', 'lines',
            'because', 'i', 'miss', 'you', 'all', 'the', 'time',

            'so', 'get', 'away',
            'another', 'way', 'to', 'feel', 'what', 'you', 'didnt', 'want', 'yourself', 'to', 'know',
            'and', 'let', 'yourself', 'go', 'you', 'know', 'you', 'didnt', 'lose', 'your', 'selfcontrol',
            'lets', 'start', 'at', 'the', 'rainbow',
            'turn', 'away',
            'another', 'way', 'to', 'be', 'where', 'you', 'didnt', 'want', 'yourself', 'to', 'go',
            'and', 'let', 'yourself', 'go',
            'is', 'that', 'a', 'compromise',

            'so', 'what', 'do', 'you', 'wanna', 'do', 'whats', 'your', 'pointofview',
            'theres', 'a', 'party', 'soon', 'do', 'you', 'wanna', 'go',
            'a', 'handshake', 'with', 'you', 'whats', 'your', 'pointofview',
            'im', 'on', 'top', 'of', 'you', 'i', 'dont', 'wanna', 'go',
            'cause', 'i', 'really', 'wanna', 'stay', 'at', 'your', 'house',
            'and', 'i', 'hope', 'this', 'works', 'out',
            'but', 'you', 'know', 'how', 'much', 'you', 'broke', 'me', 'apart',
            'im', 'done', 'with', 'you', 'im', 'ignoring', 'you',
            'i', 'dont', 'wanna', 'know',

            'and', 'im', 'aware', 'that', 'you', 'were', 'lying', 'in', 'the', 'gutter',
            'cause', 'i', 'did', 'everything', 'to', 'be', 'there', 'by', 'your', 'sideide',
            'so', 'when', 'you', 'tell', 'me', 'im', 'the', 'reason', 'i', 'just', 'cant', 'believe', 'the', 'lies',
            'and', 'why', 'do', 'i', 'so', 'want', 'to', 'call', 'you', 'call', 'you', 'call', 'you', 'call', 'you',

            'so', 'what', 'do', 'you', 'wanna', 'do', 'whats', 'your', 'pointofview',
            'theres', 'a', 'party', 'soon', 'do', 'you', 'wanna', 'go',
            'a', 'handshake', 'with', 'you', 'whats', 'your', 'pointofview',
            'im', 'on', 'top', 'of', 'you', 'i', 'dont', 'wanna', 'go',
            'cause', 'i', 'really', 'wanna', 'stay', 'at', 'your', 'house',
            'and', 'i', 'hope', 'this', 'works', 'out',
            'but', 'you', 'know', 'how', 'much', 'you', 'broke', 'me', 'apart',
            'im', 'done', 'with', 'you', 'im', 'ignoring', 'you',
            'i', 'dont', 'wanna', 'know',

            'oh',
            'ohoh', 'ohohoh',
            'i', 'dont', 'know', 'why', 'im', 'no', 'one',

            'so', 'get', 'away',
            'another', 'way', 'to', 'feel', 'what', 'you', 'didnt', 'want', 'yourself', 'to', 'know',
            'and', 'let', 'yourself', 'go',
            'you', 'know', 'you', 'didnt', 'lose', 'your', 'selfcontrol',
            'lets', 'start', 'at', 'the', 'rainbow',
            'turn', 'away',
            'another', 'way', 'to', 'be', 'where', 'you', 'didnt', 'want', 'yourself', 'to', 'go',
            'let', 'yourself', 'go',
            'is', 'that', 'a', 'compromise',

            'so', 'what', 'do', 'you', 'wanna', 'do', 'whats', 'your', 'pointofview',
            'theres', 'a', 'party', 'soon', 'do', 'you', 'wanna', 'go',
            'a', 'handshake', 'with', 'you', 'whats', 'your', 'pointofview',
            'im', 'on', 'top', 'of', 'you', 'i', 'dont', 'wanna', 'go',
            'cause', 'i', 'really', 'wanna', 'stay', 'at', 'your', 'house',
            'and', 'i', 'hope', 'this', 'works', 'out',
            'but', 'you', 'know', 'how', 'much', 'you', 'broke', 'me', 'apart',
            'im', 'done', 'with', 'you', 'im', 'ignoring', 'you',
            'i', 'dont', 'wanna', 'know'
        ]
        mycattrie = CatsTrie(sentences)
        self.assertEqual(mycattrie.autoComplete(""), "you")
        self.assertEqual(mycattrie.autoComplete("a"), "a")
        self.assertEqual(mycattrie.autoComplete("b"), "but")
        self.assertEqual(mycattrie.autoComplete("c"), "call")
        self.assertEqual(mycattrie.autoComplete("d"), "do")
        self.assertEqual(mycattrie.autoComplete("e"), "evening")
        self.assertEqual(mycattrie.autoComplete("f"), "feel")
        self.assertEqual(mycattrie.autoComplete("g"), "go")
        self.assertEqual(mycattrie.autoComplete("h"), "handshake")
        self.assertEqual(mycattrie.autoComplete("i"), "i")
        self.assertEqual(mycattrie.autoComplete("j"), "just")
        self.assertEqual(mycattrie.autoComplete("k"), "know")
        self.assertEqual(mycattrie.autoComplete("l"), "let")
        self.assertEqual(mycattrie.autoComplete("m"), "me")
        self.assertEqual(mycattrie.autoComplete("n"), "no")
        self.assertEqual(mycattrie.autoComplete("o"), "of")
        self.assertEqual(mycattrie.autoComplete("p"), "pointofview")
        self.assertEqual(mycattrie.autoComplete("q"), None)
        self.assertEqual(mycattrie.autoComplete("r"), "really")
        self.assertEqual(mycattrie.autoComplete("s"), "so")
        self.assertEqual(mycattrie.autoComplete("t"), "to")
        self.assertEqual(mycattrie.autoComplete("u"), None)
        self.assertEqual(mycattrie.autoComplete("v"), None)
        self.assertEqual(mycattrie.autoComplete("w"), "wanna")
        self.assertEqual(mycattrie.autoComplete("x"), None)
        self.assertEqual(mycattrie.autoComplete("y"), "you")
        self.assertEqual(mycattrie.autoComplete("z"), None)
        self.assertEqual(mycattrie.autoComplete("you"), "you")
        self.assertEqual(mycattrie.autoComplete("your"), "your")
        self.assertEqual(mycattrie.autoComplete("youre"), "youre")
        self.assertEqual(mycattrie.autoComplete("control"), None)
        self.assertEqual(mycattrie.autoComplete("view"), None)
        self.assertEqual(mycattrie.autoComplete("do"), "do")
        self.assertEqual(mycattrie.autoComplete("don"), "dont")
        self.assertEqual(mycattrie.autoComplete("ing"), None)
        self.assertEqual(mycattrie.autoComplete("on"), "on")
        self.assertEqual(mycattrie.autoComplete("one"), "one")
        self.assertEqual(mycattrie.autoComplete("oh"), "oh")

    def test1(self):
        sentences = ["abc", "abazacy", "dbcef", "xzz", "gdbc", "abazacy", "xyz", "abazacy", "dbcef", "xyz", "xxx",
                     "xzz"]
        mycattrie = CatsTrie(sentences)
        ans = mycattrie.autoComplete('ba')
        expect = None
        self.assertEqual(ans, expect)

        def test_default(self):
            sentences = ["abc", "abazacy", "dbcef", "xzz", "gdbc", "abazacy", "xyz", "abazacy", "dbcef", "xyz", "xxx",
                         "xzz"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("ab") == "abazacy")
            self.assertTrue(trie.autoComplete("a") == "abazacy")
            self.assertTrue(trie.autoComplete("dbcef") == "dbcef")
            self.assertTrue(trie.autoComplete("dbcefz") == None)
            self.assertTrue(trie.autoComplete("ba") == None)
            self.assertTrue(trie.autoComplete("x") == "xyz")
            self.assertTrue(trie.autoComplete("") == "abazacy")

        def test_default_2(self):
            sentences = ["abc", "abczacy", "dbcef", "xzz", "gdbc", "abczacy", "xyz", "abczacy", "dbcef", "xyz", "xxx",
                         "xzz"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("abc") == "abczacy")

        def test_01(self):
            sentences = ["ab", "a"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("") == "a")
            self.assertTrue(trie.autoComplete("a") == "a")
            self.assertTrue(trie.autoComplete("ab") == "ab")
            self.assertTrue(trie.autoComplete("abc") == None)
            self.assertTrue(trie.autoComplete("b") == None)
            self.assertTrue(trie.autoComplete("fittwozerozerofour") == None)

        def test_02(self):
            sentences = ["a", "ab"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("") == "a")
            self.assertTrue(trie.autoComplete("a") == "a")
            self.assertTrue(trie.autoComplete("ab") == "ab")
            self.assertTrue(trie.autoComplete("abc") == None)
            self.assertTrue(trie.autoComplete("b") == None)
            self.assertTrue(trie.autoComplete("fittwozerozerofour") == None)

        def test_03(self):
            sentences = ["", "", "cat"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("") == "")
            self.assertTrue(trie.autoComplete("a") == None)
            self.assertTrue(trie.autoComplete("b") == None)
            self.assertTrue(trie.autoComplete("c") == "cat")
            self.assertTrue(trie.autoComplete("ca") == "cat")
            self.assertTrue(trie.autoComplete("cat") == "cat")
            self.assertTrue(trie.autoComplete("cats") == None)
            self.assertTrue(trie.autoComplete("dat") == None)

        def test_04(self):
            sentences = ["bad", "bed", "", "", "bad", "bud", "bod", "", "bid", "", "", "bed", "bad", "", "bod", "bud",
                         "bud", "bmeow", ""]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("") == "")
            self.assertTrue(trie.autoComplete("b") == "bad")
            self.assertTrue(trie.autoComplete("bi") == "bid")
            self.assertTrue(trie.autoComplete("bo") == "bod")
            self.assertTrue(trie.autoComplete("bm") == "bmeow")

        def test_05(self):
            sentences = ["a", "a", "aa", "aa"]
            trie = CatsTrie(sentences)
            self.assertTrue(trie.autoComplete("") == "a")
            self.assertTrue(trie.autoComplete("a") == "a")
            self.assertTrue(trie.autoComplete("aa") == "aa")
            self.assertTrue(trie.autoComplete("b") == None)

    def test_07(self):
        # Extract of The Oxford 3000 word list
        # Source: https://www.oxfordlearnersdictionaries.com/wordlist/american_english/oxford3000/
        sentences = ["fit", "fix", "fixed", "flag", "flame", "flash", "flat", "flavor", "flesh", "flight", "float",
                     "flood", "floor", "flour", "flow", "flower", "flu", "fly", "flying", "focus", "fold", "folding",
                     "folk", "follow", "following", "food", "foot", "football", "for", "force", "forecast", "foreign",
                     "forest", "forever", "forget", "forgive", "fork", "form", "formal", "former", "formerly",
                     "formula", "fortune", "forward", "found", "foundation"]
        trie = CatsTrie(sentences)
        self.assertTrue(trie.autoComplete("") == "fit")
        self.assertTrue(trie.autoComplete("f") == "fit")
        self.assertTrue(trie.autoComplete("fi") == "fit")
        self.assertTrue(trie.autoComplete("fify") == None)
        self.assertTrue(trie.autoComplete("fit") == "fit")
        self.assertTrue(trie.autoComplete("fiu") == None)
        self.assertTrue(trie.autoComplete("fiv") == None)
        self.assertTrue(trie.autoComplete("fiw") == None)
        self.assertTrue(trie.autoComplete("fix") == "fix")
        self.assertTrue(trie.autoComplete("fixe") == "fixed")
        self.assertTrue(trie.autoComplete("fiy") == None)
        self.assertTrue(trie.autoComplete("fj") == None)
        self.assertTrue(trie.autoComplete("fk") == None)
        self.assertTrue(trie.autoComplete("fl") == "flag")
        self.assertTrue(trie.autoComplete("fla") == "flag")
        self.assertTrue(trie.autoComplete("flat") == "flat")
        self.assertTrue(trie.autoComplete("flo") == "float")
        self.assertTrue(trie.autoComplete("flu") == "flu")
        self.assertTrue(trie.autoComplete("fo") == "focus")
        self.assertTrue(trie.autoComplete("foo") == "food")
        self.assertTrue(trie.autoComplete("for") == "for")
        self.assertTrue(trie.autoComplete("forb") == None)
        self.assertTrue(trie.autoComplete("forc") == "force")
        self.assertTrue(trie.autoComplete("ford") == None)
        self.assertTrue(trie.autoComplete("monash") == None)

    def test_08(self):
        # Extract of the Constitution of the United States, Article I
        # Source: https://www.archives.gov/founding-docs/constitution-transcript
        sentences = [

            "we", "the", "people", "of", "the", "united", "states", "in", "order", "to", "form", "a", "more", "perfect",
            "union", "establish", "justice", "insure", "domestic", "tranquility", "provide", "for", "the", "common",
            "defence", "promote", "the", "general", "welfare", "and", "secure", "the", "blessings", "of", "liberty",
            "to", "ourselves", "and", "our", "posterity", "do", "ordain", "and", "establish", "this", "constitution",
            "for", "the", "united", "states", "of", "america",

            "article", "one",

            "section", "one",

            "all", "legislative", "powers", "herein", "granted", "shall", "be", "vested", "in", "a", "congress", "of",
            "the", "united", "states", "which", "shall", "consist", "of", "a", "senate", "and", "house", "of",
            "representatives",

            "section", "two",

            "the", "house", "of", "representatives", "shall", "be", "composed", "of", "members", "chosen", "every",
            "second", "year", "by", "the", "people", "of", "the", "several", "states", "and", "the", "electors", "in",
            "each", "state", "shall", "have", "the", "qualifications", "requisite", "for", "electors", "of", "the",
            "most", "numerous", "branch", "of", "the", "state", "legislature",
            "no", "person", "shall", "be", "a", "representative", "who", "shall", "not", "have", "attained", "to",
            "the", "age", "of", "twenty", "five", "years", "and", "been", "seven", "years", "a", "citizen", "of", "the",
            "united", "states", "and", "who", "shall", "not", "when", "elected", "be", "an", "inhabitant", "of", "that",
            "state", "in", "which", "he", "shall", "be", "chosen",
            "representatives", "and", "direct", "taxes", "shall", "be", "apportioned", "among", "the", "several",
            "states", "which", "may", "be", "included", "within", "this", "union", "according", "to", "their",
            "respective", "numbers", "which", "shall", "be", "determined", "by", "adding", "to", "the", "whole",
            "number", "of", "free", "persons", "including", "those", "bound", "to", "service", "for", "a", "term", "of",
            "years", "and", "excluding", "indians", "not", "taxed", "three", "fifths", "of", "all", "other", "persons",
            "the", "actual", "enumeration", "shall", "be", "made", "within", "three", "years", "after", "the", "first",
            "meeting", "of", "the", "congress", "of", "the", "united", "states", "and", "within", "every", "subsequent",
            "term", "of", "ten", "years", "in", "such", "manner", "as", "they", "shall", "by", "law", "direct", "the",
            "number", "of", "representatives", "shall", "not", "exceed", "one", "for", "every", "thirty", "thousand",
            "but", "each", "state", "shall", "have", "at", "least", "one", "representative", "and", "until", "such",
            "enumeration", "shall", "be", "made", "the", "state", "of", "new", "hampshire", "shall", "be", "entitled",
            "to", "chuse", "three", "massachusetts", "eight", "rhodeisland", "and", "providence", "plantations", "one",
            "connecticut", "five", "newyork", "six", "new", "jersey", "four", "pennsylvania", "eight", "delaware",
            "one", "maryland", "six", "virginia", "ten", "north", "carolina", "five", "south", "carolina", "five",
            "and", "georgia", "three",
            "when", "vacancies", "happen", "in", "the", "representation", "from", "any", "state", "the", "executive",
            "authority", "thereof", "shall", "issue", "writs", "of", "election", "to", "fill", "such", "vacancies",
            "the", "house", "of", "representatives", "shall", "chuse", "their", "speaker", "and", "other", "officers",
            "and", "shall", "have", "the", "sole", "power", "of", "impeachment",

            "section", "three",

            "the", "senate", "of", "the", "united", "states", "shall", "be", "composed", "of", "two", "senators",
            "from", "each", "state", "chosen", "by", "the", "legislature", "thereof", "for", "six", "years", "and",
            "each", "senator", "shall", "have", "one", "vote",
            "immediately", "after", "they", "shall", "be", "assembled", "in", "consequence", "of", "the", "first",
            "election", "they", "shall", "be", "divided", "as", "equally", "as", "may", "be", "into", "three",
            "classes", "the", "seats", "of", "the", "senators", "of", "the", "first", "class", "shall", "be", "vacated",
            "at", "the", "expiration", "of", "the", "second", "year", "of", "the", "second", "class", "at", "the",
            "expiration", "of", "the", "fourth", "year", "and", "of", "the", "third", "class", "at", "the",
            "expiration", "of", "the", "sixth", "year", "so", "that", "one", "third", "may", "be", "chosen", "every",
            "second", "year", "and", "if", "vacancies", "happen", "by", "resignation", "or", "otherwise", "during",
            "the", "recess", "of", "the", "legislature", "of", "any", "state", "the", "executive", "thereof", "may",
            "make", "temporary", "appointments", "until", "the", "next", "meeting", "of", "the", "legislature", "which",
            "shall", "then", "fill", "such", "vacancies",
            "no", "person", "shall", "be", "a", "senator", "who", "shall", "not", "have", "attained", "to", "the",
            "age", "of", "thirty", "years", "and", "been", "nine", "years", "a", "citizen", "of", "the", "united",
            "states", "and", "who", "shall", "not", "when", "elected", "be", "an", "inhabitant", "of", "that", "state",
            "for", "which", "he", "shall", "be", "chosen",
            "the", "vice", "president", "of", "the", "united", "states", "shall", "be", "president", "of", "the",
            "senate", "but", "shall", "have", "no", "vote", "unless", "they", "be", "equally", "divided",
            "the", "senate", "shall", "chuse", "their", "other", "officers", "and", "also", "a", "president", "pro",
            "tempore", "in", "the", "absence", "of", "the", "vice", "president", "or", "when", "he", "shall",
            "exercise", "the", "office", "of", "president", "of", "the", "united", "states",
            "the", "senate", "shall", "have", "the", "sole", "power", "to", "try", "all", "impeachments", "when",
            "sitting", "for", "that", "purpose", "they", "shall", "be", "on", "oath", "or", "affirmation", "when",
            "the", "president", "of", "the", "united", "states", "is", "tried", "the", "chief", "justice", "shall",
            "preside", "and", "no", "person", "shall", "be", "convicted", "without", "the", "concurrence", "of", "two",
            "thirds", "of", "the", "members", "present",
            "judgment", "in", "cases", "of", "impeachment", "shall", "not", "extend", "further", "than", "to",
            "removal", "from", "office", "and", "disqualification", "to", "hold", "and", "enjoy", "any", "office", "of",
            "honor", "trust", "or", "profit", "under", "the", "united", "states", "but", "the", "party", "convicted",
            "shall", "nevertheless", "be", "liable", "and", "subject", "to", "indictment", "trial", "judgment", "and",
            "punishment", "according", "to", "law",

            "section", "four",

            "the", "times", "places", "and", "manner", "of", "holding", "elections", "for", "senators", "and",
            "representatives", "shall", "be", "prescribed", "in", "each", "state", "by", "the", "legislature",
            "thereof", "but", "the", "congress", "may", "at", "any", "time", "by", "law", "make", "or", "alter", "such",
            "regulations", "except", "as", "to", "the", "places", "of", "chusing", "senators",
            "the", "congress", "shall", "assemble", "at", "least", "once", "in", "every", "year", "and", "such",
            "meeting", "shall", "be", "on", "the", "first", "monday", "in", "december", "unless", "they", "shall", "by",
            "law", "appoint", "a", "different", "day",

            "section", "five",

            "each", "house", "shall", "be", "the", "judge", "of", "the", "elections", "returns", "and",
            "qualifications", "of", "its", "own", "members", "and", "a", "majority", "of", "each", "shall",
            "constitute", "a", "quorum", "to", "do", "business", "but", "a", "smaller", "number", "may", "adjourn",
            "from", "day", "to", "day", "and", "may", "be", "authorized", "to", "compel", "the", "attendance", "of",
            "absent", "members", "in", "such", "manner", "and", "under", "such", "penalties", "as", "each", "house",
            "may", "provide",
            "each", "house", "may", "determine", "the", "rules", "of", "its", "proceedings", "punish", "its", "members",
            "for", "disorderly", "behaviour", "and", "with", "the", "concurrence", "of", "two", "thirds", "expel", "a",
            "member",
            "each", "house", "shall", "keep", "a", "journal", "of", "its", "proceedings", "and", "from", "time", "to",
            "time", "publish", "the", "same", "excepting", "such", "parts", "as", "may", "in", "their", "judgment",
            "require", "secrecy", "and", "the", "yeas", "and", "nays", "of", "the", "members", "of", "either", "house",
            "on", "any", "question", "shall", "at", "the", "desire", "of", "one", "fifth", "of", "those", "present",
            "be", "entered", "on", "the", "journal",
            "neither", "house", "during", "the", "session", "of", "congress", "shall", "without", "the", "consent",
            "of", "the", "other", "adjourn", "for", "more", "than", "three", "days", "nor", "to", "any", "other",
            "place", "than", "that", "in", "which", "the", "two", "houses", "shall", "be", "sitting",

            "section", "six",

            "the", "senators", "and", "representatives", "shall", "receive", "a", "compensation", "for", "their",
            "services", "to", "be", "ascertained", "by", "law", "and", "paid", "out", "of", "the", "treasury", "of",
            "the", "united", "states", "they", "shall", "in", "all", "cases", "except", "treason", "felony", "and",
            "breach", "of", "the", "peace", "be", "privileged", "from", "arrest", "during", "their", "attendance", "at",
            "the", "session", "of", "their", "respective", "houses", "and", "in", "going", "to", "and", "returning",
            "from", "the", "same", "and", "for", "any", "speech", "or", "debate", "in", "either", "house", "they",
            "shall", "not", "be", "questioned", "in", "any", "other", "place",
            "no", "senator", "or", "representative", "shall", "during", "the", "time", "for", "which", "he", "was",
            "elected", "be", "appointed", "to", "any", "civil", "office", "under", "the", "authority", "of", "the",
            "united", "states", "which", "shall", "have", "been", "created", "or", "the", "emoluments", "whereof",
            "shall", "have", "been", "encreased", "during", "such", "time", "and", "no", "person", "holding", "any",
            "office", "under", "the", "united", "states", "shall", "be", "a", "member", "of", "either", "house",
            "during", "his", "continuance", "in", "office",

            "section", "seven",

            "all", "bills", "for", "raising", "revenue", "shall", "originate", "in", "the", "house", "of",
            "representatives", "but", "the", "senate", "may", "propose", "or", "concur", "with", "amendments", "as",
            "on", "other", "bills",
            "every", "bill", "which", "shall", "have", "passed", "the", "house", "of", "representatives", "and", "the",
            "senate", "shall", "before", "it", "become", "a", "law", "be", "presented", "to", "the", "president", "of",
            "the", "united", "states", "if", "he", "approve", "he", "shall", "sign", "it", "but", "if", "not", "he",
            "shall", "return", "it", "with", "his", "objections", "to", "that", "house", "in", "which", "it", "shall",
            "have", "originated", "who", "shall", "enter", "the", "objections", "at", "large", "on", "their", "journal",
            "and", "proceed", "to", "reconsider", "it", "if", "after", "such", "reconsideration", "two", "thirds", "of",
            "that", "house", "shall", "agree", "to", "pass", "the", "bill", "it", "shall", "be", "sent", "together",
            "with", "the", "objections", "to", "the", "other", "house", "by", "which", "it", "shall", "likewise", "be",
            "reconsidered", "and", "if", "approved", "by", "two", "thirds", "of", "that", "house", "it", "shall",
            "become", "a", "law", "but", "in", "all", "such", "cases", "the", "votes", "of", "both", "houses", "shall",
            "be", "determined", "by", "yeas", "and", "nays", "and", "the", "names", "of", "the", "persons", "voting",
            "for", "and", "against", "the", "bill", "shall", "be", "entered", "on", "the", "journal", "of", "each",
            "house", "respectively", "if", "any", "bill", "shall", "not", "be", "returned", "by", "the", "president",
            "within", "ten", "days", "sundays", "excepted", "after", "it", "shall", "have", "been", "presented", "to",
            "him", "the", "same", "shall", "be", "a", "law", "in", "like", "manner", "as", "if", "he", "had", "signed",
            "it", "unless", "the", "congress", "by", "their", "adjournment", "prevent", "its", "return", "in", "which",
            "case", "it", "shall", "not", "be", "a", "law",
            "every", "order", "resolution", "or", "vote", "to", "which", "the", "concurrence", "of", "the", "senate",
            "and", "house", "of", "representatives", "may", "be", "necessary", "except", "on", "a", "question", "of",
            "adjournment", "shall", "be", "presented", "to", "the", "president", "of", "the", "united", "states", "and",
            "before", "the", "same", "shall", "take", "effect", "shall", "be", "approved", "by", "him", "or", "being",
            "disapproved", "by", "him", "shall", "be", "repassed", "by", "two", "thirds", "of", "the", "senate", "and",
            "house", "of", "representatives", "according", "to", "the", "rules", "and", "limitations", "prescribed",
            "in", "the", "case", "of", "a", "bill",

            "section", "eight",

            "the", "congress", "shall", "have", "power", "to", "lay", "and", "collect", "taxes", "duties", "imposts",
            "and", "excises", "to", "pay", "the", "debts", "and", "provide", "for", "the", "common", "defence", "and",
            "general", "welfare", "of", "the", "united", "states", "but", "all", "duties", "imposts", "and", "excises",
            "shall", "be", "uniform", "throughout", "the", "united", "states",
            "to", "borrow", "money", "on", "the", "credit", "of", "the", "united", "states",
            "to", "regulate", "commerce", "with", "foreign", "nations", "and", "among", "the", "several", "states",
            "and", "with", "the", "indian", "tribes",
            "to", "establish", "an", "uniform", "rule", "of", "naturalization", "and", "uniform", "laws", "on", "the",
            "subject", "of", "bankruptcies", "throughout", "the", "united", "states",
            "to", "coin", "money", "regulate", "the", "value", "thereof", "and", "of", "foreign", "coin", "and", "fix",
            "the", "standard", "of", "weights", "and", "measures",
            "to", "provide", "for", "the", "punishment", "of", "counterfeiting", "the", "securities", "and", "current",
            "coin", "of", "the", "united", "states",
            "to", "establish", "post", "offices", "and", "post", "roads",
            "to", "promote", "the", "progress", "of", "science", "and", "useful", "arts", "by", "securing", "for",
            "limited", "times", "to", "authors", "and", "inventors", "the", "exclusive", "right", "to", "their",
            "respective", "writings", "and", "discoveries",
            "to", "constitute", "tribunals", "inferior", "to", "the", "supreme", "court",
            "to", "define", "and", "punish", "piracies", "and", "felonies", "committed", "on", "the", "high", "seas",
            "and", "offences", "against", "the", "law", "of", "nations",
            "to", "declare", "war", "grant", "letters", "of", "marque", "and", "reprisal", "and", "make", "rules",
            "concerning", "captures", "on", "land", "and", "water",
            "to", "raise", "and", "support", "armies", "but", "no", "appropriation", "of", "money", "to", "that", "use",
            "shall", "be", "for", "a", "longer", "term", "than", "two", "years",
            "to", "provide", "and", "maintain", "a", "navy",
            "to", "make", "rules", "for", "the", "government", "and", "regulation", "of", "the", "land", "and", "naval",
            "forces",
            "to", "provide", "for", "calling", "forth", "the", "militia", "to", "execute", "the", "laws", "of", "the",
            "union", "suppress", "insurrections", "and", "repel", "invasions",
            "to", "provide", "for", "organizing", "arming", "and", "disciplining", "the", "militia", "and", "for",
            "governing", "such", "part", "of", "them", "as", "may", "be", "employed", "in", "the", "service", "of",
            "the", "united", "states", "reserving", "to", "the", "states", "respectively", "the", "appointment", "of",
            "the", "officers", "and", "the", "authority", "of", "training", "the", "militia", "according", "to", "the",
            "discipline", "prescribed", "by", "congress",
            "to", "exercise", "exclusive", "legislation", "in", "all", "cases", "whatsoever", "over", "such",
            "district", "not", "exceeding", "ten", "miles", "square", "as", "may", "by", "cession", "of", "particular",
            "states", "and", "the", "acceptance", "of", "congress", "become", "the", "seat", "of", "the", "government",
            "of", "the", "united", "states", "and", "to", "exercise", "like", "authority", "over", "all", "places",
            "purchased", "by", "the", "consent", "of", "the", "legislature", "of", "the", "state", "in", "which", "the",
            "same", "shall", "be", "for", "the", "erection", "of", "forts", "magazines", "arsenals", "dockyards", "and",
            "other", "needful", "buildingsand",
            "to", "make", "all", "laws", "which", "shall", "be", "necessary", "and", "proper", "for", "carrying",
            "into", "execution", "the", "foregoing", "powers", "and", "all", "other", "powers", "vested", "by", "this",
            "constitution", "in", "the", "government", "of", "the", "united", "states", "or", "in", "any", "department",
            "or", "officer", "thereof",

            "section", "nine",

            "the", "migration", "or", "importation", "of", "such", "persons", "as", "any", "of", "the", "states", "now",
            "existing", "shall", "think", "proper", "to", "admit", "shall", "not", "be", "prohibited", "by", "the",
            "congress", "prior", "to", "the", "year", "one", "thousand", "eight", "hundred", "and", "eight", "but", "a",
            "tax", "or", "duty", "may", "be", "imposed", "on", "such", "importation", "not", "exceeding", "ten",
            "dollars", "for", "each", "person",
            "the", "privilege", "of", "the", "writ", "of", "habeas", "corpus", "shall", "not", "be", "suspended",
            "unless", "when", "in", "cases", "of", "rebellion", "or", "invasion", "the", "public", "safety", "may",
            "require", "it",
            "no", "bill", "of", "attainder", "or", "ex", "post", "facto", "law", "shall", "be", "passed",
            "no", "capitation", "or", "other", "direct", "tax", "shall", "be", "laid", "unless", "in", "proportion",
            "to", "the", "census", "or", "enumeration", "herein", "before", "directed", "to", "be", "taken",
            "no", "tax", "or", "duty", "shall", "be", "laid", "on", "articles", "exported", "from", "any", "state",
            "no", "preference", "shall", "be", "given", "by", "any", "regulation", "of", "commerce", "or", "revenue",
            "to", "the", "ports", "of", "one", "state", "over", "those", "of", "another", "nor", "shall", "vessels",
            "bound", "to", "or", "from", "one", "state", "be", "obliged", "to", "enter", "clear", "or", "pay", "duties",
            "in", "another",
            "no", "money", "shall", "be", "drawn", "from", "the", "treasury", "but", "in", "consequence", "of",
            "appropriations", "made", "by", "law", "and", "a", "regular", "statement", "and", "account", "of", "the",
            "receipts", "and", "expenditures", "of", "all", "public", "money", "shall", "be", "published", "from",
            "time", "to", "time",
            "no", "title", "of", "nobility", "shall", "be", "granted", "by", "the", "united", "states", "and", "no",
            "person", "holding", "any", "office", "of", "profit", "or", "trust", "under", "them", "shall", "without",
            "the", "consent", "of", "the", "congress", "accept", "of", "any", "present", "emolument", "office", "or",
            "title", "of", "any", "kind", "whatever", "from", "any", "king", "prince", "or", "foreign", "state",

            "section", "ten",

            "no", "state", "shall", "enter", "into", "any", "treaty", "alliance", "or", "confederation", "grant",
            "letters", "of", "marque", "and", "reprisal", "coin", "money", "emit", "bills", "of", "credit", "make",
            "any", "thing", "but", "gold", "and", "silver", "coin", "a", "tender", "in", "payment", "of", "debts",
            "pass", "any", "bill", "of", "attainder", "ex", "post", "facto", "law", "or", "law", "impairing", "the",
            "obligation", "of", "contracts", "or", "grant", "any", "title", "of", "nobility",
            "no", "state", "shall", "without", "the", "consent", "of", "the", "congress", "lay", "any", "imposts", "or",
            "duties", "on", "imports", "or", "exports", "except", "what", "may", "be", "absolutely", "necessary", "for",
            "executing", "its", "inspection", "laws", "and", "the", "net", "produce", "of", "all", "duties", "and",
            "imposts", "laid", "by", "any", "state", "on", "imports", "or", "exports", "shall", "be", "for", "the",
            "use", "of", "the", "treasury", "of", "the", "united", "states", "and", "all", "such", "laws", "shall",
            "be", "subject", "to", "the", "revision", "and", "controul", "of", "the", "congress",
            "no", "state", "shall", "without", "the", "consent", "of", "congress", "lay", "any", "duty", "of",
            "tonnage", "keep", "troops", "or", "ships", "of", "war", "in", "time", "of", "peace", "enter", "into",
            "any", "agreement", "or", "compact", "with", "another", "state", "or", "with", "a", "foreign", "power",
            "or", "engage", "in", "war", "unless", "actually", "invaded", "or", "in", "such", "imminent", "danger",
            "as", "will", "not", "admit", "of", "delay"

        ]
        trie = CatsTrie(sentences)
        self.assertTrue(trie.autoComplete("") == "the")
        self.assertTrue(trie.autoComplete("monash") == None)  # :(
        self.assertTrue(trie.autoComplete("a") == "and")
        self.assertTrue(trie.autoComplete("al") == "all")
        self.assertTrue(trie.autoComplete("any") == "any")
        self.assertTrue(trie.autoComplete("b") == "be")
        self.assertTrue(trie.autoComplete("by") == "by")
        self.assertTrue(trie.autoComplete("c") == "congress")
        self.assertTrue(trie.autoComplete("d") == "during")
        self.assertTrue(trie.autoComplete("e") == "each")
        self.assertTrue(trie.autoComplete("f") == "for")
        self.assertTrue(trie.autoComplete("g") == "government")
        self.assertTrue(trie.autoComplete("h") == "house")
        self.assertTrue(trie.autoComplete("ha") == "have")
        self.assertTrue(trie.autoComplete("i") == "in")
        self.assertTrue(trie.autoComplete("j") == "journal")
        self.assertTrue(trie.autoComplete("k") == "keep")
        self.assertTrue(trie.autoComplete("l") == "law")
        self.assertTrue(trie.autoComplete("m") == "may")
        self.assertTrue(trie.autoComplete("n") == "no")
        self.assertTrue(trie.autoComplete("not") == "not")
        self.assertTrue(trie.autoComplete("o") == "of")
        self.assertTrue(trie.autoComplete("on") == "on")
        self.assertTrue(trie.autoComplete("or") == "or")
        self.assertTrue(trie.autoComplete("p") == "president")
        self.assertTrue(trie.autoComplete("q") == "qualifications")
        self.assertTrue(trie.autoComplete("r") == "representatives")
        self.assertTrue(trie.autoComplete("s") == "shall")
        self.assertTrue(trie.autoComplete("st") == "states")
        self.assertTrue(trie.autoComplete("su") == "such")
        self.assertTrue(trie.autoComplete("t") == "the")
        self.assertTrue(trie.autoComplete("to") == "to")
        self.assertTrue(trie.autoComplete("u") == "united")
        self.assertTrue(trie.autoComplete("v") == "vacancies")
        self.assertTrue(trie.autoComplete("w") == "which")
        self.assertTrue(trie.autoComplete("x") == None)
        self.assertTrue(trie.autoComplete("y") == "years")
        self.assertTrue(trie.autoComplete("z") == None)

    def test_09(self):
        # Extract from the FIT2004 S1 2023 A2 assignment brief, Monash University
        sentences = [

            "fit", "two", "zero", "zero", "four", "s", "one", "twenty", "twenty", "three", "assignment", "two",

            "deadline",

            "friday", "twenty", "sixth", "may", "twenty", "twenty", "three", "sixteen", "thirty", "sharp", "aedt",

            "late", "submission", "penalty",

            "ten", "percent", "penalty", "per", "day", "submissions", "more", "than", "seven", "calendar", "days",
            "late", "will", "receive", "zero", "the", "number", "of", "days", "late", "is", "rounded", "up", "eg",
            "five", "minutes", "late", "means", "one", "day", "late", "twenty", "seven", "hours", "late", "is", "two",
            "days", "late",
            "for", "special", "consideration", "please", "visit", "the", "following", "page", "and", "fill", "out",
            "the", "appropriate", "form",
            "https", "colon", "slash", "slash", "forms", "dot", "monash", "dot", "edu", "slash", "special", "hyphen",
            "consideration", "for", "clayton", "students",
            "https", "colon", "slash", "slash", "sors", "dot", "monash", "dot", "edu", "dot", "my", "slash", "for",
            "malaysian", "students",
            "the", "deadlines", "in", "this", "unit", "are", "strict", "last", "minute", "submissions", "are", "at",
            "your", "own", "risk",

            "programming", "criteria",

            "it", "is", "required", "that", "you", "implement", "this", "exercise", "strictly", "using", "the",
            "python", "programming", "language", "version", "should", "not", "be", "earlier", "than", "three", "point",
            "five", "this", "practical", "work", "will", "be", "marked", "on", "the", "time", "complexity", "space",
            "complexity", "and", "functionality", "of", "your", "program", "and", "your", "documentation",
            "your", "program", "will", "be", "tested", "using", "automated", "test", "scripts", "it", "is", "therefore",
            "critically", "important", "that", "you", "name", "your", "files", "and", "functions", "as", "specified",
            "in", "this", "document", "if", "you", "do", "not", "it", "will", "make", "your", "submission", "difficult",
            "to", "mark", "and", "you", "will", "be", "penalised",

            "submission", "requirement",

            "you", "will", "submit", "a", "single", "python", "file", "assignment", "two", "dot", "py", "moodle",
            "will", "not", "accept", "submissions", "of", "other", "file", "types",

            "plagiarism",

            "the", "assignments", "will", "be", "checked", "for", "plagiarism", "using", "an", "advanced", "plagiarism",
            "detector", "in", "previous", "semesters", "many", "students", "were", "detected", "by", "the",
            "plagiarism", "detector", "and", "almost", "all", "got", "zero", "mark", "for", "the", "assignment", "or",
            "even", "zero", "marks", "for", "the", "unit", "as", "penalty", "and", "as", "a", "result", "the", "large",
            "majority", "of", "those", "students", "failed", "the", "unit", "helping", "others", "to", "solve", "the",
            "assignment", "is", "not", "accepted", "please", "do", "not", "share", "your", "solutions", "partially",
            "or", "completely", "to", "others", "using", "contents", "from", "the", "internet", "books", "etc",
            "without", "citing", "is", "plagiarism", "if", "you", "use", "such", "content", "as", "part", "of", "your",
            "solution", "and", "properly", "cite", "it", "it", "is", "not", "plagiarism", "but", "you", "wouldnt", "be",
            "getting", "any", "marks", "that", "are", "possibly", "assigned", "for", "that", "part", "of", "the",
            "task", "as", "it", "is", "not", "your", "own", "work",

            "the", "use", "of", "generative", "ai", "and", "similar", "tools", "is", "not", "allowed", "in", "this",
            "unit",

            "end", "of", "page", "one",

            "learning", "outcomes",

            "this", "assignment", "achieves", "the", "learning", "outcomes", "of",
            "one", "analyse", "general", "problem", "solving", "strategies", "and", "algorithmic", "paradigms", "and",
            "apply", "them", "to", "solving", "new", "problems",
            "two", "", "prove", "correctness", "of", "programs", "analyse", "their", "space", "and", "time",
            "complexities",
            "three", "compare", "and", "contrast", "various", "abstract", "data", "types", "and", "use", "them",
            "appropriately",
            "four", "develop", "and", "implement", "algorithms", "to", "solve", "computational", "problems",

            "in", "addition", "you", "will", "develop", "the", "following", "employability", "skills",
            "text", "comprehension",
            "designing", "test", "cases",
            "ability", "to", "follow", "specifications", "precisely",

            "assignment", "timeline",

            "in", "order", "to", "be", "successful", "in", "this", "assessment", "the", "following", "steps", "are",
            "provided", "as", "a", "suggestion", "this", "is", "an", "approach", "which", "will", "be", "useful", "to",
            "you", "both", "in", "future", "units", "and", "in", "industry",

            "planning",

            "one", "read", "the", "assignment", "specification", "as", "soon", "as", "possible", "and", "write", "out",
            "a", "list", "of", "questions", "you", "have", "about", "it",
            "two", "try", "to", "resolve", "these", "questions", "by", "viewing", "the", "faq", "on", "ed", "or", "by",
            "thinking", "through", "the", "problems", "over", "time",
            "three", "as", "soon", "as", "possible", "start", "thinking", "about", "the", "problems", "in", "the",
            "assignment",
            "it", "is", "strongly", "recommended", "that", "you", "do", "not", "write", "code", "until", "you", "have",
            "a", "solid", "feeling", "for", "how", "the", "problem", "works", "and", "how", "you", "will", "solve",
            "it",
            "four", "writing", "down", "small", "examples", "and", "solving", "them", "by", "hand", "is", "an",
            "excellent", "tool", "for", "coming", "to", "a", "better", "understanding", "of", "the", "problem",
            "as", "you", "are", "doing", "this", "you", "will", "also", "get", "a", "feel", "for", "the", "kinds", "of",
            "edge", "cases", "your", "code", "will", "have", "to", "deal", "with",
            "five", "write", "down", "a", "highlevel", "description", "of", "the", "algorithm", "you", "will", "use",
            "six", "determine", "the", "complexity", "of", "your", "algorithm", "idea", "ensuring", "it", "meets",
            "the", "requirements",

            "end", "of", "page", "two",

            "implementing",

            "one", "think", "of", "test", "cases", "that", "you", "can", "use", "to", "check", "if", "your",
            "algorithm", "works",
            "use", "the", "edge", "cases", "you", "found", "during", "the", "previous", "phase", "to", "inspire",
            "your", "test", "cases",
            "it", "is", "also", "a", "good", "idea", "to", "generate", "large", "random", "test", "cases",
            "sharing", "test", "cases", "is", "allowed", "as", "it", "is", "not", "helping", "solve", "the",
            "assignment",
            "two", "code", "up", "your", "algorithm", "remember", "decomposition", "and", "comments", "and", "test",
            "it", "on", "the", "tests", "you", "have", "thought", "of",
            "three", "try", "to", "break", "your", "code", "think", "of", "what", "kinds", "of", "inputs", "you",
            "could", "be", "presented", "with", "which", "your", "code", "might", "not", "be", "able", "to", "handle",
            "large", "inputs",
            "small", "inputs",
            "inputs", "with", "strange", "properties",
            "what", "if", "everything", "is", "the", "same",
            "what", "if", "everything", "is", "different",
            "etc",

            "before", "submission",

            "make", "sure", "that", "the", "inputoutput", "format", "of", "your", "code", "matches", "the",
            "specification",
            "make", "sure", "your", "filenames", "match", "the", "specification",
            "make", "sure", "your", "functions", "are", "named", "correctly", "and", "take", "the", "correct", "inputs",
            "remove", "print", "statements", "and", "test", "code", "from", "the", "file", "you", "are", "going", "to",
            "submit",

            "end", "of", "page", "three",

            "documentation",

            "for", "this", "assignment", "and", "all", "assignments", "in", "this", "unit", "you", "are", "required",
            "to", "document", "and", "comment", "your", "code", "appropriately", "whilst", "part", "of", "the", "marks",
            "of", "each", "question", "are", "for", "documentation", "there", "is", "a", "baseline", "level", "of",
            "documentation", "you", "must", "have", "in", "order", "for", "your", "code", "to", "receive", "marks",
            "in", "other", "words",
            "insufficient", "documentation", "might", "result", "in", "you", "getting", "zero", "for", "the", "entire",
            "question", "for", "which", "it", "is", "insufficient",
            "this", "documentationcommenting", "must", "consist", "of", "but", "is", "not", "limited", "to",
            "for", "each", "function", "highlevel", "description", "of", "that", "function", "this", "should", "be",
            "a", "two", "or", "three", "sentence", "explanation", "of", "what", "this", "function", "does",
            "your", "main", "function", "in", "the", "assignment", "should", "contain", "a", "generalised",
            "description", "of", "the", "approach", "your", "solution", "uses", "to", "solve", "the", "assignment",
            "task",
            "for", "each", "function", "specify", "what", "the", "input", "to", "the", "function", "is", "and", "what",
            "output", "the", "function", "produces", "or", "returns", "if", "appropriate",
            "for", "each", "function", "the", "appropriate", "big", "oh", "or", "big", "theta", "time", "and", "space",
            "complexity", "of", "that", "function", "in", "terms", "of", "the", "input", "size", "make", "sure", "you",
            "specify", "what", "the", "variables", "involved", "in", "your", "complexity", "refer", "to", "remember",
            "that", "the", "complexity", "of", "a", "function", "includes", "the", "complexity", "of", "any",
            "function", "calls", "it", "makes",
            "within", "functions", "comments", "where", "appropriate", "generally", "speaking", "you", "would",
            "comment", "complicated", "lines", "of", "code", "which", "you", "should", "try", "to", "minimise", "or",
            "a", "large", "block", "of", "code", "which", "performs", "a", "clear", "and", "distinct", "task", "often",
            "blocks", "like", "this", "are", "good", "candidates", "to", "be",
            "their", "own", "functions",

            "a", "suggested", "function", "documentation", "layout", "would", "be", "as", "follows",

            "def", "my", "underscore", "function", "left", "bracket", "argv", "one", "comma", "argv", "two", "right",
            "bracket", "colon",
            "start", "of", "comment", "block",
            "function", "description",
            "approach", "description", "if", "main", "function",
            "input",
            "argv", "one",
            "argv", "two",
            "output", "return", "or", "postcondition",
            "time", "complexity",
            "aux", "space", "complexity",
            "end", "of", "comment", "block",
            "then", "write", "your", "codes", "here",

            "there", "is", "a", "documentation", "guide", "available", "on", "moodle", "in", "the", "assignment",
            "section", "which", "contains", "a", "demonstration", "of", "how", "to", "document", "code", "to", "the",
            "level", "required", "in", "the", "unit",

            "end", "of", "page", "four"

        ]
        trie = CatsTrie(sentences)
        self.assertTrue(trie.autoComplete("") == "the")
        self.assertTrue(trie.autoComplete("a") == "and")
        self.assertTrue(trie.autoComplete("b") == "be")
        self.assertTrue(trie.autoComplete("c") == "code")
        self.assertTrue(trie.autoComplete("d") == "documentation")
        self.assertTrue(trie.autoComplete("e") == "end")
        self.assertTrue(trie.autoComplete("f") == "for")
        self.assertTrue(trie.autoComplete("g") == "getting")
        self.assertTrue(trie.autoComplete("h") == "have")
        self.assertTrue(trie.autoComplete("i") == "is")
        self.assertTrue(trie.autoComplete("j") == None)
        self.assertTrue(trie.autoComplete("k") == "kinds")
        self.assertTrue(trie.autoComplete("l") == "late")
        self.assertTrue(trie.autoComplete("m") == "make")
        self.assertTrue(trie.autoComplete("n") == "not")
        self.assertTrue(trie.autoComplete("o") == "of")
        self.assertTrue(trie.autoComplete("p") == "plagiarism")
        self.assertTrue(trie.autoComplete("q") == "question")
        self.assertTrue(trie.autoComplete("r") == "required")
        self.assertTrue(trie.autoComplete("s") == "slash")
        self.assertTrue(trie.autoComplete("t") == "the")
        self.assertTrue(trie.autoComplete("u") == "unit")
        self.assertTrue(trie.autoComplete("v") == "variables")
        self.assertTrue(trie.autoComplete("w") == "will")
        self.assertTrue(trie.autoComplete("x") == None)
        self.assertTrue(trie.autoComplete("y") == "you")
        self.assertTrue(trie.autoComplete("z") == "zero")
        self.assertTrue(trie.autoComplete("ab") == "about")
        self.assertTrue(trie.autoComplete("ac") == "accept")
        self.assertTrue(trie.autoComplete("ad") == "addition")
        self.assertTrue(trie.autoComplete("ae") == "aedt")
        self.assertTrue(trie.autoComplete("ai") == "ai")
        self.assertTrue(trie.autoComplete("al") == "algorithm")
        self.assertTrue(trie.autoComplete("an") == "and")
        self.assertTrue(trie.autoComplete("ap") == "appropriate")
        self.assertTrue(trie.autoComplete("ar") == "are")
        self.assertTrue(trie.autoComplete("as") == "as")
        self.assertTrue(trie.autoComplete("at") == "at")
        self.assertTrue(trie.autoComplete("au") == "automated")
        self.assertTrue(trie.autoComplete("av") == "available")
        self.assertTrue(trie.autoComplete("ba") == "baseline")
        self.assertTrue(trie.autoComplete("be") == "be")
        self.assertTrue(trie.autoComplete("bi") == "big")
        self.assertTrue(trie.autoComplete("bl") == "block")
        self.assertTrue(trie.autoComplete("bo") == "books")
        self.assertTrue(trie.autoComplete("br") == "bracket")
        self.assertTrue(trie.autoComplete("bu") == "but")
        self.assertTrue(trie.autoComplete("by") == "by")
        self.assertTrue(trie.autoComplete("ca") == "cases")
        self.assertTrue(trie.autoComplete("ch") == "check")
        self.assertTrue(trie.autoComplete("ci") == "cite")
        self.assertTrue(trie.autoComplete("cl") == "clayton")
        self.assertTrue(trie.autoComplete("co") == "code")
        self.assertTrue(trie.autoComplete("cr") == "criteria")
        self.assertTrue(trie.autoComplete("da") == "days")
        self.assertTrue(trie.autoComplete("de") == "description")
        self.assertTrue(trie.autoComplete("di") == "different")
        self.assertTrue(trie.autoComplete("do") == "documentation")
        self.assertTrue(trie.autoComplete("du") == "during")
        self.assertTrue(trie.autoComplete("ea") == "each")
        self.assertTrue(trie.autoComplete("ed") == "edge")
        self.assertTrue(trie.autoComplete("eg") == "eg")
        self.assertTrue(trie.autoComplete("em") == "employability")
        self.assertTrue(trie.autoComplete("en") == "end")
        self.assertTrue(trie.autoComplete("et") == "etc")
        self.assertTrue(trie.autoComplete("ev") == "everything")
        self.assertTrue(trie.autoComplete("ex") == "examples")
        self.assertTrue(trie.autoComplete("fa") == "failed")
        self.assertTrue(trie.autoComplete("fe") == "feel")
        self.assertTrue(trie.autoComplete("fi") == "file")
        self.assertTrue(trie.autoComplete("fo") == "for")
        self.assertTrue(trie.autoComplete("fr") == "from")
        self.assertTrue(trie.autoComplete("fu") == "function")
        self.assertTrue(trie.autoComplete("ge") == "getting")
        self.assertTrue(trie.autoComplete("go") == "good")
        self.assertTrue(trie.autoComplete("gu") == "guide")
        self.assertTrue(trie.autoComplete("ha") == "have")
        self.assertTrue(trie.autoComplete("he") == "helping")
        self.assertTrue(trie.autoComplete("hi") == "highlevel")
        self.assertTrue(trie.autoComplete("ho") == "how")
        self.assertTrue(trie.autoComplete("ht") == "https")
        self.assertTrue(trie.autoComplete("hy") == "hyphen")
        self.assertTrue(trie.autoComplete("id") == "idea")
        self.assertTrue(trie.autoComplete("if") == "if")
        self.assertTrue(trie.autoComplete("im") == "implement")
        self.assertTrue(trie.autoComplete("in") == "in")
        self.assertTrue(trie.autoComplete("is") == "is")
        self.assertTrue(trie.autoComplete("it") == "it")
"""


if __name__ == '__main__':
    connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),(0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
    maxIn = [5000, 3000, 3000, 3000, 2000]
    maxOut = [5000, 3000, 3000, 2500, 1500]
    origin = 0
    targets = [4, 2]
    max = maxThroughput(connections,maxIn,maxOut,origin,targets)
    #print(max)

    sentences = ["abc", "abazacy", "dbcef", "xzz", "gdbc", "abazacy", "xyz", "abazacy", "dbcef", "xyz", "xxx","xzz"]
    mycattrie = CatsTrie(sentences)
    ans = mycattrie.autoComplete('x')
    #print(ans)




