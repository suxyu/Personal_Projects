#author:xiaoyu sun
#student id:32950683
#####################the following is problem1#################
import math
import heapq

"""
Approach description:
first calculate the minimum time taken using dijkstra without any passenagers, call it min

then construct a reverse graph and calculate the dist_reverse[] from end to start
then go through all passenagers, calculate the forward dist_forward[p] + dist_reverse[p], if it is smaller than min, 
update min, until find the smallest time taken considering having passenagers
"""

def find_v(r):
    """
    Function description:
    from roads list to find the numner of vertex of the graph

    :Input:
    argv1:r (roads list)

    :Output:
    v the number of vertex, |L| in this problem

    :Time complexity: O(|R|)
    :Aux space complexity:O(1)
    """
    v = 0  # number of vertices
    for item in r:
        if item[0] > v:
            v = item[0]

    for item in r:
        if item[1] > v:
            v = item[1]
    v = v + 1
    return v

def construct_graph(r,s,d):#road,start,end
    """
    Function description:
    construct the linked list representation of the graph

    :Input:
    argv1:r(roads list)
    argv2:s(start)
    argv3:d(end)

    :Output:
    argv1:graph(forward graph in linked list representation
    argv2:e(the number of edges)

    :Time complexity: O(|R|)
    :Aux space complexity:O(|L|)
    """
    e = len(r)

    v = find_v(r)

    graph = [[] for _ in range(v)]

    for item in r:
        position = item[0]# in roads list, each element is of form (a,b,c,d), use a to construt forward graph
        graph[position].append(item)

    return graph,e


def construct_reverse_graph(r,s,d):#road,start,end
    """
     Function description:
     construct the linked list representation of the reverse graph

     :Input:
     argv1:r(roads list)
     argv2:s(start)
     argv3:d(end)

     :Output:
     argv1:graph(reverse graph in linked list representation)
     argv2:e(the number of edges)

     :Time complexity: O(|R|)
     :Aux space complexity:O(|L|)
     """

    e = len(r)

    v = find_v(r)

    graph = [[] for _ in range(v)]

    for item in r:
        position = item[1]#in roads list, it is of form (a,b,c,d), use b to construct the reverse graph
        graph[position].append(item)

    return graph,e

def relax_p(g_,current_,adj_,dist_,pred_,tumple,alone,q):#D is the distance between cuurent and adj
    """
     Function description:
     relax function with additional feature that can shift distance between with/without passenages

     :Input:
     argv1:g_(linked list reprensentation of the graph)
     argv2:current_(the node currently at)
     argv3:adj_(the adjecenct node of current node)
     argv4:dist_(the distance list)
     argv5:pred_(the predict list)
     argv6:tumple(the element in roads list, in the form (a,b,c,d))
     argv7:alone(True for traveling alone, False for traveling with pasenagers)
     argv8:q(the priority queue in heap format)

     :Output:
     argv1:dist_(updated distance list)
     argv2:pred_(updated predict list)
     argv3:q(updated priority queue)

     :Time complexity: O(log(|L|))
     :Aux space complexity:O(1)
     """
    if alone == True:# if tumple is True, is the third number for distance, if False, use the fourth number
        D = tumple[2]
    if alone == False:
        D = tumple[3]

    #relaxation function
    if dist_[adj_]>dist_[current_] + D:
        dist_[adj_]=dist_[current_]+D
        pred_[adj_] = current_
        heapq.heappush(q,[dist_[adj_],adj_])#time complexity of heappush is O(log(|L|))

    return dist_,pred_,q


def dijkstra(g, s, e,alone):  # graph,start,end,edge,alon. if alone is true is the forward weight, if the alone is false, use the reverse weight
    """
     Function description:
     dijkstra algorithm

     :Input:
     argv1:g(graph in linked list format)
     argv2:s(start)
     argv3:e(number of edge)
     argv4:alone(True for alone, Fasle for traveling with passenager)

     :Output:
     argv1:dist(distance list)
     argv2:pred(predict list)

     :Time complexity: O(|R|log|L|)
     :Aux space complexity:O(|L|)
     """

    inf = math.inf
    v = len(g)
    Q = []

    dist = [inf for _ in range(v)]
    dist[s] = 0
    pred = [None for _ in range(v)]

    current = s
    heapq.heappush(Q,[dist[s],s])


    heapq.heapify(Q)

    while Q:
        key, current = heapq.heappop(Q)#time complexity of heappop is O(log(|L|))
        if dist[current] == key:#do not process out of date entry
            for item in g[current]:
                #if travel alone, use forward graph, if travel with passenage, use reverse graph
                if alone == True:
                    adj = item[1]
                if alone == False:
                    adj = item[0]
                dist, pred,Q = relax_p(g, current, adj, dist, pred, item,alone,Q)#time complaxity O(log|L|)



    return dist, pred

def find_sequence(pred_f,pred_r,p,start,end):#find the path
    """
     Function description:
     find the sequence if taking passenagers

     :Input:
     argv1:pred_f(forward predict list)
     argv2:pred_r(reverse predict list)
     argv3:p(the location of passeenage, it is an index)
     argv4:start
     argv5:end

     :Output:
     argv1:sequence(the total path sequence if taking a passenager)

     :Time complexity: O(|L|)
     :Aux space complexity:O(|L|)
     """
    sequence =[]
    current =p
    while pred_f[current] != None:#use forward predict list to find sequene before pickup location
        sequence.append(current)
        current = pred_f[current]

    sequence = sequence+[start]
    sequence.reverse()# takes O(|L|)


    current = p
    while pred_r[current] != None:#use reverse predict list to find path sequence after pickup location
        sequence.append(pred_r[current])
        current = pred_r[current]

    return sequence


def find_sequence_dijkastra(pred,start,end):
    """

     Function description:
     find the sequence from start to end using dijkastra

     :Input:
     argv1:pred(predict list)
     argv2:start
     argv3:end

     :Output:
     argv1:sequence(the sequence in list format from start to end using dikstra)


     :Time complexity: O(|L|)
     :Aux space complexity:O(|L|)
     """
    sequence =[]
    current =end
    while pred[current] !=None:
        sequence.append(current)
        current = pred[current]

    sequence = sequence + [start]
    sequence.reverse()

    return sequence


def optimalRoute(s,d,p,r): #start,end,passenage,roads
    """
     Function description:
     the main function to find the optimal route for the question 1

     Approach description:
     first calculate the minimum time taken using dijkstra without any passenagers, call it min

     then construct a reverse graph and calculate the dist_reverse[] from end to start
     then go through all passenagers, calculate the forward dist_forward[p] + dist_reverse[p], if it is smaller than min,
     update min, until find the smallest time taken considering having passenagers

     :Input:
     argv1:s(start)
     argv2:d(end)
     argv3:p(passenager list)
     argv4:r(roads list)

     :Output:
     argv1:sequence(the optimal path sequence)

     :Time complexity: O(|R|log(|L|))
     :Aux space complexity:O(|L|)
     """
    graph, edge = construct_graph(r, s, d)#construct forward graph

    graph_reverse, edge = construct_reverse_graph(r,s,d)#construct reverse graph


    dist_forward,pred_forward = dijkstra(graph,s,edge,True)#find distance ,prediction list of forward graph


    sequence = find_sequence_dijkastra(pred_forward,s,d)#assume the optimal path sequence is the path without passenagers


    dist_reverse,pred_reverse  = dijkstra(graph_reverse,d,edge,False)#find distance, prediction list of reverse graph


    min = dist_forward[d]# this is the shortest time for dirving alone

    for item in p:#go through all passenagers and calculate the time from start to pickup location and the time from pickup location to end.
        if (dist_forward[item] + dist_reverse[item]) < min:
            min = dist_forward[item] + dist_reverse[item]
            sequence = find_sequence(pred_forward, pred_reverse, item, s, d)


    return sequence



"""
####the following is the test case
import unittest

class TestA1(unittest.TestCase):
    def test1(self):
        start = 0
        end = 4
        passengers = [2,1]
        roads = [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10),
             (2, 4, 30, 25), (2, 0, 2, 2), (0, 1, 10, 10), (1, 4, 30, 20)]
        result = [0, 3, 2, 0, 3, 4]  # Optimal route is 11 minutes
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test2(self):
        start = 0
        end = 3
        passengers = [2]
        roads = [(0, 1, 10, 10), (1, 3, 1, 1), (0, 2, 2, 2)]
        result = [0, 1, 3]  # Optimal route is 11 minutes
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test3(self):
        start = 0
        end = 3
        passengers = [2]
        roads = [(0, 1, 10, 10), (1, 3, 1, 1), (0, 2, 2, 2), (2, 3, 60, 60)]
        result = [0, 1, 3]  # Optimal route is 11 minutes
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test4(self):
        start = 0
        end = 3
        passengers = [2, 1]
        roads = [(0, 1, 10, 10), (1, 3, 100, 1), (0, 2, 2, 2), (2, 3, 60, 60)]
        result = [0, 1, 3]  # Optimal route is 11 minutes
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test5(self):
        start = 0
        end = 3
        passengers = [2, 1]
        roads = [(0, 1, 10, 10), (1, 3, 100, 1), (0, 2, 2, 2), (2, 3, 60, 60), (2, 1, 6, 3)]
        result = [0, 2, 1, 3]  # Optimal route is 6 minutes
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def some_very_long_journey(self):
        start = 54
        end = 62
        passengers = [29, 63, 22, 18, 2, 23, 48, 41, 15, 31, 13, 4, 24, 16, 27, 17, 50, 67, 37, 58, 28, 64, 35, 10, 68,
                      38, 59, 26, 69, 43, 44, 30, 46, 7]
        roads = [
            (31, 45, 23, 12), (48, 3, 14, 7), (58, 50, 25, 10),
            (5, 3, 26, 23), (12, 32, 29, 3), (65, 4, 16, 16),
            (13, 46, 14, 13), (63, 29, 10, 2), (19, 56, 30, 19),
            (52, 47, 19, 12), (47, 52, 12, 8), (30, 42, 22, 19),
            (46, 60, 17, 17), (54, 22, 8, 7), (19, 8, 23, 10),
            (33, 51, 22, 5), (12, 17, 20, 5), (64, 62, 22, 18),
            (66, 25, 28, 10), (48, 19, 23, 8), (36, 13, 22, 19),
            (26, 48, 6, 3), (31, 30, 26, 9), (24, 29, 22, 11),
            (23, 36, 27, 11), (59, 37, 16, 10), (60, 44, 12, 8),
            (40, 7, 18, 1), (22, 3, 13, 12), (36, 35, 15, 15),
            (43, 2, 23, 6), (29, 27, 27, 6), (34, 0, 17, 4),
            (52, 50, 13, 4), (27, 23, 15, 1), (15, 10, 7, 6),
            (36, 65, 23, 1), (41, 64, 27, 8), (45, 34, 12, 1),
            (51, 24, 12, 10), (16, 12, 29, 7), (9, 67, 25, 24),
            (49, 38, 16, 4), (38, 7, 10, 1), (50, 13, 23, 16),
            (5, 33, 27, 10), (23, 42, 29, 15), (9, 2, 13, 7),
            (59, 52, 23, 17), (59, 54, 8, 6), (1, 8, 10, 8),
            (33, 30, 15, 2), (6, 26, 18, 6), (39, 57, 13, 12),
            (54, 26, 13, 9), (57, 41, 4, 4), (37, 66, 16, 12),
            (36, 9, 12, 5), (2, 68, 7, 3), (69, 28, 18, 2),
            (44, 1, 14, 3), (48, 9, 6, 4), (17, 38, 13, 1),
            (61, 49, 4, 4), (9, 10, 6, 3), (46, 37, 21, 8),
            (23, 53, 21, 8), (7, 24, 28, 26), (62, 20, 22, 7),
            (1, 18, 10, 1), (7, 41, 9, 1), (13, 18, 6, 4),
            (25, 21, 21, 3), (1, 61, 21, 16), (49, 40, 13, 5),
            (19, 25, 11, 10), (62, 50, 5, 5), (33, 46, 10, 9),
            (28, 25, 14, 6), (56, 51, 6, 4), (18, 19, 15, 1),
            (30, 9, 23, 13), (60, 21, 23, 7), (52, 37, 16, 6),
            (50, 42, 11, 4)
        ]
        # Optimal route is 198 mins
        result = [54, 26, 48, 19, 56, 51, 24, 29, 27, 23, 36, 13, 46, 60, 44, 1, 61, 49, 38, 7, 41, 64, 62]
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test_no_passenger_1(self):
        start = 4
        end = 0
        passengers = []
        roads = [
            (0, 1, 28, 22),
            (3, 2, 21, 10),
            (4, 1, 26, 20),
            (1, 3, 5, 3),
            (0, 4, 24, 13),
            (2, 1, 26, 15),
            (2, 0, 26, 26)
        ]
        result = [4, 1, 3, 2, 0]  # Optimal route is 78 mins
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test_no_passenger_2(self):
        start = 1
        end = 2
        passengers = []
        roads = [
            (3, 4, 24, 10),
            (4, 1, 16, 6),
            (0, 2, 28, 14),
            (1, 3, 27, 12),
            (4, 0, 5, 4),
            (2, 4, 15, 9)
        ]
        result = [1, 3, 4, 0, 2]  # Optimal route is 84 mins
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)

    def test_take_previous_locations(self):
        start = 4
        end = 9
        passengers = [2, 6, 0]
        roads = [
            (4, 6, 30, 18),
            (3, 1, 8, 1),
            (9, 1, 9, 5),
            (1, 9, 30, 2),
            (8, 5, 12, 12),
            (8, 9, 8, 6),
            (1, 8, 25, 2),
            (2, 4, 4, 2),
            (6, 0, 25, 5),
            (4, 3, 6, 6),
            (1, 2, 15, 7)
        ]
        result = [4, 3, 1, 2, 4, 3, 1, 9]  # Optimal route is 40 mins
        self.assertEqual(optimalRoute(start, end, passengers, roads), result)
"""





############################the following is problem 2##############################3
import math

"""
Approach description:
update the occupancy matrix:

for matrix[i][j]:
     find the minimum of matrix[i-1][j],matrix[i-1][j-1],matrix[i-1][j+1], and add to matrix[i][j]
     note: be careful about the  j+1 and j-1  and make sure they are within boundary

after update, the minimul value in the last row is the expected answer
assume it is matrix[a][b]
backward find the index of minimul of matrix[a-1][b], matrix[a-1][b-1],matrix[a-1][b+1],
repeat the process to find the location list
"""


def minimum(m, i_, j_):
    """
     Function description:
     output the minimum of m[i_][j_],m[i_][j_-1],m[i_][j_+1]

     :Input:
     argv1:m the occupancy probability matrix
     argv2:i_ the row index
     argv3:j_ the column index

     :Output:
     min the minimum of m[i_][j_],m[i_][j_-1],m[i_][j_+1]


     :Time complexity: O(1)
     :space complexity: O(nm)
     :Aux space complexity:O(1)
     """
    num_of_colum = len(m[0])
    if j_ == 0:
        a = math.inf
        b = m[i_][j_]
        c = m[i_][j_ + 1]


    elif j_ == num_of_colum - 1:
        a = m[i_][j_ - 1]
        b = m[i_][j_]
        c = math.inf
    else:
        a = m[i_][j_ - 1]
        b = m[i_][j_]
        c = m[i_][j_ + 1]

    min = a
    if b < a:
        min = b
        if c < b:
            min = c
    if c < a:
        min = c
        if b < c:
            min = b

    return min


def update(matrix):
    """
     Function description:
     update the occupancy matrix
     for matrix[i][j]:
     find the minimum of matrix[i-1][j],matrix[i-1][j-1],matrix[i-1][j+1], and add to matrix[i][j]

     :Input:
     argv1:matrix the occupancy probability matrix


     :Output:
     matrix the updated matrix


     :Time complexity: O(nm)
     :space complexity: O(nm)
     :Aux space complexity:O(1)
     """

    row = len(matrix)
    column = len(matrix[0])
    # print(matrix)
    min = math.inf

    for i in range(row - 1):
        for j in range(column):
            if j != 0 and j != column - 1:  # boundary condition check
                matrix[i + 1][j] = matrix[i + 1][j] + minimum(matrix, i, j)
            if j == 0:  # boundary condition check
                matrix[i + 1][j] = matrix[i + 1][j] + minimum(matrix, i, j)

            if j == column - 1:  # boundary condition check
                matrix[i + 1][j] = matrix[i + 1][j] + minimum(matrix, i, j)

    return matrix


def select_sections(matrix):
    """
      Function description:
      this is the main function, output the required value


      :Input:
      argv1:matrix the occupancy probability matrix


      :Output:
      [the_minimum_total_occupancy,the_list_of_selections_locations]


      :Time complexity: O(nm)
      :space complexity: O(nm)
      :Aux space complexity:O(m) number of column
      """

    new = update(matrix)  # update the matrix
    row = len(new)
    column = len(matrix[0])
    s = []  # to store the selection location list

    total_min_val = min(new[row - 1])  # total minimum occupancy is the mini of the last row of the updated matrix
    b = new[row - 1].index(total_min_val)  # b here represent the column choice of the selection location
    s.append((row - 1, b))

    for i in range(row - 1):  # complexity: O(m)
        if b != 0 and b != column - 1:  # boundary condition check
            min_val = min(new[row - 2 - i][b], new[row - 2 - i][b + 1],
                          new[row - 2 - i][b - 1])  # complexity of min: O(1)
            b = new[row - 2 - i].index(min_val)
            s.append((row - 2 - i, b))

        elif b == 0:  # boundary condition check
            min_val = min(new[row - 2 - i][b], new[row - 2 - i][b + 1])
            b = new[row - 2 - i].index(min_val)
            s.append((row - 2 - i, b))

        elif b == column - 1:  # boundary condition check
            min_val = min(new[row - 2 - i][b], new[row - 2 - i][b - 1])
            b = new[row - 2 - i].index(min_val)
            s.append((row - 2 - i, b))

    s.reverse()  # complexity O(m)

    return [total_min_val, s]


"""
####test case######
import unittest

class TestA2(unittest.TestCase):
    def test_which_to_go(self):
        occupancy_probability = [
        [0, 76, 38, 2],
        [1, 94, 54, 1],
        [2, 86, 86, 99],
        [3, 0, 0, 0],
        [99, 99, 99, 0]]
        expec_res_1 = [87, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)]]
        #print(select_sections(occupancy_probability))
        self.assertTrue(select_sections(occupancy_probability) == expec_res_1)

    def test_selectsections_7(self):
        occupancy_probability = [
            [19, 76, 38, 22, 0],
            [56, 20, 54, 0, 34],
            [71, 86, 0, 99, 89],
            [81, 0, 82, 22, 45],
            [0, 22, 22, 93, 23]
        ]
        expected = [0, [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]]
        my_res = select_sections(occupancy_probability)
        self.assertTrue(my_res == expected)

    def test_selectsections_8(self):
        occupancy_probability = [
            [19, 76, 38, 22, 0],
            [56, 20, 54, 0, 34],
            [71, 86, 0, 99, 89],
            [81, 34, 82, 0, 45],
            [62, 22, 22, 93, 0]
        ]
        expected = [0, [(0, 4), (1, 3), (2, 2), (3, 3), (4, 4)]]
        my_res = select_sections(occupancy_probability)
        self.assertTrue(my_res == expected)

    def test_selectsections_6(self):
        occupancy_probability = [
            [19, 76, 38, 22],
            [56, 20, 54, 68],
            [71, 86, 15, 99],
            [81, 82, 82, 22],
            [36, 22, 22, 93]
        ]
        expected = [98, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 2)]]
        my_res = select_sections(occupancy_probability)
        self.assertTrue(my_res == expected)
"""



if __name__ == "__main__":
    roads = [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10),
             (2, 4, 30, 25), (2, 0, 2, 2), (0, 1, 10, 10), (1, 4, 30, 20)]
    start = 0
    end =4
    passengers = [2,1]
    result = optimalRoute(start,end,passengers,roads)
    #print(result)
    #unittest.main()

    occupancy_probability = [
        [31, 54, 94, 34, 12],
        [26, 25, 24, 16, 87],
        [39, 74, 50, 13, 82],
        [42, 20, 81, 21, 52],
        [30, 43, 19, 5, 47],
        [37, 59, 70, 28, 15],
        [2, 16, 14, 57, 49],
        [22, 38, 9, 19, 99]]
    m = select_sections(occupancy_probability)
    #print(m)
    # unittest.main()


