
class Solution:
    def __init__(self) -> None:
        self.max_d =0
    
    
    def find_longest_length(self, matrix, i ,j,visited=None):

        rows = len(matrix)
        cols = len(matrix[0])

        if visited is None:
            visited =[]

        visited.append((i,j))

        print(visited,len(visited),self.max_d)
        

        if len(visited)>self.max_d:
            #print(visited,len(visited),max_d)
            self.max_d = len(visited)

        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        
        for dir in directions:
            new_i = i+dir[0]
            new_j = j+dir[1]
            if new_i>=0 and new_i<rows and new_j>=0 and new_j<cols:
                if (new_i,new_j) not in visited and matrix[new_i][new_j]>matrix[i][j]:
                    path = visited[:]
                    self.find_longest_length(matrix,new_i,new_j,path)


        

        return self.max_d

        


            





    def longestIncreasingPath(self, matrix) -> int:

        rows = len(matrix)
        cols = len(matrix[0])
        max_d =0
        for i in range(rows):
            for j in range(cols):
                d = self.find_longest_length(matrix,i,j)
                if d>max_d:
                    max_d =d

        return max_d



if __name__ == "__main__":
    matrix=[[7,6,1,1],[2,7,6,0],[1,3,5,1],[6,6,3,2]]
    #print(matrix[2][1])
    solution = Solution()
    res = solution.find_longest_length(matrix,1,3)
    print(res)