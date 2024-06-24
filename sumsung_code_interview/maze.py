def bfs_shortest_path(maze):
    #initialize queue with (0,0,0)// (x,y,distance)
    queue = [(0,0,0)]
    #initialize visited set (0,0)
    visited = set((0,0))

    while queue != []:
        x,y,d = queue.pop(0)
        #if we reach the end, return the distance
        if x == len(maze)-1 and y == len(maze[0])-1:
            return d
        #check the four directions
        for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx,ny = x+dx, y+dy
            #if the next position is in the maze and not visited
            if 0<=nx<len(maze) and 0<=ny<len(maze[0]) and (nx,ny) not in visited:
                #if the next position is not a wall
                if maze[nx][ny] == 0:
                    #add the next position to the queue
                    queue.append((nx,ny,d+1))
                    #add the next position to the visited set
                    visited.add((nx,ny))


    return -1






#make a 2 by 2 matrix
matrix = [[0,0,0,0,0,0,0,0],[0,0,0,1,1,1,1,0],[0,0,0,1,0,0,0,0],[1,1,0,1,0,1,1,0],[0,0,0,1,0,0,1,1],[0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0]]

print(bfs_shortest_path(matrix))