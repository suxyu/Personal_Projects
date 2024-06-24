from typing import List
from collections import defaultdict, deque
import copy

class Solution:
    def dfs(self, graph, start, end,path,visited):
        
        visited.append(start)

        if start ==end:
            path.append(visited[:])

        for n in graph[start]:
            if n not in visited:
                self.dfs(graph,n,end,path,visited[:])

        return path
        

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                d[word[:i]+"*"+word[i+1:]].append(word)

        print(d)

        if endWord not in wordList:
            return []
        else:
            queue = [beginWord]
            visited = set()
            visited.add(beginWord)
            tree = defaultdict(list)

            

            while queue:
                node = queue.pop(0)
                print("node:",node)
                if node==endWord:
                    print("found")
                
                neighbor =[]
                for i in range(len(node)):
                    check = node[:i]+"*"+node[i+1:]
                    print(d)
                    print(f"d[{check}]:",d[check])
                    if check in d.keys() and d[check]!=[]:
                        print(f"d[{check}]:",d[check])
                        temp = copy.deepcopy(d[check])
                        
                        
                        while temp:############the problem now is the tree not correct!!!!!!!!!!!!
                            n = temp.pop(0)
                            print(n)
                            neighbor.append(n)
                            if n not in visited:
                                
                                
                                visited.add(n)
                                queue.append(n)
                                print("visited:",visited,"queue:",queue)
                            
                            
                tree[node]=neighbor[:]
                
                print("after loop:","node:",node,"neighbor:",neighbor,"tree:",tree)

        

            paths =self.dfs(tree,beginWord,endWord,path=[],visited=[])
            min_path = 1000
            res =[]
            for path in paths:
                length = len(path)
                if length<=min_path:
                    min_path = length

            for path in paths:
                if len(path)==min_path:
                    res.append(path)

            return res
                
if __name__ == "__main__":
    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot","dot","dog","lot","log","cog"]
    s = Solution()
    print(s.findLadders(beginWord,endWord,wordList))