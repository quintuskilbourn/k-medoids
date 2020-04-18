import itertools
import networkx as nx
import itertools
import heapq as hq
import timeit
import math
import random
import numpy as np
import operator

class Kmed:
    def __init__(self,G,k):
        self.dbQ = PQ()
#         self.optQ = PQ()
        self.lQ = PQ()
        self.G = G
        self.k=k
        self.dist = dict(nx.all_pairs_shortest_path_length(G))
        self.lowest_centrality = math.inf


    def density_edges(self,V,d):
        '''
        For creating number of edges based on density calculation
        '''
        return round((d/2)*(V*(V-1)))
    
    def makeClusters(self,medoids):
        '''
        forms clusters around medoids
        '''
        clusters = {med:[] for med in medoids}
        for n in self.G.nodes:
            clusters[min(medoids, key=lambda x:self.dist[n][x])].append(n)
        return clusters

    def find_medoid(self,nodes):
        '''
        finds medoid given subset of nodes
        '''
        return min(nodes, key=lambda n: sum(self.dist[n][u] for u in nodes))

    def p_j(self):
        '''
        Algorithm in Park and Jun paper
        '''
        G = self.G
        k = self.k
        starttime = timeit.default_timer()
        v = {}
        for n in G.nodes:
            v[n] = sum(self.dist[n][i]/sum(self.dist[i][j] for j in G.nodes) for i in G.nodes)
        medoids = [x[0] for x in sorted(v.items(), key=operator.itemgetter(1))[:k]]
        while(1):
            clusters = self.makeClusters(medoids)
            new_medoids = [self.find_medoid(clusters[med]) for med in medoids]
            if new_medoids == medoids:
                break
            medoids=new_medoids
        self.PJ_medoids = medoids
        self.PJ_centrality = self.distance_centrality(medoids)[0]
        return timeit.default_timer()-starttime

            

    def BUILD_mid(self):
        centre = min([(key,sum(val.values())) for key,val in self.dist.items()],key=operator.itemgetter(1))
        m = [centre[0]] * self.k
        TD = centre[1]
        return TD,tuple(m)
    
    
    def BUILD(self):
        TD=math.inf
        m=['']*self.k
        dist = self.dist
        for x in self.G.nodes:
            TDj=0
            for y in self.G.nodes:
                if y==x:
                    continue
                TDj+=dist[x][y]
            if TDj<TD:
                TD = TDj
                m[0] = x
        for i in range(1,self.k):
            dTD = math.inf
            for x in self.G.nodes:
                TDj=0
                for y in self.G.nodes:
                    if y==x:
                        continue
                    d = dist[x][y]-min([dist[mi][y] for mi in m if mi!=''])
                    if d<0:
                        TDj+=d
                if TDj<dTD:
                    dTD=TDj
                    xi=x
            TD+=dTD
            m[i] = xi
        return TD,tuple(m)
    
    def LAB(self):
        TD=math.inf
        m=['']*self.k
        samp = random.sample(self.G.nodes,10+int(np.log(len(self.G))))
        dist = self.dist
        for x in samp:
            TDj=0
            for y in samp:
                if y==x:
                    continue
                TDj+=dist[x][y]
            if TDj<TD:
                TD = TDj
                m[0] = x
        for i in range(1,self.k):
            dTD = math.inf
            samp = random.sample(set(self.G.nodes)-set(m),10+int(np.log(len(self.G))))
            for x in samp:
                TDj=0
                for y in samp:
                    if y==x:
                        continue
                    d = dist[x][y]-min([dist[mi][y] for mi in m if mi!=''])
                    if d<0:
                        TDj+=d
                if TDj<dTD:
                    dTD=TDj
                    xi=x
            TD+=dTD
            m[i] = xi
        return TD,tuple(m)

    def FASTPAM2(self,m,TD):
        dist=self.dist
        while 1:
            ind = dict(zip(m,range(self.k)))
            TDres = [0]*self.k
            x = [None]*self.k
            nearest={node:sorted([(dist[node][mn],mn) for mn in m])[:2] for node in self.G.nodes} #get nearest and second nearest medoid,distance
            for n in self.G.nodes():
                if n in m:
                    continue
                dTD = [0]*self.k
                for v in self.G.nodes():
                    dTD[ind[nearest[v][0][1]]] += min(dist[v][n],nearest[v][1][0]) - nearest[v][0][0] #pos is worse and neg is better
                    if dist[v][n]<nearest[v][0][0]:
                        for mn in m:
                            if mn==nearest[v][0][1]:
                                continue
                            dTD[ind[mn]]+=dist[v][n]-nearest[v][0][0]
                for i in range(self.k):
                    if dTD[i]<TDres[i]:
                        TDres[i] = dTD[i]
                        x[i] = n
            if min(TDres)>=0: #needs to separate from while loop below
                break
            mini = min(TDres)
            while mini<0:
                i = TDres.index(mini)
                m[i]=x[i]
                TD += mini
                TDres[i]=0
                nearest={node:sorted([(dist[node][mn],mn) for mn in m])[:2] for node in self.G.nodes}
                for j in range(self.k):
                    if TDres[j]<0:
                        TDres[j]=0
                        for n in self.G.nodes():
                            if nearest[n][0][1]==m[j]:
                                TDres[j]+= min(dist[x[j]][n],nearest[n][1][0])-nearest[n][0][0]
                            else:
                                TDres[j]+=min(dist[x[j]][n]-nearest[n][0][0],0)
                mini = min(TDres)
        return TD,tuple(m)

    def calc_bfs(self,start):
        '''
        This is the BFS which assigns vertices their level and counts number of vertices per level
        '''
        starttime = timeit.default_timer()
        nodeLevel={}
        for n,lvl in nx.single_source_shortest_path_length(self.G,start).items():
            try:
                nodeLevel[lvl].append(n)
            except:
                nodeLevel[lvl]=[n]
        return nodeLevel,timeit.default_timer()-starttime


    def calc_level_bound(self,nlvl):
        Q = self.lQ
        starttime = timeit.default_timer()
        '''
        diam^k calc of level-based lower bound for all combinations of levels
        '''
        def rec_level_bound(nlvl,bound,lcount,S,memo,Q,k=1):
            if type(S[k]) is int:
                if lcount[S[k-1]]>0: #cannot have more sources than nodes per level
                    S[k]=S[k-1]
                else:
                    S[k] = S[k-1]+1
                lcount[S[k]] -= 1
                while S[k]<len(nlvl):
                    newBound = bound           ## S needs a dud at the end and a static 0 to at the start
                    if S[k] != S[k-1]:
                        if (S[k-1],S[k]) in memo:
                            newBound += memo[(S[k-1],S[k])]
                        else:
                            m = (S[k]+S[k-1])//2 #m is the midpoint between two levels
                            count = 0
                            for x in range(S[k-1]+1,m+1):
                                count+=len(nlvl[x])*(x-S[k-1])
                            for x in range(m+1,S[k]):
                                count+=len(nlvl[x])*(S[k]-x)
                            count += len(nlvl[S[k]])
                            memo[(S[k-1],S[k])] = count
                            newBound += count
                    rec_level_bound(nlvl,newBound,lcount,S,memo,Q,k+1)
                    lcount[S[k]] +=1
                    S[k]+=1
                    lcount[S[k]] -= 1
            else:
                    count = 0
                    if (S[k-1],-1) in memo:
                        bound += memo[(S[k-1],-1)]
                    else:
                        count = 0
                        for i in range(S[k-1],len(nlvl)):  #not adding one to start of range because using S[k-1] old position
                            count += len(nlvl[i])*(i-S[k-1])
                        memo[(S[k-1],-1)] = count
                        bound += count
                    bound -= (len(S)-1)   #counted source nodes as 1 when should be 0, excluding 'F' entry for final segment
                    if bound<self.lowest_centrality:
                        Q.add_task(tuple(S[:-1]),bound)  #add zero to denote level bound

        S = [0 for i in range(self.k)]+['F']
        memo = {}
        lcount = [len(nlvl[lvl]) for lvl in nlvl.keys()]+[self.k] #keep track of how many source nodes there can be per level
        for i in range(len(nlvl)):  #all possible starting positions
            lcount[i] -= 1
            bound = 0
            for j in range(0,i):
                bound += len(nlvl[j])*(i-j)
            bound += len(nlvl[i])
            rec_level_bound(nlvl,bound,lcount,S,memo,Q)
            lcount[i]+=1
            S[0]+=1
        return timeit.default_timer()-starttime

    
    def db_v2(self,S,lvln,lvlBound): #needs some testing and comments
        Q = self.dbQ
        starttime = timeit.default_timer()
        bound = lvlBound - len(S)
        m = (S[0]+S[1])//2 +1
        for lvl in range(max(0,S[0]-1),min(S[0]+2,m)): #includes m
            bound += len(lvln[lvl])
        superQ = [[[node],bound] for node in lvln[S[0]]]
        for supr in superQ:
            supr[1] -= self.G.degree[supr[0][0]]
        degDict = {node:self.G.degree[node] for i in range(len(S[1:-1])) for node in lvln[S[i+1]]}
        for i in range(len(S[1:-1])):      
            mNext = (S[i+1]+S[i+2])//2 + 1  #we use m as a way to prevent calculating the same level more than once.
            bound = 0
            for lvl in range(max(m,S[i+1]-1),min(S[i+1]+2,mNext)):
                bound += len(lvln[lvl])
            m = mNext
            superQ = [[supr[0]+[node],supr[1]+bound - degDict[node]] for supr in superQ for node in lvln[S[i+1]] if node not in supr[0]]
        cnt=0
        bound = 0
        for lvl in range(max(m,S[-1]),min(S[-1]+2,len(lvln))):
            bound += len(lvln[lvl])
        cnt = 0
        degDict = {node:bound - self.G.degree[node] for node in lvln[S[-1]]}
        enQ = [[tuple(supr[0]+[node]),supr[1]+deg] for node,deg in degDict.items() for supr in superQ if node not in supr[0]]
        t=[]
        for S, bound in enQ:
            cnt+=1
            if bound<self.lowest_centrality:
                Q.add_task(S,bound)
        return timeit.default_timer() - starttime,cnt

    
    def distance_centrality(self,S):
        '''
        Exact calculation of supernode total distance
        '''
        starttime = timeit.default_timer() #for testing purposes
        centrality=0
        for n in self.G.nodes:
            centrality+=min(self.dist[n][s] for s in S)
            if centrality>=self.lowest_centrality:
                return 0,timeit.default_timer()-starttime
        return centrality, timeit.default_timer()-starttime

    def distance_centrality_no_thresh(self,S):
        '''
        Exact calculation of supernode total distance
        '''
        starttime = timeit.default_timer() #for testing purposes
        centrality=0
        for n in self.G.nodes:
            centrality+=min(self.dist[n][s] for s in S)
        return centrality, timeit.default_timer()-starttime
    
    def FP2(self):
        '''
        container function which runs the fastPAM2 algos
        '''
        starttime = timeit.default_timer()
        TD,m = self.LAB()
        lowest_centrality = self.distance_centrality(m)[0]
        self.lowest_centrality,self.lowest_supernode = self.FASTPAM2(list(m),lowest_centrality)
        self.approx = (self.lowest_centrality,self.lowest_supernode)
        return timeit.default_timer()-starttime
    
    def FP2_2(self):
        '''
        container function which runs the fastPAM2 algos
        '''
        starttime = timeit.default_timer()
        TD,m = self.LAB()
        lowest_centrality = self.distance_centrality_no_thresh(m)[0]
        self.approx_2 = self.FASTPAM2(list(m),lowest_centrality)
        return timeit.default_timer()-starttime
    
    def FP2_mid(self):
        '''
        container function which runs the fastPAM2 algos
        '''
        starttime = timeit.default_timer()
        TD,m = self.BUILD_mid()
        lowest_centrality = self.distance_centrality_no_thresh(m)[0]
        self.approx_mid = self.FASTPAM2(list(m),lowest_centrality)
        return timeit.default_timer()-starttime
    
    def FP2_BUILD(self):
        '''
        container function which runs the fastPAM2 algos
        '''
        starttime = timeit.default_timer()
        TD,m = self.BUILD()
        lowest_centrality = self.distance_centrality_no_thresh(m)[0]
        self.approx_BUILD = self.FASTPAM2(list(m),lowest_centrality)
        return timeit.default_timer()-starttime


    def find_central_supernode(self):#use different queues so we dont have to hold extra int
        starttime = timeit.default_timer()
        PJ_time = self.p_j()
        FP2_time = self.FP2()
        FP2_2_time = self.FP2_2()
        FP2_mid_time = self.FP2_mid()
        FP2_BUILD_time = self.FP2_BUILD()
        nlvl,bfs_time = self.calc_bfs(0)
        lbound_time = self.calc_level_bound(nlvl)
        dbound_time = 0
        dbound_tups = 0
        dbound_runs = 0
        opt_time = 0
        opt_runs = 0
        while 1:
            if self.lowest_centrality<=self.dbQ.front()[0]:
                if self.lowest_centrality<=self.lQ.front()[0]:
                    return {'Mid_val':self.distance_centrality_no_thresh(self.approx_mid[1])[0],
                            'Mid_set':self.approx_mid[1],
                            'Opt_val':self.lowest_centrality,
                            'Opt_set':self.lowest_supernode,
                            'FP2_val':self.approx[0],
                            'FP2_set':self.approx[1],
                            'FP2_val':self.approx_2[0],
                            'FP2_set':self.approx_2[1],
                            'PJ_val':self.PJ_centrality,
                            'PJ_set':self.PJ_medoids,
                            'Build_val':self.distance_centrality_no_thresh(self.approx_BUILD[1])[0],
                            'Build_set':self.approx_BUILD[1],
                            'k':self.k,
                            'V':len(self.G),
                            'E':self.G.size(),
                            'nlvls':{dist:len(nodes) for dist,nodes in nlvl.items()},
                            'bfs':bfs_time,
                            'lbound_time':lbound_time,
                            'dbound_time':dbound_time,
                            'FP2_time':FP2_time,
                            'FP2_2_time':FP2_2_time,
                            'PJ_time':PJ_time,
                            'Mid_time':FP2_mid_time,
                            'BUILD_time':FP2_BUILD_time,
                            'opt':opt_time,
                            'total':timeit.default_timer()-starttime,
                            'dbound runs':dbound_runs,
                            "dbound_tups":dbound_tups,
                            'opt runs':opt_runs,
                            'dbQ.counter':self.dbQ.counter,
                            'lQ.counter':self.lQ.counter,
                           }
                else:
                    supr,bound = self.lQ.pop_task()
                    dbound_runs +=1
                    db=self.db_v2(supr,nlvl,bound)
                    dbound_time += db[0]
                    dbound_tups+=db[1]

            elif self.dbQ.front()[0]<=self.lQ.front()[0]:
                supr,bound = self.dbQ.pop_task()
                opt_runs+=1
                opt = self.distance_centrality(supr)
                opt_time+= opt[1]
                
                #new best
                if opt[0]:
                    self.lowest_centrality = opt[0]
                    self.lowest_supernode = supr
            
            else:
                supr,bound = self.lQ.pop_task()
                dbound_runs +=1
                db=self.db_v2(supr,nlvl,bound)
                dbound_time += db[0]
                dbound_tups+=db[1]


class PQ:
    def __init__(self, ls=[]):
        self.pq = []                         # list of entries arranged in a heap
#         self.entry_finder = {}               # mapping of tasks to entries
#         self.REMOVED = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count
        if ls:
            for i in ls:
                self.add_task(i[0],i[1])

    def add_task(self,task, priority=0):
        'Add a new task or update the priority of an existing task'
#         if task in self.entry_finder:
#             self.remove_task(task)
        next(self.counter) #         count = next(self.counter)
        entry = [priority, task]
#         self.entry_finder[task] = entry
        hq.heappush(self.pq, entry)

#     def remove_task(self,task):
#         'Mark an existing task as REMOVED.  Raise KeyError if not found.'
#         entry = self.entry_finder.pop(task)
#         entry[-1] = self.REMOVED

    def front(self):
        try:
            return self.pq[0]
        except:
            return [math.inf]

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, task = hq.heappop(self.pq)
#             if task is not self.REMOVED:
#                 del self.entry_finder[task]
            return task,priority
        raise KeyError('pop from an empty priority queue')

    def costly_print(self):
        tasks = []
        while self.pq:
            task,priority = self.pop_task()
            print(task," ",priority)
            tasks.append((task,priority))
        for i in tasks:
            self.add_task(i[0],i[1])
