import DCC as mq
import networkx as nx
import pickle
import datetime
res = []
file = "watts_"+str(int(datetime.datetime.now().timestamp()))+".pkl"
for k in [2,3]:#4
    for p in [0.03,0.1,0.7,0.01]:#[0.3,0.3,0.7,0.7,0.7]:
        for n in [25,50,75,100,125,150,175,200,225,250,275,300]: #[]
            G = nx.watts_strogatz_graph(n,6,p)
            test = mq.Kmed(G,k)
            res.append((p,test.find_central_supernode()))
            print(k,n)
            with open(file,'wb') as f:
                pickle.dump(res,f)
                
print('barabasi')
res = []
file = "barab_"+str(int(datetime.datetime.now().timestamp()))+".pkl"
for k in [2,3,4]:#4
    for n in [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750]: #[]
        G = nx.barabasi_albert_graph(n,6)
        test = mq.Kmed(G,k)
        res.append(test.find_central_supernode())
        print(k,n)
        with open(file,'wb') as f:
            pickle.dump(res,f)