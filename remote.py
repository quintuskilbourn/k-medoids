import medoid_multiq as mq
import networkx as nx
import pickle
res = []
for p in [0.01,0.3,0.5,0.7]:
    for n in [150,200,250]:
        G = nx.watts_strogatz_graph(n,6,p)
        test = mq.Kmed(G,3)
        res.append(((p,n),test.find_central_supernode()))

with open("result.pkl",'wb') as f:
    pickle.dump(res,f)
