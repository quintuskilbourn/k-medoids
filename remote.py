import medoid_multiq as mq
import networkx as nx
import pickle
res = []
for p in [0.01,0.3,0.5,0.7]:

    G = nx.watts_strogatz_graph(200,6,p)
    test = mq.Kmed(G,3)
    res.append(test.find_central_supernode())

with open("result.pkl",'wb') as f:
    pickle.dump(res,f)
