import medoid_multiq as mq
import networkx as nx
import pickle
import datetime
res = []
for p in [0.3]:#[0.3,0.3,0.7,0.7,0.7]:
    for n in [275,300]: #[25,50,75,100,125,150,175,200,250]
        G = nx.watts_strogatz_graph(n,6,p)
        test = mq.Kmed(G,3)
        res.append((p,test.find_central_supernode()))
        print(n)

with open("watts_"+str(datetime.datetime.now())+".pkl",'wb') as f:
    pickle.dump(res,f)
