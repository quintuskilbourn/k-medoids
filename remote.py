import medoid_multiq as mq
import networkx as nx
import pickle
import datetime
res = []
for k in [2,3]:#4
	for p in [0.03]:#[0.3,0.3,0.7,0.7,0.7]:
	    for n in [25,50,75,100,125,150,175,200,225,250]: #[275,300]
	        G = nx.watts_strogatz_graph(n,6,p)
	        test = mq.Kmed(G,k)
	        res.append((p,test.find_central_supernode()))
	        print(n)
	        with open("watts_"+str(int(datetime.datetime.now().timestamp()))+".pkl",'wb') as f:
	            pickle.dump(res,f)
