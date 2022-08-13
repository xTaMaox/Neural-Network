import networkx as nx
import sys
from math import tanh
sys.path.append('pysrc')
import neural_networks as nn
from functools import partial

DG = nx.DiGraph()
DG.add_nodes_from(['X1','X2','H0','H1','H2','O1'])

#set the activation functions
DG.node['H0']['af']=1.0
for node in ['H1','H2','O1']:
    DG.node[node]['af']=tanh
    
#set the derivatives of the activation functions for all nodes except the output, this is done below
def zero(x):
    return 0
for node in ['X1','X2','H0']: #the inputs and bias terms have zero derivatives
    DG.node[node]['daf'] = zero
def dtanh(x):
    return 1.0 - tanh(x) * tanh(x)
for node in ['H1','H2']:
    DG.node[node]['daf']=dtanh
    
#create the edges
for source in ['X1','X2']:
    for target in ['H1','H2']:
        DG.add_weighted_edges_from([(source,target,1.0)])
for source in ['H0','H1','H2']:
    DG.add_weighted_edges_from([(source, 'O1', 1.0)])
    
#set the input values
DG.node['X1']['af']=0
DG.node['X2']['af']=1

#given these inputs, the correct output should be 1
#we'll use a partial function so we can assign the correct 'daf' value
#dynamically when we iteratively train the network
def dout(x, t):
    if x<0:
        xx = 0
    else:
        xx = 1
    return xx - t
DG.node['O1']['daf']=partial(dout, t=1)

nn.forward_prop(DG)

nn.error_back_prop(DG)

for node in ['X1','X2','H0','H1','H2','O1']:
    print "node {0} has output {1} and error {2}".format(node, DG.node[node]['o'], DG.node[node]['e'])