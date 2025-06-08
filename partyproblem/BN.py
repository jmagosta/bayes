# BN.py
# object to manipulate a Bayes network
#
#   JMA April 2025

# Use to parse xdsl files
import xml.etree.ElementTree as et

from tabulate import tabulate
import networkx as nx

import Potential
from Potential import new_Potential
import numpy as np


# NOTE: all extract_* functions have side-effects that modify node_dict
class BN (object):

    def __init__(self, name_dict={}):
        ''
        # Reduce the xdsl to a dict of dicts.
        # Use it's keys for a list of nodes
        self.n_dict = name_dict
        # a networkx graph object
        self.network = None 
        # Build a reverse topological order to the DAG
        self.node_order = None
        # Derived from  parent lists
        self.edges = None

    def set_kind(self, a_node):
        self.n_dict[a_node.get('id')] = {'kind': a_node.tag}

    def extract_parents(self, a_node):
        parent_list = []
        p = a_node.find('parents')
        if p is not None:
            parent_list = p.text.split(' ') 
        self.n_dict[a_node.get('id')]['parents' ] = parent_list
        return self.n_dict

    def extract_states(self, a_node):
        state_list = []
        for element in a_node:
            if element.tag == 'state':
                state_list.append(element.get('id'))
        self.n_dict[a_node.get('id')]['states' ] = state_list
        return self.n_dict

    def state_size(self, node_name):
        # Deterministic nodes such as utilities have only one state. 
        
        # if a_node.tag == 'utilities':
        #     return 1
        # else:
        #     node_name = a_node.get('id') 
        return len(self.n_dict[node_name]['states'])
    
    def build_tensor(self, a_node, elements):
        node_name = a_node.get('id')
        dim_names = [node_name]
        # Need the parents to dimension the cpt
        state_counts = [self.state_size(node_name)]    
        parents = self.get_parents(node_name)
        dim_names.extend(parents)  
        for p in parents:
            state_counts.append(self.state_size(p))      #list of dimensions
        # print('S', state_counts)
        try:
            # if len(prob_list) == state_counts[0]:             # One dimension, no conditioning 
            potential = new_Potential(elements, state_counts, dim_names)   
            #     cpt = torch.tensor(prob_list).reshape(state_counts)
            self.n_dict[node_name]['potential' ] = potential
        except Exception as e:
            print('Err ', e)
            print(f'list of len {elements} is not a consistent with {state_counts}.')

    def extract_probabilities(self, a_node):
        # Probabilities are stored as a flat list, in row major order, e.g. 
        # for each conditioning, the probs for each state are listed together
        # sequentially. 
        p = a_node.find('probabilities')
        if p is not None:
            prob_list = [float(k) for k in p.text.split(' ')]
            self.build_tensor(a_node, prob_list)
        # except Exception as e:
        #     print('Err ', e)
        #     print(f'list of len {prob_list} is not a consistent with {state_counts}.')
        return self.n_dict

    def extract_utilities(self, a_node):
        u = a_node.find('utilities')
        self.n_dict[a_node.get('id')]['states' ] = ['utility']   # a dimension with just one state. 
        if u is not None:
            u_list = [float(k) for k in u.text.split(' ')]
            self.build_tensor(a_node, u_list)
            # TODO The utilities list dimension with  parent states. 
        # self.n_dict[a_node.get('id')]['utilities' ] = u_list
        return self.n_dict

    # Note: Node extensions also have the display name of the node, which is an
    # alternative to it's id. 
    def extract_positions(self, a_node_extension):
        u = a_node_extension.find('position')
        if u is not None:
            u_list = [int(k) for k in u.text.split(' ')]
            # The utilities list cannot be dimensioned until we know it's parent states. 
        self.n_dict[a_node_extension.get('id')]['position' ] = u_list
        return self.n_dict
    
    def weave(self):
        'From the reaped list of nodes connect them into a network.'
        # Assemble edge lists
        edges = []
        for (k, attr) in self.n_dict.items():
            parents = self.get_parents(k)
            if len(parents) > 0:
                [edges.append((z, k )) for z in parents] # Arc direction: z -> k
        print('Edges: ',edges, '\n')
        self.edges = edges
        self.network = nx.DiGraph(self.edges)
        return edges

    # TODO Create an a-cyclic graph from the parents of each node. 

    ### Accessors ###

    def get_states(self, a_node):
        return self.n_dict[a_node]['states' ]
                
    def get_parents(self, a_node):
        return self.n_dict[a_node]['parents' ]

    def get_potential(self, a_node):
        'Find the probability np array in the node, and label it using parents in the graph'
        # The states of the RV label the columns, so that the matrix is row-markov
        the_cpt = self.n_dict[a_node]['potential']
        return the_cpt
    
    def pr_potential(self, a_node):
        the_potential = self.get_potential(a_node)
        print('\tpotential: ', [k for k in the_potential.shape.keys()])
        print('\t', str(the_potential.p).replace('tensor(','').replace(')', ''))
    
    def node_centers(self):
        xmax = ymax = - pow(2,16)
        xmin = ymin =   pow(2,16)
        center_dict = {}
        for k,attr in self.n_dict.items():
            v = attr['position']
            x = (v[0] + v[2])/2
            y = -(v[1] + v[3])/2
            center_dict[k]  = np.array((x,y))
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)
        return center_dict
        
    ### Print functions ###
    def pr_nodes(self):
        'Node names and types'
        for k, attrs in self.n_dict.items():
            print(f'>>> {k} <<<')
            # if attrs['kind'] == 'cpt' or attrs['kind'] == 'utility':
            for f, v in attrs.items():
                    if f == 'potential':
                        self.pr_potential(k)
                    else:
                        print(f'\t{f}: {v}')
            print()
    
    def pr_named_tensors(self):
        'Show all the model tensors'
        name_dict = self.n_dict
        for a_node in name_dict:
            if name_dict[a_node]['kind'] == 'cpt' or name_dict[a_node]['kind'] == 'utility':
                print(a_node, '\n\t', self.get_potential(a_node),'\n')

    # Format one-dim tensors 
    # from collections import deque
    def one_dim_table(self, the_potential, the_var, **args):
        def que_copy(prefix, queue):
            if not isinstance(queue, list):
                queue = [queue]
            queue.insert(0, prefix)
            return queue
        states = self.n_dict[the_var]['states']
        values = the_potential.tolist()
        # Flatten nested lists
        while len(values)  == 1:   # TODO is this test necessary?
            values = values[0]
        print(f' *** {the_var} ***')
        values = [que_copy(s, v) for s, v in zip(states, values)]
        print(tabulate(values, **args))

##  Walk the xdsl elements of a network
def extract_net(xdsl_file):
    '''Finds the first element under the top level that contains a list of nodes,
    and returns a dict of node element objects.'''
    tree = et.parse(xdsl_file)
    root = tree.getroot()
    # BN structure is contained under the node element
    node_tree = root.findall('nodes')[0]
    print(f'found {node_tree.tag}')
    extensions = root.find('extensions')
    extensions_tree = extensions.find('genie')
    node_extensions = extensions_tree.findall('node')
    return  list(node_tree), list(node_extensions)

# create a BN object from the parsing 
def reap(the_parse):
    'Factory to parse the attributes of each node, returning a list with the attributes in a dict.'
    bn = BN()
    the_nodes, the_extensions = the_parse
    for a_node in the_nodes:
        # Set the node kind
        bn.set_kind(a_node)
        # node_dict[a_node.get('id')] = {'kind': a_node.tag}
        bn.extract_parents(a_node)
        # CPT and decision nodes have states
        if (a_node.tag == 'cpt') or (a_node.tag == 'decision'):
            bn.extract_states(a_node)
        if (a_node.tag == 'cpt'):
            bn.extract_probabilities(a_node)
        if (a_node.tag == 'utility'):
            bn.extract_utilities(a_node)
    for an_ex in the_extensions:
        bn.extract_positions(an_ex)
    bn.edges = bn.weave()
    return bn

### Main ###
if __name__ == '__main__':

    NETWORK_FILE = 'PartyProblem_asym.xdsl'
    # BN structure is contained under the node branch
    parsed = extract_net(NETWORK_FILE)
    bn = reap(parsed)
    bn.pr_nodes()
    print()
    print(bn.node_centers())  


    