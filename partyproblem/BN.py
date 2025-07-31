# BN.py
# object to manipulate a Bayes network
#
#   JMA April 2025

# Use to parse xdsl files
import xml.etree.ElementTree as et

# Pretty print potentials
from tabulate import tabulate
import networkx as nx

from Potential import * 
# import Potential
import numpy as np

DEBUG = 0


# NOTE: all extract_* functions have side-effects that modify node_dict
class BN (object):

    def __init__(self, node_dict={}):
        ''
        # Reduce the xdsl to a dict of dicts.
        # Use it's keys for a list of nodes
        self.n_dict = node_dict
        # a networkx graph object
        self.network = None 
        # Build a reverse topological order to the DAG
        self.node_order = None
        # Derived from  parent lists
        self.edges = None
        self.center_dict = {}
        self.enclosing_rect = dict(xmin= pow(2,16) ,
                                   ymin =   pow(2,16),
                                   xmax = - pow(2,16),
                                   ymax = - pow(2,16))
    ### Accessors ###

    def get_node(self, node_name:str):
        return self.n_dict[node_name]  

    # def get_states(self, a_node):
    #     return self.n_dict[a_node]['states' ]
                
    # def get_parents(self, a_node):
    #     return self.n_dict[a_node]['parents' ]
    

    # def get_potential(self, a_node):
    #     'Find the probability np array in the node, and label it using parents in the graph'
    #     # The states of the RV label the columns, so that the matrix is row-markov
    #     the_cpt = self.n_dict[a_node]['potential']
    #     return the_cpt


    

    # def set_kind(self, a_node):
    #     'Both create the node key and its kind.'
    #     self.n_dict[a_node.get('id')] = ID_node(a_node.tag) 
    #     self.get_node(a_node) = {'kind': a_node.tag}
    #     # self.n_dict[a_node.get('id')] = {'kind': a_node.tag}

    def extract_parents(self, a_node):
        parent_list = []
        p = a_node.find('parents')
        if p is not None:
            parent_list = p.text.split(' ') 
        # self.n_dict[a_node.get('id')]['parents' ] = parent_list
        return parent_list

    def extract_states(self, a_node):
        state_list = []
        for element in a_node:
            if element.tag == 'state':
                state_list.append(element.get('id'))
        # self.n_dict[a_node.get('id')]['states' ] = state_list
        return state_list

# A property of the ID_node
    # def state_size(self, node_name):
    #     # Deterministic nodes such as utilities have only one state. 
        
    #     # if a_node.tag == 'utilities':
    #     #     return 1
    #     # else:
    #     #     node_name = a_node.get('id') 
    #     return len(self.n_dict[node_name]['states'])
    
    def build_potential(self, elements, features):
        # node_name = a_node.get('id')
        dim_names = [features['name']]
        # Need the parents to dimension the cpt! 
        parents = features['parents']
        states = features['states']
        state_sizes = [len(states)]    
        dim_names.extend(parents)  
        for p in parents:
            state_sizes.append(self.get_node(p).state_size())      #list of dimensions
        # print('S', state_sizes)
        try:
           # One dimension, no conditioning 
            potential = new_Potential(elements, state_sizes, dim_names)   
                #     cpt = torch.tensor(prob_list).reshape(state_sizes)
                # self.n_dict[node_name]['potential' ] = potential
            return potential 
        except Exception as e:
            print('Err ', e)
            print(f'list of len {elements} is not a consistent with {state_sizes}.')
            return None

    # def extract_probabilities(self, p, features):
    #     # Probabilities are stored as a flat list, in row major order, e.g. 
    #     # for each conditioning, the probs for each state are listed together
    #     # sequentially.
    #     node_name = features['name']
    #     states_list = features['states']
    #     if p is not None:
    #         prob_list = [float(k) for k in p.text.split(' ')]
    #         potential = self.build_tensor(node_name, prob_list, states_list)
    #     # except Exception as e:
    #     #     print('Err ', e)
    #     #     print(f'list of len {prob_list} is not a consistent with {state_sizes}.')
    #     return potential

    # def extract_utilities(self,u, features):
        
    #     # TODO does xdsl not label utility / deterministic node states?
    #     node_name = features['name'] 
    #     # self.n_dict[a_node.get('id')]['states' ] = ['utility']   # a dimension with just one state. 
    #     if u is not None:
    #         u_list = [float(k) for k in u.text.split(' ')]
    #         potential = self.build_tensor(node_name, u_list, ['utility'])
    #         # TODO The utilities list dimension with  parent states. 
    #     # self.n_dict[a_node.get('id')]['utilities' ] = u_list
    #     return potential
    
    def uniform_potential(self, features):
        states = features['states']
        # TODO lookup in self ()
        parents = features.get('parents')
        if len(parents) == 0:
            uniform = [1] * len(states)
            dim = [len(states)]
            return new_Potential(uniform, dim, [features['name']])
        return None

    # Note: Node extensions also have the display name of the node, which is an
    # alternative to it's id. 
    def extract_positions(self, a_node_extension):
        u = a_node_extension.find('position')
        if u is not None:
            u_list = [int(k) for k in u.text.split(' ')]
            # The utilities list cannot be dimensioned until we know it's parent states. 
        # self.n_dict[a_node_extension.get('id')]['position' ] = u_list
        return u_list
    
    def node_centers(self):
        'Parse node dimensions for plotting purposes.'
        for k,attr in self.n_dict.items():
            v = attr['position']
            x = (v[0] + v[2])/2
            y = -(v[1] + v[3])/2
            self.center_dict[k]  = np.array((x,y))
            self.enclosing_rect['xmin'] = min(x, self.enclosing_rect['xmin'])
            self.enclosing_rect['ymin'] = min(y, self.enclosing_rect['ymin'])
            self.enclosing_rect['xmax'] = max(x, self.enclosing_rect['xmax'])
            self.enclosing_rect['ymax'] = max(y, self.enclosing_rect['ymax'])
        return self.center_dict
    
    def weave(self):
        'From the reaped list of nodes connect them into a network.'
        # Assemble edge lists
        edges = []
        for (k, attr) in self.n_dict.items():
            parents = self.get_parents(k)
            if len(parents) > 0:
                [edges.append((z, k )) for z in parents] # Arc direction: z -> k
        if DEBUG > 0: print('Edges: ',edges, '\n')
        self.edges = edges
        self.network = nx.DiGraph(self.edges)
        self.node_centers()
        # An ordering respecting the arc directions. e.g the
        # utility node should be last on the list. 
        self.node_order = list(nx.topological_sort(self.network))
        return edges

    # TODO Create an a-cyclic graph from the parents of each node. 
        
    ### Print functions ###
    def pr_influences(self):
        'Print both the nodes parents and children.'
        print('Node\t{ancestors}\n\t{descendants}\n')
        for n in self.network:
            print(n, ': ',nx.ancestors(self.network,n), '\n\t', 
                nx.descendants(self.network,n), '\n')
            

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

### BN

### Examples for parsing a xdsl file
def extract_floats(element):
    return [float(z) for z in element.text.split(' ')]

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
def reap(the_parse_tuple):
    'Factory to parse the attributes of each node, returning a list with the attributes in a dict.'
    # Pass the node tree and extensions from 'extract net'
    bn = BN()
    the_nodes, the_extensions = the_parse_tuple
    for a_node in the_nodes:
        # Create a local container for node features. 
        features = dict(name=a_node.get('id'))
        # Set the node kind
        features['kind'] = a_node.tag 
        # id_node.set_kind(a_node.tag)
        # node_dict[a_node.get('id')] = {'kind': a_node.tag}
        features['parents'] = bn.extract_parents(a_node)
        # CPT and decision nodes have states, not deterministic or value nodes
        if (a_node.tag == 'cpt') or (a_node.tag == 'decision'):
            features['states'] = bn.extract_states(a_node)
        elif (a_node.tag == 'utility'):
            features['states'] = [a_node.tag]  # Utilities have only one state, it is arbitrary.
        if (a_node.tag == 'cpt'):
            probs = a_node.find('probabilities')
            elements = extract_floats(probs)
            features['potential'] = bn.build_potential(elements, features)
        if (a_node.tag == 'utility'):
            utils = a_node.find('utilities')
            elements = extract_floats(utils)
            features['potential'] = bn.build_potential(elements, features)
        if (a_node.tag == 'decision'):
            # TODO use build_potential instead. Need default elements 
            # for all parents. 
            features['potential'] = bn.uniform_potential(features)
        # id_node = ID_node(a_node.get('id'))
        # TODO - all other node types
        if a_node.tag in ('cpt', 'decision', 'utility'):
            node_object = create_from_dict(features)
            # The set of  nodes is kept in the BN object dict. 
            # This function assumes that a nodes parents are created before it is. 
            # TODO use BN::get_node() instead
            bn.n_dict[node_object.label] = node_object
    for an_ex in the_extensions:
        # Only node types that were parsed will be in bn.dict
        if bn.n_dict.get(an_ex.attrib['id']) is not None:
            bn.n_dict[an_ex.attrib['id']].set_positions(bn.extract_positions(an_ex))
    # Build the network, and 
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
    print(bn.center_dict)  


    