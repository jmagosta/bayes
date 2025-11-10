# BN.py
# object to manipulate a Bayes network
#
#   JMA April 2025

import sys

# Use to parse xdsl files
import xml.etree.ElementTree as et

# Pretty print potentials
from tabulate import tabulate
import networkx as nx

# from Potential import * 
from ID_node import *
# import Potential
import numpy as np

# 
from potential_operations import condition_decision

# from tabulate import tabulate
import matplotlib.pyplot as plt
# import seaborn as sn 

DEBUG = 1


# NOTE: all extract_* functions have side-effects that modify node_dict
class BN (object):

    def __init__(self, node_dict={}):
        ''
        # Reduce the xdsl to a dict of dicts.
        # Use it's keys for a list of nodes
        self.n_dict = node_dict
        # Derived from  parent lists
        self.edges = None
        # a networkx graph object
        self.network = nx.DiGraph(self.edges)
        # Build a reverse topological order to the DAG
        self.node_order = None
        # Node locations
        self.center_dict = {}
        # Canvas extent
        self.enclosing_rect = dict(xmin= pow(2,16) ,
                                   ymin =   pow(2,16),
                                   xmax = - pow(2,16),
                                   ymax = - pow(2,16))
    ### Accessors ###

    def get_node(self, node_name:str) -> ID_node:
        return self.n_dict[node_name]  

    def get_states(self, a_node: str) -> list:
        return self.n_dict[a_node].get_states()
                
    def get_parents(self, a_node: str):
        return self.n_dict[a_node].parents
    
    def get_potential(self, a_node: str) -> Potential:
        'Find the Potential object in the node'
        # The states of the RV label the columns, so that the matrix is row-markov
        the_p = self.get_node(a_node).potential
        return the_p
    
    ### Modify network
    def remove_node(self, the_node, show_graph=False):
        '''Remove the node from the network, updating the parents by connecting the node's parents to its children.
        (Assuming the Potentials have already been updated for this variable
        by adjusting Potential conditionings.)'''

        if the_node not in self.n_dict:
            print(f"Error, there is no node {the_node}")
            return None

        # Get node's predecessors (parents) and successors (children)
        parents = list(self.network.predecessors(the_node))
        children = list(self.network.successors(the_node))

        # Edge case - Multiple parents and no children. 
        # Removing the node will disconnect the network
        if len(children) == 0 and len(parents) > 1:
            print(f'Warning: Multiple parents to childless node: {the_node}\n'
                  'Removal will disconnect the network')

        # Add edges from each parent to each child to bypass the node
        for parent in parents:
            for child in children:
                self.network.add_edge(parent, child)
                # Update the parents lists of children to include 
                # the bypassed parents for child in children:
                if parent not in self.n_dict[child].parents:
                    self.n_dict[child].parents.append(parent)

        # Remove the node from the graph
        self.network.remove_node(the_node)
        # Remove the node from the node dictionary
        del self.n_dict[the_node]
        # Remove node position
        del self.center_dict[the_node]
        # Update the edges list
        self.edges = list(self.network.edges())
        # Recompute the topological order
        self.node_order = list(nx.topological_sort(self.network))
        if show_graph:
            self.pr_network()
        return None

    
    ### Print functions ###
    def pr_order(self):
        'Ordering of the node that respects the DAGs partial ordering.'
        print('Topological sort: ', self.node_order)

    def pr_influences(self):
        'Print both the nodes parents and children.'
        print('Node\t{ancestors}\n\t{descendants}\n')
        for n in self.network:
            print(n, ': ',nx.ancestors(self.network,n), '\n\t', 
                nx.descendants(self.network,n), '\n')
            
    def pr_nodes(self):
        'Node names and types'
        for k, node in self.n_dict.items():
            if DEBUG: print(f'>>> {k} <<<')
            node.pr_node()
            print()
        self.pr_order()

    def pr_locations(self):
        print('\nNode Centers:')
        for n, loc in bn.center_dict.items():
            print(f'{n}: \t{loc}')

    # Use the extracted dimensions to plot with networkx
    def pr_network(self):
        positions = self.center_dict
        DG = self.network
        plt.figure(figsize=(6,3))
        nx.draw_networkx_labels(DG, pos=positions)
        nx.draw_networkx_nodes(DG, pos=positions, node_color='lightgrey')
        nx.draw_networkx_edges(DG, pos=positions)
    
    # Similar to pr_nodes()
    def pr_named_tensors(self):
        'Show all the model tensors'
        name_dict = self.n_dict
        for a_node in name_dict:
            if name_dict[a_node].get_kind() in ('cpt', 'decision', 'utility'):
                print(a_node, '\n\t', self.get_potential(a_node),'\n')

    # Format one-dim tensors 
    from collections import deque
    def pr_one_dim_table(self, the_var, **args):
        potential = self.get_potential(the_var) 
        states = self.get_node(the_var).states
        pr_table(potential, the_var, states, **args)
    
    ### parse xdsl

    def extract_parents(self, a_node):
        parent_list = []
        p = a_node.find('parents')
        if p is not None:
            parent_list = p.text.split(' ') 
        return parent_list

    def extract_states(self, a_node):
        state_list = []
        for element in a_node:
            if element.tag == 'state':
                state_list.append(element.get('id'))
        return state_list
    
    # TODO move this to ID_node? 
    def build_potential(self, elements, f_dict):
        # node_name = a_node.get('id')
        name = [f_dict['name']]
        # Need the parents to dimension the cpt! 
        dim_names = f_dict['parents'].copy()
        states = f_dict['states']
        state_sizes = []   
        for p in dim_names:
            state_sizes.append(self.get_node(p).state_size())      #list of dimensions
        # The marginal, (similarly the value) should be the last dim
        dim_names.extend(name)  
        # print('S', state_sizes)
        state_sizes.append(len(states))
        try:
           # One dimension, no conditioning 
            potential = new_Potential(elements, state_sizes, dim_names)   
            return potential 
        except Exception as e:
            print('Err ', e)
            print(f'list of len {elements} is not a consistent with {state_sizes}.')
            return None

    def uniform_potential(self, features):
        states = features['states']
        parents = features.get('parents')
        state_cnt = len(states)
        uniform = [1/state_cnt] * state_cnt
        unconditioned_d = new_Potential(uniform, [state_cnt], [features['name']])
        if len(parents) == 0:
            return unconditioned_d
        else:
            # Stack dimensions for parents
            # TODO we need the parents names and sizes
            return condition_decision(unconditioned_d, the_observation: Potential)

    # Note: Node extensions also have the display name of the node, which is an
    # alternative to it's id. 
    def extract_positions(self, a_node_extension):
        u = a_node_extension.find('position')
        if u is not None:
            u_list = [int(k) for k in u.text.split(' ')]
            # The utilities list cannot be dimensioned until we know it's parent states. 
        return u_list
    
    def node_centers(self):
        'Parse node dimensions for plotting purposes.'
        for k,attr in self.n_dict.items():
            v = attr.positions
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
            parents = attr.get_parents()
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
### End BN

def pr_table(potential, label, states, **args):
    ''
    def que_copy(prefix, queue):
        if not isinstance(queue, list):
            queue = [queue]
        queue.insert(0, prefix)
        return queue
        potential = self.get_node(the_var).potential
        states = self.get_node(the_var).states
    values = potential.cpt.tolist()
    # Flatten nested lists
    while len(values)  == 1:   # TODO is this test necessary?
        values = values[0]
    print(f' *** {label} ***')
    values = [que_copy(s, v) for s, v in zip(states, values)]
    print(tabulate(values, **args)) 


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
        if DEBUG: 
            print(f'reap: {a_node.get('id')}, {a_node.tag}', file=sys.stderr)
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
        if is_nodekind(a_node.tag): #a_node.tag in ('cpt', 'decision', 'utility'):
            if DEBUG > 1: 
                print(f'\tfeatures: {features}')
            node_object = create_from_dict(features)
            # The set of  nodes is kept in the BN object dict. 
            # This function assumes that a nodes parents are created before it is. 
            # TODO use BN::get_node() instead
            bn.n_dict[node_object.label] = node_object
        else:
            print(f'Unsupported node type: {a_node.tag}', file=sys.stderr )
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
    bn.pr_one_dim_table('Weather')
    bn.pr_nodes()
    bn.remove_node('Adjustor')
    bn.pr_order()
    bn.remove_node('Detector')
    bn.pr_order()
    bn.pr_locations()
    print()
    bn.pr_named_tensors()
