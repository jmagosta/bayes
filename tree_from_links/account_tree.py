#!/usr/bin/python3
#  26 May 2018 JMA
# account_tree.py
'''
Assemble subcategory tree from the tree's directed arcs

To generate the hierachy to stdout run
$ python3 account_tree.py Account_Hierarchy_ForMS.csv

Usage:
$ ./account_tree.py [-v] hierarchy_file.csv data_file.csv

    -v verbose output
''' 


### Python standard modules - remove those not necessary
import copy
import csv
import os
import os.path
import pprint
import sys
import re
import string
import time


### config constants
DBG=False
VERBOSE = False
OUT_DIR    = os.getcwd() + '/'        ## Default for -o option
CONFIG_DIR = os.getcwd() + '/'        ## Default for -l option
# py_dir     = '../py/'                               ## Used to load modules when testing in the git archive. 

ACTIVE =  [1+a for a in range(46)] # Those accounts with TS values.  To be derived from the data file

########################################################################
class Ccategory_tree (object):
    'Build the hierarchy for each account by creating a tree from the parent-child pairs.'

    def __init__(self):
        self.tree = None
        self.input = None
        self.data = None
        self.paths = None
        

    def account_no(self, an_account):
        'Format elements when they are read'
        account_match = re.match('Account_', an_account)
        if account_match:
            return int(re.sub('Account_','', an_account))
        else:
            return an_account
        

    def rd_matrix(self, mat_file, sep=','):
        'Input a delimited file as a list of lists.'
        try:
            fd = open(mat_file, 'rt')
            a_lol = []
            reader = csv.reader(fd, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
            for k, row in enumerate(reader):
                # Eliminate comments. # Remove header
                if row[0] != '#' and k != 0:
                    l_of_elements = [self.account_no(x) for x in row]
                    if DBG: print('Row:', l_of_element)
                    a_lol.append(l_of_elements)
        except Exception as e:
            print('Failed to read ', mat_file, e, file=sys.stderr)
            fd.close()
            sys.exit(-1)
        fd.close()
        return a_lol

    
    def rd_hierarchy(self, hierarchy_file):
        '''Read a table with two columns, parent and child. Each row indicates
        one child of the parent.'''
        ## Convert list of lists 
        self.input = self.rd_matrix(hierarchy_file)
        if VERBOSE:
            print('# Read hierarchy: ', len(self.input), 'by', len(self.input[0]))

            
    def rd_data_table(self, data_file):
        self.data = self.rd_matrix(data_file)
        if VERBOSE:
            print('# Read data: ', len(self.data), 'by', len(self.data[0]))


    def find_root_nodes(self):
        ' A root node is any node that does not appear as a child.'
        set_of_parents = set([k[0] for k in self.input])
        set_of_children = set([k[1] for k in self.input])
        self.roots = set.difference(set_of_parents, set_of_children)
        self.terminals = set.difference(set_of_children, set_of_parents)
        if VERBOSE:
            print('# {} Terminals: {}'.format(len(self.terminals), sorted(self.terminals)))
                                            

    def convert_to_dict(self):
        'The raw input put in useful form by consolidating over all children.'
        # The dict is parent => child_list
        self.arcs = dict()
        for pair in self.input:
            if pair[0] in self.arcs.keys():
                self.arcs[pair[0]].append(pair[1])
            else:
                self.arcs[pair[0]] = [pair[1]]
        if VERBOSE:
            print('# {} Arcs in total'.format(len(self.arcs)))

            
    def value_of(self, a_node):
        'Create a marker for terminal nodes'
        if a_node in ACTIVE:
            return ['A', a_node]
        else:
            return [None, a_node]
        

    def expand_node(self, a_node):
        'Recur over the dict of children to build the tree.'
        # print('expand_node', a_node)
        if a_node in self.arcs.keys():
            return [a_node, [self.expand_node(k) for k in self.arcs[a_node]]]
        else:
            # Build a terminal node
            return self.value_of(a_node)

            
    def convert_dict_to_tree(self):
        'Start with the root, then add leaves by looking up children in the dict'
        self.tree = dict()
        # root is one node that points to all nodes that do not appear as children (could be only one).
        self.tree = [self.expand_node(k) for k in self.roots] # remove outer list if only one root.
        if VERBOSE:
            pp = pprint.PrettyPrinter(indent=2, width=120, compact=True)
            print("self.tree\n ", pp.pprint(self.tree))
        
    ##################################### Walk all paths in the hierarchy tree ####
    def all_tree_paths_aux(self, a_node, path_so_far):
        '''Add the current node to the path, then descend by recuring, 
        and add the list to self.path when at a leaf.'''
        if DBG:
           print('atp: ', end='')
           pr_list(a_node)
        if type(a_node) is list and a_node[0] == 'A':
            # if the list is terminal
            path_so_far.append(a_node)
            if DBG: print('AT: ', path_so_far, end='\n')
            print(pr_account_path(path_so_far))
        elif type(a_node) is list and  a_node[0] is None:
            # if the list is empty terminal
            path_so_far.append(a_node)
            if DBG: print('NT: ', path_so_far, end='\n')
            print(pr_account_path(path_so_far))
        elif type(a_node) is list and type(a_node[0]) is int:
            if DBG: print('descend node ', a_node[0])
            path_so_far.append(a_node[0])
            self.descend_nodelist(a_node[1], path_so_far)
        else:
            print('Oops! Badly formed tree @\t', end='', file=sys.stderr)
            pr_list(a_node)
 

    def descend_nodelist(self, a_nodelist, path_so_far):
        ' Pursue all the branches in a nodes nodelist'
        if DBG:
            print('nodelist ', path_so_far, len(a_nodelist), end='\t')
            pr_list(a_nodelist)
        for x, nd in enumerate(a_nodelist):
            if DBG: print('ND ', x)
            self.all_tree_paths_aux(nd, path_so_far[:])
 
        
    def all_tree_paths(self):
        'Convert the tree to a list of all paths from root to leaves (so it can be inverted)'
        # Recur down all branches.
        self.descend_nodelist(self.tree, [])
          
        #####################################
    def process(self):
        'Sequence of processing steps'
        self.find_root_nodes()
        self.convert_to_dict()
        self.convert_dict_to_tree()
        self.all_tree_paths()
        # self.output = self.input


    def wr_header(self, fd, args):
            fd.write('# ' + '\t'.join(sys.argv) + '\n')
            host = re.sub('\n', '', subprocess.check_output('hostname'))
            user = os.environ['USER']
            date = time.asctime()
            fd.write('# ' + host+ '\t'+ user+ '\t'+ date + '\n')
        

    def wr_matrix(self, output_fn):
        try:
            out_fd = open(output_fn, 'wb')
            self.wr_header(out_fd, sys.argv)
            np.savetxt(out_fd, self.output, delimiter='\t', fmt='%8d')
        except Exteption as e:
            print('Failed to write ', e, output_fn, file=sys.stderr)
            out_fd.close()
            sys.exit(-1)
        out_fd.close()


########################################################################
def pr_list(lst):
    if type(lst) is list:
        print('[', ' '.join(map(str, lst))[0:80], '...')
    else:
        print('Z ',lst)


def pr_account_path(account_l):
    the_path = 'Account_' + '\tAccount_'.join([str(k) for k in account_l[:-1]])
    last_l = account_l[-1]
    the_path = the_path + '\tAccount_' + str(last_l[1])
    if last_l[0] is not None:
        the_path = the_path + '\t*'
    return the_path

    
def assemble_output_file_name(prefix, index ='', suffix ='.txt', output_dir=OUT_DIR):
    '''Place the file in the proper path, adding a prefix & suffix. eg.
    root_dir/sub_dir/prefix + index + suffix
    The output files will add a prefix & suffix to the frame_index, e.g. ~/run/posteriors/regions_43.pkl '''
    ## Make sure the suffix starts with .
    if suffix[0] != '.':
        suffix = '.' + suffix
    ## Check if output dir exists
    # output_dir = os.path.join(root_dir, sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('# Created:', output_dir, file=sys.stderr)
    ## Assemble the file name
    return os.path.join(output_dir, prefix + index + suffix)


########################################################################
def tree(input_file, data_file, output_file):

    ## Create objects
    the_tree = Ccategory_tree()

    ## Read in the input e.g. as a matrix
    the_tree.rd_hierarchy(input_file)

    ## Read in the data frame file
    if data_file:
        the_tree.rd_data_table(data_file)

    ## Do something to it
    the_tree.process()

    ## Save it
    # the_template.wr_matrix(output_file)


###############################################################################
def main(args):

    output_file = None
    ## Inputs
    input_file = args[0] # os.path.expanduser(args[0]) # Assuming the path is relative to the user's home path
    if len(args) > 1:
        data_file = args[1]
    else:
        data_file = None
        output_file = assemble_output_file_name('tree_as_table')
    ## Run 
    tree(input_file, data_file, output_file)

    print('# Output to:', output_file, file=sys.stderr)
    return 0


########################################################################
if __name__ == '__main__':

    if '-v' in sys.argv:
        k = sys.argv.index('-v')
        VERBOSE = True
        del(sys.argv[k:(k+1)])

    args = sys.argv[1:]
    exit_value = main(args)
    print(exit_value, '<-', sys.argv, "Done in ", '%5.3f' % time.process_time(), " secs!", file=sys.stderr)
#EOF
