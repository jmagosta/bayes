# Matrix read write routines
########################################################################
class Ctemplate (object):
    'Module names and class names can be the same.'

    def __init__(self):
        self.input = None

    def rd_matrix(self, mat_file):
        'Input a tab-delimited integer file'
        try:
            with open(mat_file, 'rb') as fd:
                mat = pd.read_csv(fd, delimiter='\t')
        except Exception as e:
            print >> sys.stderr, 'Failed to read ', e, mat_file
            sys.exit(-1)
        self.input = mat

    def process(self):
        'Null operation'
        self.output = self.input

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
        except Exception as e:
            print >> sys.stderr, 'Failed to write ', e, output_fn
            # out_fd.close()
            sys.exit(-1)
        out_fd.close()