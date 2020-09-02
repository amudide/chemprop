#!/usr/bin/python26
'''
Created on 2015-10-02

@author: Jinchao Yu
'''
import shutil
import re
import sys

# tools #
def get_pdbfile_containing_given_chains(PDB_file, chains, outfile=None):
    """
    chains (in string): 'AB', ' ', etc., '_' represents ' ' (void ID).
    For example, 'AB_A' or 'A B' both represent chains 'A', 'B' and ' '
    if outfile is None, no file will created. Outfile content will always be returned.
    Return outfile content
    """
    # no chain ID is given
    if len(chains)<1 or chains=='all':
        if outfile:
            shutil.copy(PDB_file, outfile)
        FHi = open(PDB_file, 'r')
        out_content = FHi.read()
        FHi.close()
        return out_content
    
    try:
        FH = open(PDB_file,"r")
    except:
        print("can not open the PDB file: " + PDB_file)
        return 1
    chain_set = set(chains)
    if '_' in chain_set:
        chain_set.discard('_')
        chain_set.add(' ')
    out_content = ""
    for line in FH:
        res_match_ATOM = re.match("ATOM  ",line)
        res_match_HETATM = re.match("HETATM",line)
        
        if res_match_ATOM or res_match_HETATM:
            try:
                if line[21] in chain_set:
                    out_content += line
            except:
                print("Abnormal PDB format detected")
        else:
            res_match_END = re.match("END",line)
            if res_match_END:
                out_content += line
    FH.close()
    if outfile:
        FHo = open(outfile,'w')
        FHo.write(out_content)
        FHo.close()
    return out_content


## END tools ##
def main():
    """
    """
    import optparse
    usage = "get_pdbfile_containing_given_chains.py -i in.pdb -c AB_ -o out.pdb\n"+\
            "get a pdb file from given chains in the input pdb file \n"+\
            "chains (in string): 'AB', 'A_BA',etc. '_' represents ' ' (void ID)."+\
            "For example, 'AB_A' represents chains 'A', 'B' and ' '"+\
            "PS: This method (get_pdbfile_containing_given_chains) is in super_toolbox (STL) module."
    
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-i','--in_pdb', action="store",dest='in_pdb',type='string',
                      help = "input pdb path")
    parser.add_option('-c','--chains', action="store",dest='chains',type='string',
                      help = "'AB', 'A_BA',etc. '_' represents ' ' (void ID).")
    parser.add_option('-o','--out_pdb', action="store",dest='out_pdb',type='string',
                      help = "output pdb path")
    if len(sys.argv) == 1:
        print(usage)
        print("Type -h or --help for more help information")
        sys.exit(1)
    (options, args) = parser.parse_args(sys.argv)
    if len(args) > 1: # sys.argv[0] is the name of this program
        print("Leftover arguments:" + str(args[1:]))
        
    get_pdbfile_containing_given_chains(options.in_pdb, options.chains, options.out_pdb)
    
if __name__ == '__main__':
    main()