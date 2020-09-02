from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

import pandas as pd


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph1: BatchMolGraph, mol_graph2: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        ret = []

        for mol_graph in [mol_graph1, mol_graph2]:


            if self.use_input_features:
                features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

                if self.features_only:
                    return features_batch

            f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

            if self.atom_messages:
                a2a = mol_graph.get_a2a().to(self.device)

            # Input
            if self.atom_messages:
                input = self.W_i(f_atoms)  # num_atoms x hidden_size
            else:
                input = self.W_i(f_bonds)  # num_bonds x hidden_size
            message = self.act_func(input)  # num_bonds x hidden_size

            # Message passing
            for depth in range(self.depth - 1):
                if self.undirected:
                    message = (message + message[b2revb]) / 2

                if self.atom_messages:
                    nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                    nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                    nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                    message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
                else:
                    # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                    # message      a_message = sum(nei_a_message)      rev_message
                    nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                    a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                    rev_message = message[b2revb]  # num_bonds x hidden
                    message = a_message[b2a] - rev_message  # num_bonds x hidden

                message = self.W_h(message)
                message = self.act_func(input + message)  # num_bonds x hidden_size
                message = self.dropout_layer(message)  # num_bonds x hidden

            a2x = a2a if self.atom_messages else a2b
            nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

            # Readout
            mol_vecs = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                    mol_vec = mol_vec.sum(dim=0) / a_size
                    mol_vecs.append(mol_vec)

            mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
            
            if self.use_input_features:
                features_batch = features_batch.to(mol_vecs)
                if len(features_batch.shape) == 1:
                    features_batch = features_batch.view([1, features_batch.shape[0]])
                mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

            ret.append(mol_vecs)  # num_molecules x hidden
        
        return ret[0] - ret[1]


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)

        print("\n Anish: ", "atom dim: ", self.atom_fdim, ". bond dim: ", self.bond_fdim, "\n")

        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)

        self.lig2morf = {}

        #change below code to read in two files and make the map between them (done)

        df1 = pd.read_csv("App2/363.csv")
        df2 = pd.read_csv("App2/363_test_morf.csv")

        #print(df1['smiles'])
        #print(df2['smiles'])

        for i in range(0, len(df1['smiles'])):
            self.lig2morf[df1['smiles'][i]] = df2['smiles'][i]

        #print(self.lig2morf)

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

       # print("\nAnish: The type of the batch is: ", type(batch), "\n")

        if type(batch) != BatchMolGraph:

            batch = mol2graph(batch)

        #for mymol in batch.molgraphs:

        currentsmiles = []

        # open file and read the content in a list
        with open('cursmiles.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                thissmiles = line[:-1]

                # add item to the list
                currentsmiles.append(thissmiles)

        morfsmiles = []
        for i in range(len(currentsmiles)):
            morfsmiles.append(self.lig2morf[currentsmiles[i]])

        batch2 = mol2graph(morfsmiles)

      #  print("Mapping between ligands and morfs: \n")
      #  for i in range(len(currentsmiles)):
      #      print(currentsmiles[i], ", PAIRED WITH ", morfsmiles[i])
     #   print("END\n\n")

    #    if features_batch is None:
     #       print("\n Anish: Yes there are no extra features \n")

        output = self.encoder.forward(batch, batch2, features_batch)
        #output2 = self.encoder.forward(batch2, features_batch)

        #print("\n\n ANISH: explicit outputs")
        #print(output)
        #print(output2)
        #print("\n\n END")

     #   print(list(output.size()))

     #   print("\n\n DIV \n\n")

     #   print(list(output2.size()))

        #ret = (output - output2)

        #print(output)

        return output
