import torch
import pickle
import random
import torch.nn.init as init
from . import utils
import numpy as np
from .model import BatchGFR, GFR
import wandb
#from init_net import InitNet
#from network_utils import nonlinearity_lookup, nonlinearity_derivative_lookup, spectral_radius_power, compute_timescale

def get_random_neurons(n_neurons,params = None, save_path="model/gfr_dataset.pickle", bin_size=20, activation_bin_size=20):
    neurons = []
    if params is None:
        with open(save_path, "rb") as f:
            all_params = pickle.load(f)
        df = all_params[(bin_size, activation_bin_size)]
        cell_ids = df["cell_id"].tolist()

        chosen_ids = random.sample(cell_ids, k=n_neurons)
        for cell_id in chosen_ids:
            neurons.append(utils.load_gfr_model(all_params, cell_id, bin_size, activation_bin_size))
    else:
        for i in range(n_neurons):
            neurons.append(GFR.from_params(params[i]))
        chosen_ids = None
        
    return neurons, chosen_ids

def get_neuron_layer(n_neurons, freeze_g=False, default=False, neuron_parameters = None, bin_size = 20, activation_bin_size = 20):
    if default:
        #these are not biologically realistic parameters, used to train on MNIST
        neurons = [GFR.default() for _ in range(n_neurons)]
    else:
        neurons, _ = get_random_neurons(n_neurons,params = neuron_parameters, bin_size=bin_size, 
            activation_bin_size=activation_bin_size)
    return BatchGFR(neurons, freeze_g=freeze_g)

# GFR-RNN with default parameters

class Network(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim,
            out_dim ,  
            source_unit_indices,
            ids,
            config,
            neuron_parameters = None,
            gain = None,
            bin_size = 100,
            freeze_neurons=False,
            freeze_g=False,
            device=None,
            sparse = False
        ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        
        self.input_layer = torch.nn.Linear(in_dim, hidden_dim)
        #fc2 needs to be replaced by W_measured, gains etc
        #self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        
        if sparse:
            w_measured = torch.empty(hidden_dim, hidden_dim)
            init.sparse_(w_measured, 0.5)
            w_measured = w_measured.to_sparse_csr(size = (hidden_dim, hidden_dim))
        else:
            w_measured = torch.empty(hidden_dim, hidden_dim)
            init.xavier_normal_(w_measured)
        
        self.register_buffer("alpha", torch.tensor(.05))  #weight exponentiation factor
        self.register_buffer("W0", torch.median(torch.abs(w_measured)))  #weight baseline for exponentiation  
        self.register_buffer("Wmin", torch.tensor(0.0))  #minimum weight for exponentiation
        self.register_buffer("Wmax", torch.max(w_measured))  #maximum weight for exponentiation
        
        self.register_buffer("input_layer_zeros", torch.zeros_like(self.input_layer.weight))
        self.register_buffer("W_measured", w_measured)
        self.register_buffer("known_connection_mask", torch.zeros(hidden_dim, hidden_dim))
        self.register_buffer("unknown_connection_mask", torch.ones(hidden_dim, hidden_dim))
        #initialize to all ones for no dales law enforcement
        self.register_buffer("dales_signs", torch.ones(self.W_measured.shape[1]))
        #neurons included in the model
        if ids is not None:
            self.register_buffer("model_ids", torch.from_numpy(ids.astype(np.int64)))
        #subset of neurons whose activity is known
        if source_unit_indices is not None:
            self.register_buffer("source_unit_indices", torch.from_numpy(source_unit_indices.astype(np.int64)))
       
        self.max_rate = torch.nn.Parameter(torch.abs(torch.randn(hidden_dim)))
        if gain is None:
            self.gain = torch.nn.Parameter(torch.ones(hidden_dim,hidden_dim)) 
        else:
            self.gain = torch.nn.Parameter(torch.ones(hidden_dim,hidden_dim)*gain) 
        
        if out_dim is not None:
            self.fc3 = torch.nn.Linear(hidden_dim, out_dim) #output layer 

        self.hidden_neurons = get_neuron_layer(hidden_dim, freeze_g=freeze_g,neuron_parameters = neuron_parameters,
             bin_size = bin_size,activation_bin_size = bin_size)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()

        self.temp_xin = []
        self.temp_xrec = []
    
    def reset(self, batch_size,init_state=None,measured_indices = None):
        self.hidden_neurons.reset(batch_size)
        if init_state is not None:
            if measured_indices is not None:
                state  = torch.zeros(batch_size, self.hidden_dim).to(self.device)
                state[:,measured_indices] = init_state
            else:
                state = init_state
        else:
            state = torch.zeros(batch_size, self.hidden_dim)

        self.xh =  state.to(self.device)  # initial state

    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim).to(self.device)
    
    def set_known_connection_mask(self, known_connection_mask):
        self.register_buffer("known_connection_mask", torch.from_numpy( known_connection_mask.astype(np.float32)))
        self.register_buffer("unknown_connection_mask", torch.from_numpy((1 - known_connection_mask).astype(np.float32)))

    def set_model_ids(self,pt_root_ids):
        self.register_buffer("model_ids", torch.from_numpy(np.array(pt_root_ids)))

    def set_weights(self, recurrent_weights=None,alpha= None,w_baseline = None, input_weights=None, output_weights=None,enforce_dales_law = False):
        #recurrent_weights, if set, must be a listt of length 2 where the first element is the weight matrix
        # and the second element contains the signs of the weights
        #the signs can optionally be constrained by Dale's law during training or not.
        if recurrent_weights is not None:
            del self.W_measured
            #this can also enforce zeros where there are no connections
            W_measured = torch.tensor(recurrent_weights[0].astype(np.float32))
            self.register_buffer("dales_signs", torch.tensor(recurrent_weights[1],dtype = torch.float32))
            self.register_buffer("W_measured", W_measured)
            #signs = torch.where(W_measured ==0,torch.tensor(1.0),torch.sign(W_measured)) 
            
            
            if alpha is not None:
                self.register_buffer("alpha", torch.tensor(alpha))
            if w_baseline is not None:
                self.register_buffer("Wmin", torch.tensor(w_baseline[0]))
                self.register_buffer("W0", torch.tensor(w_baseline[2]))  
                self.register_buffer("Wmax", torch.tensor(w_baseline[-1]))   
                
        if input_weights is not None:
            with torch.no_grad():
                self.input_layer.weight.copy_(torch.from_numpy(input_weights.astype(np.float32)))
                del self.input_layer_zeros
                input_layer_zeros = torch.from_numpy(np.where(input_weights == 0, np.zeros_like(input_weights), np.ones_like(input_weights)).astype(np.float32))
                self.register_buffer("input_layer_zeros", input_layer_zeros)
        if output_weights is not None:
            with torch.no_grad():
                self.output_layer.weight.copy_(torch.from_numpy(output_weights.astype(np.float32)))


    # x: [batch_size, in_dim]
    def forward(self, x):
        #x_in = self.input_layer(x)*10
        x_in = 10.*torch.einsum("ij,j->ij", self.input_layer(x), self.hidden_neurons.g.max_current) / self.in_dim
        #x_rec = self.fc2(self.xh)
        W_eff = self.gain * self.dales_signs * self.W_measured/self.in_dim
        x_rec = 10*(self.xh @ W_eff.T + self.hidden_bias)/ self.hidden_dim
        self.xh = self.hidden_neurons((x_in + x_rec))
        print('xh', self.xh, 'x_in', x_in, 'x_rec', x_rec)
        wandb.log({"state": self.xh, "ext_input": x_in, "rec_input": x_rec})
       
        #out = self.fc3(self.xh)
        return self.xh

# GFR-RNN with biological parameters
class BiologicalGFRNetwork(torch.nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            out_dim, 
            freeze_neurons=True,
            freeze_g=True,
            device=None
        ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            self.fc1.weight.normal_(1.5, 3)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

        self.hidden_neurons = get_neuron_layer(hidden_dim, freeze_g=freeze_g, default=False)
        self.hidden_neurons.device = device
        if freeze_neurons:
            self.hidden_neurons.freeze_parameters()
    
    def reset(self, batch_size):
        self.hidden_neurons.reset(batch_size)
        self.state = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        
    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim).to(self.device)
    
    # x: [batch_size, in_dim]
    def forward(self, x):
        x_in = torch.einsum("ij,j->ij", self.fc1(x), self.hidden_neurons.g.max_current) / self.in_dim
        x_rec = self.fc2(self.state) / self.hidden_dim
        self.state = self.hidden_neurons(x_in + x_rec)
        out = self.fc3(self.state)
        return out