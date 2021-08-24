# Need these for BMI
from bmipy import Bmi
import time
import data_tools
# Basic utilities
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
# Here is the LSTM model we want to run
import nextgen_cuda_lstm
# Configuration file functionality
from neuralhydrology.utils.config import Config
# LSTM here is based on PyTorch
import torch
from torch import nn

class bmi_LSTM(Bmi):

    def __init__(self):
        """Create a Bmi LSTM model that is ready for initialization."""
        super(bmi_LSTM, self).__init__()
        print('thank you for choosing LSTM')
        self._model = None
        self._values = {}
        self._var_units = {}
        self._var_loc = {}
        self._grids = {}
        self._grid_type = {}
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        self._time_units = "s"

    #----------------------------------------------
    # Required, static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'LSTM for Next Generation NWM',
        'version':            '1.0',
        'author_name':        'Jonathan Martin Frame',
        'grid_type':          'none',
        'time_step_type':     'donno',
        'step_method':        'none',
        'time_units':         '1 hour' }

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    # LWDOWN,PSFC,Q2D,RAINRATE,SWDOWN,T2D,U2D,V2D
    #---------------------------------------------
    _input_var_names = [
        'land_surface_radiation~incoming~longwave__energy_flux',
        'land_surface_air__pressure',
        'atmosphere_air_water~vapor__relative_saturation',
        'atmosphere_water__liquid_equivalent_precipitation_rate',
        'land_surface_radiation~incoming~shortwave__energy_flux',
        'land_surface_air__temperature',
        'land_surface_wind__x_component_of_velocity',
        'land_surface_wind__y_component_of_velocity']

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['land_surface_water__runoff_volume_flux']

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    # This is going to get long, 
    #     since the input variable names could come from any forcing...
    #------------------------------------------------------
    _var_name_map_long_first = {'atmosphere_water__liquid_equivalent_precipitation_rate':'total_precipitation',
                     'land_surface_air__temperature':'temperature',
                     'basin__mean_of_elevation':'elev_mean',
                     'basin__mean_of_slope':'slope_mean'}
    _var_name_map = {'total_precipitation':'atmosphere_water__liquid_equivalent_precipitation_rate',
                     'temperature':'land_surface_air__temperature',
                     'elev_mean':'basin__mean_of_elevation',
                     'slope_mean':'basin__mean_of_slope'}

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the units of each model variable.
    #------------------------------------------------------
    _var_units_map = {
        'land_surface_water__runoff_volume_flux':'mm',
        #--------------------------------------------------
         'land_surface_radiation~incoming~longwave__energy_flux':'W m-2',
         'land_surface_air__pressure':'Pa',
         'atmosphere_air_water~vapor__relative_saturation':'kg kg-1',
         'atmosphere_water__liquid_equivalent_precipitation_rate':'kg m-2',
         'land_surface_radiation~incoming~shortwave__energy_flux':'W m-2',
         'land_surface_air__temperature':'K',
         'land_surface_wind__x_component_of_velocity':'m s-1',
         'land_surface_wind__y_component_of_velocity':'m s-1'}

    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print('###################################################')
            print(' ERROR: Could not find attribute: ' + att_name)
            print('###################################################')
            print()

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #def get_value(self, var_name, dest):
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        #dest[:] = self.get_value_ptr(var_name).flatten()
        return self.get_value_ptr(var_name)
        #return dest

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):

        return str( type(self.get_value( long_var_name )) )

    #-------------------------------------------------------------------
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return 0.0

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return (self.n_steps * self.dt)


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.dt

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, long_var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        var_name = self.get_var_name( long_var_name )
        setattr( self, var_name, value ) 
 
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 

    #-------------------------------------------------------------------
    def initialize( self, bmi_cfg_file=None ):
        
        # First read in the BMI configuration. This will direct all the next moves.
        if bmi_cfg_file is not None:
            self.cfg_bmi = Config._read_and_parse_config(bmi_cfg_file)
        else:
            print("Error: No configuration provided, nothing to do...")
    
        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] >= 1:
            print("Initializing LSTM")

        # Now load in the configuration file for the specific LSTM
        # This will include all the details about how the model was trained
        # Inputs, outputs, hyper-parameters, etc.
        if self.cfg_bmi['train_cfg_file'] is not None:
            self.cfg_train = Config._read_and_parse_config(self.cfg_bmi['train_cfg_file'])

        # Collect the LSTM model architecture details from the configuration file
        self.input_size        = len(self.cfg_train['dynamic_inputs']) + len(self.cfg_train['static_attributes'])
        self.hidden_layer_size = self.cfg_train['hidden_size']
        self.output_size       = len(self.cfg_train['target_variables']) 
        self.batch_size        = 1 #self.cfg_train['batch_size']

        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] >=5:
            print('LSTM model architecture')
            print('input size', type(self.input_size), self.input_size)
            print('hidden layer size', type(self.hidden_layer_size), self.hidden_layer_size)
            print('output size', type(self.output_size), self.output_size)
        
        # Now we need to initialize an LSTM model.
        self.lstm = nextgen_cuda_lstm.Nextgen_CudaLSTM(input_size=self.input_size, 
                                                       hidden_layer_size=self.hidden_layer_size, 
                                                       output_size=self.output_size, 
                                                       batch_size=1, 
                                                       seq_length=1)

        # load in the model specific values (scalers, weights, etc.)

        # Scaler data from the training set. This is used to normalize the data (input and output).
        with open(self.cfg_train['run_dir'] / 'train_data' / 'train_data_scaler.p', 'rb') as fb:
            self.train_data_scaler = pickle.load(fb)

        # Mean and standard deviation for the inputs and LSTM outputs 
        self.out_mean = self.train_data_scaler['xarray_feature_center']['qobs_mm_per_hour'].values
        self.out_std = self.train_data_scaler['xarray_feature_scale']['qobs_mm_per_hour'].values
        self.input_mean = [self.train_data_scaler['xarray_feature_center'][x].values for x in self.cfg_train['dynamic_inputs']]
        self.input_mean.extend([self.train_data_scaler['attribute_means'][x] for x in self.cfg_train['static_attributes']])
        self.input_std = [self.train_data_scaler['xarray_feature_scale'][x].values for x in self.cfg_train['dynamic_inputs']]
        self.input_std.extend([self.train_data_scaler['attribute_means'][x] for x in self.cfg_train['static_attributes']]) 
        self.input_mean = np.array(self.input_mean)
        self.input_std = np.array(self.input_std)

        # Save the default model weights. We need to make sure we have the same keys.
        default_state_dict = self.lstm.state_dict()

        # Trained model weights from Neuralhydrology.
        trained_model_file = self.cfg_train['run_dir'] / 'model_epoch{}.pt'.format(str(self.cfg_train['epochs']).zfill(3))
        trained_state_dict = torch.load(trained_model_file, map_location=torch.device('cpu'))

        # Changing the name of the head weights, since different in NH
        trained_state_dict['head.weight'] = trained_state_dict.pop('head.net.0.weight')
        trained_state_dict['head.bias'] = trained_state_dict.pop('head.net.0.bias')
        trained_state_dict = {x:trained_state_dict[x] for x in default_state_dict.keys()}

        # Load in the trained weights.
        self.lstm.load_state_dict(trained_state_dict)

        # ----    Initialize the values for the input to the LSTM    ---- #
        self.set_static_attributes()
        self.initialize_forcings()
        self.all_lstm_inputs = []
        self.all_lstm_inputs.extend(self.cfg_train['dynamic_inputs'])
        self.all_lstm_inputs.extend(self.cfg_train['static_attributes'])

        self._values = {'atmosphere_water__liquid_equivalent_precipitation_rate':self.total_precipitation,
                        'land_surface_air__temperature':self.temperature,
                        'basin__mean_of_elevation':self.elev_mean,
                        'basin__mean_of_slope':self.slope_mean}

        self.t = 0
        
        if self.cfg_bmi['initial_state'] == 'zero':
            self.h_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()
            self.c_t = torch.zeros(1, self.batch_size, self.hidden_layer_size).float()

        self.output_factor =  self.cfg_bmi['area_sqkm'] * 35.315 # from m3/s to ft3/s

        # ----    print some stuff for troubleshooting    ---- #
        if self.cfg_bmi['verbose'] >=5:
            print('out_mean:', self.out_mean)
            print('out_std:', self.out_std)
    
    #------------------------------------------------------------ 
    def create_scaled_input_tensor(self):
        self.input_array = np.array([self._values[self._var_name_map[x]] for x in self.all_lstm_inputs])
        self.input_array_scaled = self.input_array * self.input_std + self.input_mean
        self.input_tensor = torch.tensor(self.input_array_scaled)

    #------------------------------------------------------------ 
    def scale_output(self):
        self.streamflow = (self.lstm_output[0,0,0].numpy().tolist() * self.out_std + self.out_mean) * self.output_factor

    #------------------------------------------------------------ 
    def update(self):
        with torch.no_grad():
            
            print('updating LSTM for t: ', self.t)

            self.create_scaled_input_tensor()

            self.lstm_output, self.h_t, self.c_t = self.lstm.forward(self.input_tensor, self.h_t, self.c_t)
            
            self.scale_output()
            
            self.t += 1
            
            print('for time: {} lstm output: {}'.format(self.t,self.streamflow))
    
    #------------------------------------------------------------ 
    def update_until(self, last_update):
        first_update=self.t
        for t in range(first_update, last_update):
            self.update()
    #------------------------------------------------------------    
    def finalize( self ):
        return 0

    #-------------------------------------------------------------------
    def read_initial_states(self):
        h_t = np.genfromtxt(self.h_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.h_t = torch.tensor(h_t).view(1,1,-1)
        c_t = np.genfromtxt(self.c_t_init_file, skip_header=1, delimiter=",")[:,1]
        self.c_t = torch.tensor(c_t).view(1,1,-1)

    #-------------------------------------------------------------------
    def load_scalers(self):

        with open(self.scaler_file, 'rb') as fb:
            self.scalers = pickle.load(fb)

    #---------------------------------------------------------------------------- 
    def set_static_attributes(self):
        #------------------------------------------------------------ 
        if 'elev_mean' in self.cfg_train['static_attributes']:
            self.elev_mean = self.cfg_bmi['elev_mean']
        #------------------------------------------------------------ 
        if 'slope_mean' in self.cfg_train['static_attributes']:
            self.slope_mean = self.cfg_bmi['slope_mean']
        #------------------------------------------------------------ 
    
    #---------------------------------------------------------------------------- 
    def initialize_forcings(self):
        #------------------------------------------------------------ 
        if 'total_precipitation' in self.cfg_train['dynamic_inputs']:
            self.total_precipitation = 0
        #------------------------------------------------------------ 
        if 'temperature' in self.cfg_train['dynamic_inputs']:
            self.temperature = 0
        #------------------------------------------------------------ 

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self._name

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    #------------------------------------------------------------ 
    def get_var_nbytes(self, var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        dest[:] = self.get_value_ptr(var_name).take(indices)
        return dest

    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    def get_grid_rank(self, grid_id):
        raise NotImplementedError("get_grid_rank") 

    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    def get_grid_size(self, grid_id):
        raise NotImplementedError("get_grid_size") 

    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    def get_grid_type(self):
        raise NotImplementedError("get_grid_type") 

    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 

    def get_var_grid(self):
        raise NotImplementedError("get_var_grid") 

    def get_var_itemsize(self, name):
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_location(self, name):
        return self._var_loc[name]



