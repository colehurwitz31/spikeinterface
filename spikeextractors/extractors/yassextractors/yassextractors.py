import numpy as np
from pathlib import Path

from spikeextractors import SortingExtractor
from spikeextractors.extractors.numpyextractors import NumpyRecordingExtractor
from spikeextractors.extraction_tools import check_get_unit_spike_train

try:
    import yaml
    HAVE_YASS = True
except:
    HAVE_YASS = False


class YassSortingExtractor(SortingExtractor):

    extractor_name = 'YassSorting'
    mode = 'folder'
    installed = HAVE_YASS  # check at class level if installed or not

    has_default_locations = False
    is_writable = False
    installation_mesg = "To use the Yass extractor, install pyyaml: \n\n pip install pyyaml\n\n"  # error message when not installed
    
    
    def __init__(self, folder_path):
        assert self.installed, self.installation_mesg
        SortingExtractor.__init__(self)

        self.root_dir = folder_path
        r = Path(self.root_dir)

        self.fname_spike_train = r / 'tmp' / 'output' / 'spike_train.npy'
        self.fname_templates = r /'tmp' / 'output' / 'templates' / 'templates_0sec.npy'
        self.fname_config = r / 'config.yaml'
        
        
        # set defaults to None so they are only loaded if user requires them
        
        self.spike_train = None
        self.temps = None

        # Read CONFIG File
        with open(self.fname_config, 'r') as stream:
            self.config = yaml.safe_load(stream)
        
        self._sampling_frequency = self.config['recordings']['sampling_rate']

    def get_unit_ids(self):

        if self.spike_train is None:
            self.spike_train = np.load(self.fname_spike_train)
        
        unit_ids = np.unique(self.spike_train[:,1])
        
        return unit_ids
    
    def get_temps(self):

        # Electrical images/templates.
        
        if self.temps is None:
            self.temps = np.load(self.fname_templates)
                    
        return self.temps

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        """Code to extract spike frames from the specified unit.
        """

        if self.spike_train is None:
            self.spike_train = np.load(self.fname_spike_train)
        
        # find unit id spike times
        idx = np.where(spike_train[:,1]==unit_id)[0]
        spike_times = spike_train[idx,0]

        # find spike times
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = 1E50 # use large time

        idx2 = np.where(np.logical_and(spike_times>=start_frame, spike_times<end_frame))[0]
        spike_times = spike_times[idx2]

        return spike_times
    
    
    
