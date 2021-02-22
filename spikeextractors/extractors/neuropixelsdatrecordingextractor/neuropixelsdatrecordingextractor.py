from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
import warnings

try:
    import xmltodict
    HAVE_XMLTODICT = True
except ImportError:
    HAVE_XMLTODICT = False


class NeuropixelsDatRecordingExtractor(BinDatRecordingExtractor):
    """
    Read raw Neurpoixels recordings from Open Ephys dat file and settings.xml
    
    This extractor is currently compatible with the 960 channel Neuropixels probes,
    where a maximum of 384 channels are recorded simulatenously. The array 
    configuration can be specified by passing the settings.xml file created by 
    OpenEphys (it can be found in the directory tree with teh recordings). If this 
    is not provided, the default configuration using 384 channels at the probe tip 
    will be used (a is warning printed).
   
    Parameters
    ----------
    file_path: str
        The raw data file (usually continuous.dat)
    settings_file: None or str
        The file settings.xml generated by OpenEphys containing the array 
        configuration. If not provided the default configuration using 384 
        channels at the probe tip will be used.
    verbose: bool
        Print probe configuration
    
    """
    extractor_name = 'NeuropixelsDatRecording'
    has_default_locations = False
    installed = HAVE_XMLTODICT
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the NeuropixelsDat extractor, install xmltodict: \n\n pip install xmltodict\n\n"

    def __init__(self, file_path, settings_file=None, is_filtered=None, verbose=False):
        assert HAVE_XMLTODICT, self.installation_mesg
        source_dir = Path(Path(__file__).parent)
        self._settings_file = settings_file
        datfile = Path(file_path)
        time_axis = 0
        dtype = 'int16'
        sampling_frequency = float(30000)
        offset = 0
        
        channel_locations = np.loadtxt(source_dir / 'channel_positions_neuropixels.txt')
        if self._settings_file is not None:
            with open(self._settings_file) as f:
                xmldata = f.read()
                settings = xmltodict.parse(xmldata)['SETTINGS']
            channel_info = settings['SIGNALCHAIN']['PROCESSOR'][0]['CHANNEL_INFO']
            channels = settings['SIGNALCHAIN']['PROCESSOR'][0]['CHANNEL']
            recorded_channels = []
            for c in channels:
                if c['SELECTIONSTATE']['@record'] == '1':
                    recorded_channels.append(int(c['@number']))
            used_channels = []
            used_channel_gains = []
            for c in channel_info['CHANNEL']:
                if 'AP' in c['@name'] and int(c['@number']) in recorded_channels:
                    used_channels.append(int(c['@number']))        
                    used_channel_gains.append(float(c['@gain']))
            if verbose:
                print(f'{len(recorded_channels)} total channels found, with {len(used_channels)} recording AP')
                print(f'Channels used:\n{used_channels}')
            numchan = len(used_channels)
            geom = channel_locations[:,np.array(used_channels)].T
            gain = used_channel_gains[0]
            channels = used_channels
        else:
            warnings.warn("No information about this recording available,"
                          "using a default of 384 channels at the probe tip."
                          "If the recording differs, use settings_file=settings.xml")
            numchan = 384
            geom = channel_locations[:,:384].T
            gain = None
            channels = range(384)

        BinDatRecordingExtractor.__init__(self, file_path=datfile, numchan=numchan, dtype=dtype,
                                          sampling_frequency=sampling_frequency, gain=gain, offset=offset, geom=geom,
                                          recording_channels=channels, time_axis=time_axis, is_filtered=is_filtered)

        self._kwargs = {'filename': str(Path(file_path).absolute()), 'settings_file': settings_file,
                        'is_filtered': is_filtered}
