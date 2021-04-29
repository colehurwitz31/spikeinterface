from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from spikeextractors.extraction_tools import check_get_traces_args, check_get_unit_spike_train

import numpy as np
from pathlib import Path
from distutils.version import StrictVersion

try:
    import MEArec as mr
    import neo
    import quantities as pq
    if StrictVersion(mr.__version__) >= '1.5.0':
        HAVE_MREX = True
    else:
        print("MEArec version requires an update (>=1.5). Please upgrade with 'pip install --upgrade MEArec'")
        HAVE_MREX = False
except ImportError:
    HAVE_MREX = False


class MEArecRecordingExtractor(RecordingExtractor):
    extractor_name = 'MEArecRecording'
    has_default_locations = True
    has_unscaled = False
    installed = HAVE_MREX  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"  # error message when not installed

    def __init__(self, file_path, locs_2d=True):
        assert self.installed, self.installed
        self._recording_path = file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._recgen = None
        self._locs_2d = locs_2d
        self._locations = None
        self._initialize()
        RecordingExtractor.__init__(self)

        if self._locations is not None:
            self.set_channel_locations(self._locations)

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'locs_2d': locs_2d}

    def _initialize(self):
        self._recgen = mr.load_recordings(recordings=self._recording_path, return_h5_objects=True, check_suffix=False,
                                          load=['recordings', 'channel_positions'])
        self._fs = self._recgen.info['recordings']['fs']
        self._recordings = self._recgen.recordings
        self._num_frames, self._num_channels = self._recordings.shape
        if len(np.array(self._recgen.channel_positions)) == self._num_channels:
            self._locations = np.array(self._recgen.channel_positions)
            if self._locs_2d:
                if 'electrodes' in self._recgen.info.keys():
                    if 'plane' in self._recgen.info['electrodes'].keys():
                        probe_plane = self._recgen.info['electrodes']['plane']
                        if probe_plane == 'xy':
                            self._locations = self._locations[:, :2]
                        elif probe_plane == 'yz':
                            self._locations = self._locations[:, 1:]
                        elif probe_plane == 'xz':
                            self._locations = self._locations[:, [0, 2]]
                if self._locations.shape[1] == 3:
                    self._locations = self._locations[:, 1:]

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if np.any(np.diff(channel_ids) < 0):
            sorted_channel_ids = np.sort(channel_ids)
            sorted_idx = np.array([list(sorted_channel_ids).index(ch) for ch in channel_ids])
            recordings = self._recordings[start_frame:end_frame, sorted_channel_ids.tolist()]
            return np.array(recordings[:, sorted_idx]).T
        else:
            if sorted(channel_ids) == channel_ids and np.all(np.diff(channel_ids) == 1):
                channel_ids = slice(channel_ids[0], channel_ids[0] + len(channel_ids))
            return np.array(self._recordings[start_frame:end_frame, channel_ids]).T
        
    @staticmethod
    def write_recording(recording, save_path, check_suffix=True):
        """
        Save recording extractor to MEArec format.
        Parameters
        ----------
        recording: RecordingExtractor
            Recording extractor object to be saved
        save_path: str
            .h5 or .hdf5 path
        """
        assert HAVE_MREX, MEArecRecordingExtractor.installation_mesg
        save_path = Path(save_path)
        if save_path.is_dir():
            print("The file will be saved as recording.h5 in the provided folder")
            save_path = save_path / 'recording.h5'
        if (save_path.suffix == '.h5' or save_path.suffix == '.hdf5') or (not check_suffix):
            info = {'recordings': {'fs': recording.get_sampling_frequency()}}
            rec_dict = {'recordings': recording.get_traces().transpose()}
            if 'location' in recording.get_shared_channel_property_names():
                positions = recording.get_channel_locations()
                rec_dict['channel_positions'] = positions
            recgen = mr.RecordingGenerator(rec_dict=rec_dict, info=info)
            mr.save_recording_generator(recgen, str(save_path), verbose=False)
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")


class MEArecSortingExtractor(SortingExtractor):
    extractor_name = 'MEArecSorting'
    installed = HAVE_MREX  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert self.installed, self.installed
        SortingExtractor.__init__(self)
        self._recording_path = file_path
        self._num_units = None
        self._spike_trains = None
        self._unit_ids = None
        self._fs = None
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    def _initialize(self):
        recgen = mr.load_recordings(recordings=self._recording_path, return_h5_objects=True, check_suffix=False,
                                    load=['spiketrains'])
        self._num_units = len(recgen.spiketrains)
        if 'unit_id' in recgen.spiketrains[0].annotations:
            self._unit_ids = [int(st.annotations['unit_id']) for st in recgen.spiketrains]
        else:
            self._unit_ids = list(range(self._num_units))
        self._spike_trains = recgen.spiketrains
        self._fs = recgen.info['recordings']['fs'] * pq.Hz  # fs is in kHz
        self._sampling_frequency = recgen.info['recordings']['fs']

        if 'soma_position' in self._spike_trains[0].annotations:
            for u, st in zip(self._unit_ids, self._spike_trains):
                self.set_unit_property(u, 'soma_location', st.annotations['soma_position'])

    def get_unit_ids(self):
        if self._unit_ids is None:
            self._initialize()
        return self._unit_ids

    def get_num_units(self):
        if self._num_units is None:
            self._initialize()
        return self._num_units

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if self._spike_trains is None:
            self._initialize()
        times = (self._spike_trains[self.get_unit_ids().index(unit_id)].times.rescale('s') *
                 self._fs.rescale('Hz')).magnitude
        inds = np.where((start_frame <= times) & (times < end_frame))
        return np.rint(times[inds]).astype(int)

    @staticmethod
    def write_sorting(sorting, save_path, sampling_frequency, check_suffix=True):
        """
        Save sorting extractor to MEArec format.
        Parameters
        ----------
        sorting: SortingExtractor
            Sorting extractor object to be saved
        save_path: str
            .h5 or .hdf5 path
        sampling_frequency: int
            Sampling frequency in Hz

        """
        assert HAVE_MREX, MEArecSortingExtractor.installation_mesg
        save_path = Path(save_path)
        if save_path.is_dir():
            print("The file will be saved as sorting.h5 in the provided folder")
            save_path = save_path / 'sorting.h5'
        if (save_path.suffix == '.h5' or save_path.suffix == '.hdf5') or (not check_suffix):
            # create neo spike trains
            spiketrains = []
            for u in sorting.get_unit_ids():
                st = neo.SpikeTrain(times=sorting.get_unit_spike_train(u) / float(sampling_frequency) * pq.s,
                                    t_start=np.min(sorting.get_unit_spike_train(u) / float(sampling_frequency)) * pq.s,
                                    t_stop=np.max(sorting.get_unit_spike_train(u) / float(sampling_frequency)) * pq.s)
                st.annotate(unit_id=u)
                spiketrains.append(st)

            assert len(spiketrains) > 0, """
                The sorting for output contains no unit, please check the sorting.
            """

            duration = np.max([st.t_stop.magnitude for st in spiketrains])
            info = {'recordings': {'fs': sampling_frequency}, 'spiketrains': {'duration': duration}}
            rec_dict = {'spiketrains': spiketrains}
            recgen = mr.RecordingGenerator(rec_dict=rec_dict, info=info)
            mr.save_recording_generator(recgen, str(save_path), verbose=False)
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")
