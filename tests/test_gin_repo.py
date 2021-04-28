import tempfile
import unittest
from pathlib import Path
import numpy as np
import sys

from datalad.api import install, Dataset
from parameterized import parameterized

import spikeextractors as se
from spikeextractors.testing import check_recordings_equal, check_sortings_equal

run_local = False
test_nwb = True
test_caching = True

if sys.platform == "linux" or run_local:
    class TestNwbConversions(unittest.TestCase):

        def setUp(self):
            pt = Path.cwd() / 'ephy_testing_data'
            if pt.exists():
                self.dataset = Dataset(pt)
            else:
                self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')
            self.savedir = Path(tempfile.mkdtemp())

        @parameterized.expand([
            (
                se.AxonaRecordingExtractor,
                "axona",
                dict(filename=str(Path.cwd() / "ephy_testing_data" / "axona" / "axona_raw.set"))
            ),
            (
                se.BlackrockRecordingExtractor,
                "blackrock/blackrock_2_1",
                dict(
                    filename=str(Path.cwd() / "ephy_testing_data" / "blackrock" / "blackrock_2_1" / "l101210-001"),
                    seg_index=0,
                    nsx_to_load=5
                )
            ),
            (
                se.IntanRecordingExtractor,
                "intan",
                dict(file_path=Path.cwd() / "ephy_testing_data" / "intan" / "intan_rhd_test_1.rhd")
            ),
            (
                se.IntanRecordingExtractor,
                "intan",
                dict(file_path=Path.cwd() / "ephy_testing_data" / "intan" / "intan_rhs_test_1.rhs")
            ),
            # Klusta - no .prm config file in ephy_testing
            # (
            #     se.KlustaRecordingExtractor,
            #     "kwik",
            #     dict(folder_path=Path.cwd() / "ephy_testing_data" / "kwik")
            # ),
            (
                se.MEArecRecordingExtractor,
                "mearec/mearec_test_10s.h5",
                dict(file_path=Path.cwd() / "ephy_testing_data" / "mearec" / "mearec_test_10s.h5")
            ),
            (
                se.NeuralynxRecordingExtractor,
                "neuralynx/Cheetah_v5.7.4/original_data",
                dict(
                    dirname=Path.cwd() / "ephy_testing_data" / "neuralynx" / "Cheetah_v5.7.4" / "original_data",
                    seg_index=0
                )
            ),
            (
                se.NeuroscopeRecordingExtractor,
                "neuroscope/test1",
                dict(file_path=Path.cwd() / "ephy_testing_data" / "neuroscope" / "test1" / "test1.dat")
            ),
            # Nixio - RuntimeError: Cannot open non-existent file in ReadOnly mode!
            # (
            #     se.NIXIORecordingExtractor,
            #     "nix",
            #     dict(file_path=str(Path.cwd() / "ephy_testing_data" / "neoraw.nix"))
            # ),
            (
                se.OpenEphysRecordingExtractor,
                "openephys/OpenEphys_SampleData_1",
                dict(folder_path=Path.cwd() / "ephy_testing_data" / "openephys" / "OpenEphys_SampleData_1")
            ),
            (
                se.OpenEphysRecordingExtractor,
                "openephysbinary/v0.4.4.1_with_video_tracking",
                dict(folder_path=Path.cwd() / "ephy_testing_data" / "openephysbinary" / "v0.4.4.1_with_video_tracking")
            ),
            (
                se.OpenEphysNPIXRecordingExtractor,
                "openephysbinary/v0.5.3_two_neuropixels_stream",
                dict(
                    folder_path=Path.cwd() / "ephy_testing_data" / "openephysbinary" / "v0.5.3_two_neuropixels_stream"
                                / "Record_Node_107")
            ),
            (
                se.NeuropixelsDatRecordingExtractor,
                "openephysbinary/v0.5.3_two_neuropixels_stream",
                dict(
                    file_path=Path.cwd() / "ephy_testing_data" / "openephysbinary" / "v0.5.3_two_neuropixels_stream" /
                              "Record_Node_107" / "experiment1" / "recording1" / "continuous" /
                              "Neuropix-PXI-116.0" / "continuous.dat",
                    settings_file=Path.cwd() / "ephy_testing_data" / "openephysbinary" /
                                  "v0.5.3_two_neuropixels_stream" / "Record_Node_107" / "settings.xml")
            ),
            (
                se.PhyRecordingExtractor,
                "phy/phy_example_0",
                dict(folder_path=Path.cwd() / "ephy_testing_data" / "phy" / "phy_example_0")
            ),
            # Plexon - AssertionError: This file have several channel groups spikeextractors support only one groups
            # (
            #     se.PlexonRecordingExtractor,
            #     "plexon",
            #     dict(filename=Path.cwd() / "ephy_testing_data" / "plexon" / "File_plexon_2.plx")
            # ),
            (
                se.SpikeGLXRecordingExtractor,
                "spikeglx/Noise4Sam_g0",
                dict(
                    file_path=Path.cwd() / "ephy_testing_data" / "spikeglx" / "Noise4Sam_g0" / "Noise4Sam_g0_imec0" /
                    "Noise4Sam_g0_t0.imec0.ap.bin"
                )
            )
        ])
        def test_convert_recording_extractor_to_nwb(self, se_class, dataset_path, se_kwargs):
            print(f"\n\n\n TESTING {se_class.extractor_name}...")
            dataset_stem = Path(dataset_path).stem
            self.dataset.get(dataset_path)
            recording = se_class(**se_kwargs)


            # # test writing to NWB
            if test_nwb:
                nwb_save_path = self.savedir / f"{se_class.__name__}_test_{dataset_stem}.nwb"
                se.NwbRecordingExtractor.write_recording(recording, nwb_save_path, write_scaled=True)
                nwb_recording = se.NwbRecordingExtractor(nwb_save_path)
                check_recordings_equal(recording, nwb_recording)

                if recording.has_unscaled:
                    nwb_save_path_unscaled = self.savedir / f"{se_class.__name__}_test_{dataset_stem}_unscaled.nwb"
                    if np.all(recording.get_channel_offsets() == 0):
                        se.NwbRecordingExtractor.write_recording(recording, nwb_save_path_unscaled, write_scaled=False)
                        nwb_recording = se.NwbRecordingExtractor(nwb_save_path_unscaled)
                        check_recordings_equal(recording, nwb_recording, return_scaled=False)
                        # Skip check when NWB converts uint to int
                        if recording.get_dtype(return_scaled=False) == nwb_recording.get_dtype(return_scaled=False):
                            check_recordings_equal(recording, nwb_recording, return_scaled=True)

            # test caching
            if test_caching:
                rec_cache = se.CacheRecordingExtractor(recording)
                check_recordings_equal(recording, rec_cache)
                if recording.has_unscaled:
                    rec_cache_unscaled = se.CacheRecordingExtractor(recording, return_scaled=False)
                    check_recordings_equal(recording, rec_cache_unscaled, return_scaled=False)
                    check_recordings_equal(recording, rec_cache_unscaled, return_scaled=True)

        @parameterized.expand([
            (
                se.BlackrockSortingExtractor,
                "blackrock/blackrock_2_1",
                dict(
                    filename=str(Path.cwd() / "ephy_testing_data" / "blackrock" / "blackrock_2_1" / "l101210-001"),
                    seg_index=0,
                    nsx_to_load=5
                 )
            ),
            (
                se.KlustaSortingExtractor,
                "kwik",
                dict(file_or_folder_path=Path.cwd() / "ephy_testing_data" / "kwik" / "neo.kwik")
            ),
            # Neuralynx - units_ids = nwbfile.units.id[:] - AttributeError: 'NoneType' object has no attribute 'id'
            # Is the GIN data OK? Or are there no units?
            # (
            #     se.NeuralynxSortingExtractor,
            #     "neuralynx/Cheetah_v5.7.4/original_data",
            #     dict(
            #         dirname=Path.cwd() / "ephy_testing_data" / "neuralynx" / "Cheetah_v5.7.4" / "original_data",
            #         seg_index=0
            #     )
            # ),
            # NIXIO - return [int(da.label) for da in self._spike_das]
            # TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'
            # (
            #     se.NIXIOSortingExtractor,
            #     "nix/nixio_fr.nix",
            #     dict(file_path=str(Path.cwd() / "ephy_testing_data" / "nix" / "nixio_fr.nix"))
            # ),
            (
                se.MEArecSortingExtractor,
                "mearec/mearec_test_10s.h5",
                dict(file_path=Path.cwd() / "ephy_testing_data" / "mearec" / "mearec_test_10s.h5")
            ),
            (
                se.PhySortingExtractor,
                "phy/phy_example_0",
                dict(folder_path=Path.cwd() / "ephy_testing_data" / "phy" / "phy_example_0")
            ),
            (
                se.PlexonSortingExtractor,
                "plexon",
                dict(filename=Path.cwd() / "ephy_testing_data" / "plexon" / "File_plexon_2.plx")
            ),
            (
                se.SpykingCircusSortingExtractor,
                "spykingcircus/spykingcircus_example0",
                dict(
                    file_or_folder_path=Path.cwd() / "ephy_testing_data" / "spykingcircus" / "spykingcircus_example0" /
                                        "recording"
                )
            ),
            # # Tridesclous - dataio error, GIN data is not correct?
            # (
            #     se.TridesclousSortingExtractor,
            #     "tridesclous/tdc_example0",
            #     dict(folder_path=Path.cwd() / "ephy_testing_data" / "tridesclous" / "tdc_example0")
            # )
        ])
        def test_convert_sorting_extractor_to_nwb(self, se_class, dataset_path, se_kwargs):
            print(f"\n\n\n TESTING {se_class.extractor_name}...")
            dataset_stem = Path(dataset_path).stem
            self.dataset.get(dataset_path)

            sorting = se_class(**se_kwargs)
            sf = sorting.get_sampling_frequency()
            if sf is None:  # need to set dummy sampling frequency since no associated acquisition in file
                sf = 30000
                sorting.set_sampling_frequency(sf)

            if test_nwb:
                nwb_save_path = self.savedir / f"{se_class.__name__}_test_{dataset_stem}.nwb"
                se.NwbSortingExtractor.write_sorting(sorting, nwb_save_path)
                nwb_sorting = se.NwbSortingExtractor(nwb_save_path, sampling_frequency=sf)
                check_sortings_equal(sorting, nwb_sorting)

            if test_caching:
                sort_cache = se.CacheSortingExtractor(sorting)
                check_sortings_equal(sorting, sort_cache)


if __name__ == '__main__':
    unittest.main()
