import tempfile
import unittest
from pathlib import Path

from datalad.api import install, Dataset
from parameterized import parameterized

import spikeextractors as se
from spikeextractors.testing import check_recordings_equal


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        pt = Path.cwd() / 'ephy_testing_data'
        if pt.exists():
            self.dataset = Dataset(pt)
        else:
            self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')
        self.savedir = Path(tempfile.mkdtemp())

    @parameterized.expand([
        #(
        #    se.NeuralynxRecordingExtractor,
        #    'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #    'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #    'neuralynx_test.nwb',
        #    'neuralynx_test.Ncs'
        #)
        (
            se.NeuroscopeRecordingExtractor,
            "neuroscope/test1",
            "neuroscope/test1/test1.dat"
        ),
        (
            se.IntanRecordingExtractor,
            "intan/intan_rhd_test_1.rhd",
            "intan/intan_rhd_test_1.rhd"
        ),
        (
            se.IntanRecordingExtractor,
            "intan/intan_rhd_test_1.rhs",
            "intan/intan_rhd_test_1.rhs"
        )
    ])
    def test_convert_recording_extractor_to_nwb(self, se_class, dataset_path, se_path_arg):
        nwb_fname = f"{se_class.__name__}_test.nwb"
        nwb_save_path = self.savedir / nwb_fname
        self.dataset.get(dataset_path)

        re = se_class(Path.cwd() / "ephy_testing_data" / se_path_arg)
        se.NwbRecordingExtractor.write_recording(re, nwb_save_path)
        nwb_re = se.NwbRecordingExtractor(nwb_save_path)
        check_recordings_equal(re, nwb_re)


if __name__ == '__main__':
    unittest.main()
