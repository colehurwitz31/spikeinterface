from .extractors.mdaextractors.mdaextractors import MdaRecordingExtractor, MdaSortingExtractor
from .extractors.mearecextractors.mearecextractors import MEArecRecordingExtractor, MEArecSortingExtractor
from .extractors.biocamrecordingextractor.biocamrecordingextractor import BiocamRecordingExtractor
from .extractors.exdirextractors.exdirextractors import ExdirRecordingExtractor, ExdirSortingExtractor
from .extractors.intanrecordingextractor.intanrecordingextractor import IntanRecordingExtractor
from .extractors.hdsortsortingextractor.hdsortsortingextractor import HDSortSortingExtractor
from .extractors.hs2sortingextractor.hs2sortingextractor import HS2SortingExtractor
from .extractors.klustaextractors.klustaextractors import KlustaSortingExtractor, KlustaRecordingExtractor
from .extractors.kilosortextractors.kilosortextractors import KiloSortSortingExtractor, KiloSortRecordingExtractor
from .extractors.numpyextractors.numpyextractors import NumpyRecordingExtractor, NumpySortingExtractor
from .extractors.nwbextractors.nwbextractors import NwbRecordingExtractor, NwbSortingExtractor
from .extractors.openephysextractors.openephysextractors import OpenEphysRecordingExtractor, \
    OpenEphysSortingExtractor, OpenEphysNPIXRecordingExtractor
from .extractors.maxwellextractors import MaxOneRecordingExtractor, MaxOneSortingExtractor, MaxTwoRecordingExtractor, \
    MaxTwoSortingExtractor
from .extractors.phyextractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor
from .extractors.bindatrecordingextractor.bindatrecordingextractor import BinDatRecordingExtractor
from .extractors.spykingcircusextractors.spykingcircusextractors import SpykingCircusSortingExtractor, \
    SpykingCircusRecordingExtractor
from .extractors.spikeglxrecordingextractor.spikeglxrecordingextractor import SpikeGLXRecordingExtractor
from .extractors.tridescloussortingextractor.tridescloussortingextractor import TridesclousSortingExtractor
from .extractors.npzsortingextractor.npzsortingextractor import NpzSortingExtractor
from .extractors.mcsh5recordingextractor.mcsh5recordingextractor import MCSH5RecordingExtractor
from .extractors.shybridextractors import SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor
from .extractors.nixioextractors.nixioextractors import NIXIORecordingExtractor, NIXIOSortingExtractor
from .extractors.neoextractors import (AxonaRecordingExtractor, PlexonRecordingExtractor, PlexonSortingExtractor,
                                       NeuralynxRecordingExtractor, NeuralynxSortingExtractor,
                                       BlackrockRecordingExtractor, BlackrockSortingExtractor,
                                       MCSRawRecordingExtractor, SpikeGadgetsRecordingExtractor)
from .extractors.neuroscopeextractors import NeuroscopeRecordingExtractor, NeuroscopeMultiRecordingTimeExtractor, \
    NeuroscopeSortingExtractor, NeuroscopeMultiSortingExtractor
from .extractors.waveclussortingextractor import WaveClusSortingExtractor
from .extractors.yassextractors import YassSortingExtractor
from .extractors.combinatosortingextractor import CombinatoSortingExtractor
from .extractors.alfsortingextractor import ALFSortingExtractor
from .extractors.cedextractors import CEDRecordingExtractor
from .extractors.cellexplorersortingextractor import CellExplorerSortingExtractor
from .extractors.neuropixelsdatrecordingextractor import NeuropixelsDatRecordingExtractor

recording_extractor_full_list = [
    MdaRecordingExtractor,
    MEArecRecordingExtractor,
    BiocamRecordingExtractor,
    ExdirRecordingExtractor,
    OpenEphysRecordingExtractor,
    OpenEphysNPIXRecordingExtractor,
    IntanRecordingExtractor,
    BinDatRecordingExtractor,
    KlustaRecordingExtractor,
    KiloSortRecordingExtractor,
    SpykingCircusRecordingExtractor,
    SpikeGLXRecordingExtractor,
    PhyRecordingExtractor,
    MaxOneRecordingExtractor,
    MaxTwoRecordingExtractor,
    MCSH5RecordingExtractor,
    SHYBRIDRecordingExtractor,
    NIXIORecordingExtractor,
    NwbRecordingExtractor,
    NeuroscopeRecordingExtractor,
    NeuroscopeMultiRecordingTimeExtractor,
    CEDRecordingExtractor,
    NeuropixelsDatRecordingExtractor,

    # neo based
    AxonaRecordingExtractor,
    PlexonRecordingExtractor,
    NeuralynxRecordingExtractor,
    BlackrockRecordingExtractor,
    MCSRawRecordingExtractor,
    SpikeGadgetsRecordingExtractor,
]

recording_extractor_dict = {recording_class.extractor_name: recording_class
                            for recording_class in recording_extractor_full_list}
installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

sorting_extractor_full_list = [
    MdaSortingExtractor,
    MEArecSortingExtractor,
    ExdirSortingExtractor,
    HDSortSortingExtractor,
    HS2SortingExtractor,
    KlustaSortingExtractor,
    KiloSortSortingExtractor,
    OpenEphysSortingExtractor,
    PhySortingExtractor,
    SpykingCircusSortingExtractor,
    TridesclousSortingExtractor,
    MaxTwoSortingExtractor,
    MaxOneSortingExtractor,
    NpzSortingExtractor,
    SHYBRIDSortingExtractor,
    NIXIOSortingExtractor,
    NeuroscopeSortingExtractor,
    NeuroscopeMultiSortingExtractor,
    NwbSortingExtractor,
    WaveClusSortingExtractor,
    YassSortingExtractor,
    CombinatoSortingExtractor,
    ALFSortingExtractor,
    # neo based
    PlexonSortingExtractor,
    NeuralynxSortingExtractor,
    BlackrockSortingExtractor,
    CellExplorerSortingExtractor
]

installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]
sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in sorting_extractor_full_list}

writable_sorting_extractor_list = [sx for sx in installed_sorting_extractor_list if sx.is_writable]
writable_sorting_extractor_dict = {sorting_class.extractor_name: sorting_class
                                   for sorting_class in writable_sorting_extractor_list}
