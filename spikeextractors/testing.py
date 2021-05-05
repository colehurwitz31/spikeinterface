import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import numpy as np

from .extraction_tools import load_extractor_from_pickle, load_extractor_from_dict, \
    load_extractor_from_json


def check_recordings_equal(RX1, RX2, return_scaled=True, force_dtype=None, check_times=True):
    N = RX1.get_num_frames()
    # get_channel_ids
    assert np.allclose(RX1.get_channel_ids(), RX2.get_channel_ids())
    # get_num_channels
    assert np.allclose(RX1.get_num_channels(), RX2.get_num_channels())
    # get_num_frames
    assert np.allclose(RX1.get_num_frames(), RX2.get_num_frames())
    # get_sampling_frequency
    assert np.allclose(RX1.get_sampling_frequency(), RX2.get_sampling_frequency())
    # get_traces
    if force_dtype is None:
        assert np.allclose(RX1.get_traces(return_scaled=return_scaled), RX2.get_traces(return_scaled=return_scaled))
    else:
        assert np.allclose(RX1.get_traces(return_scaled=return_scaled).astype(force_dtype),
                           RX2.get_traces(return_scaled=return_scaled).astype(force_dtype))
    sf = 0
    ef = N
    ch = [RX1.get_channel_ids()[0], RX1.get_channel_ids()[-1]]
    if force_dtype is None:
        assert np.allclose(RX1.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled),
                           RX2.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef, return_scaled=return_scaled))
    else:
        assert np.allclose(RX1.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef,
                                          return_scaled=return_scaled).astype(force_dtype),
                           RX2.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef,
                                          return_scaled=return_scaled).astype(force_dtype))
    if check_times:
        for f in range(0, RX1.get_num_frames(), 10):
            assert np.isclose(RX1.frame_to_time(f), RX2.frame_to_time(f))
            assert np.isclose(RX1.time_to_frame(RX1.frame_to_time(f)), RX2.time_to_frame(RX2.frame_to_time(f)))
    # get_snippets
    frames = [30, 50, 80]
    snippets1 = RX1.get_snippets(reference_frames=frames, snippet_len=20, return_scaled=return_scaled)
    snippets2 = RX2.get_snippets(reference_frames=frames, snippet_len=(10, 10), return_scaled=return_scaled)
    if force_dtype is None:
        for ii in range(len(frames)):
            assert np.allclose(snippets1[ii], snippets2[ii])
    else:
        for ii in range(len(frames)):
            assert np.allclose(snippets1[ii].astype(force_dtype), snippets2[ii].astype(force_dtype))


def check_recording_properties(RX1, RX2):
    # check properties
    assert sorted(RX1.get_shared_channel_property_names()) == sorted(RX2.get_shared_channel_property_names())
    for prop in RX1.get_shared_channel_property_names():
        for ch in RX1.get_channel_ids():
            if not isinstance(RX1.get_channel_property(ch, prop), str):
                assert np.allclose(np.array(RX1.get_channel_property(ch, prop)),
                                   np.array(RX2.get_channel_property(ch, prop)))
            else:
                assert RX1.get_channel_property(ch, prop) == RX2.get_channel_property(ch, prop)


def check_recording_return_types(RX):
    channel_ids = RX.get_channel_ids()
    assert isinstance(RX.get_num_channels(), (int, np.integer))
    assert isinstance(RX.get_num_frames(), (int, np.integer))
    assert isinstance(RX.get_sampling_frequency(), (float, np.float))
    assert isinstance(RX.get_traces(start_frame=0, end_frame=10), (np.ndarray, np.memmap))

    for channel_id in channel_ids:
        assert isinstance(channel_id, (int, np.integer))


def check_sorting_return_types(SX):
    unit_ids = SX.get_unit_ids()
    assert (all(isinstance(id, int) or isinstance(id, np.integer) for id in unit_ids))
    for id in unit_ids:
        train = SX.get_unit_spike_train(id)
        # print(train)
        assert (all(isinstance(x, int) or isinstance(x, np.integer) for x in train))


def check_sortings_equal(SX1, SX2):
    # get_unit_ids
    ids1 = np.sort(np.array(SX1.get_unit_ids()))
    ids2 = np.sort(np.array(SX2.get_unit_ids()))
    assert (np.allclose(ids1, ids2))
    for id in ids1:
        train1 = np.sort(SX1.get_unit_spike_train(id))
        train2 = np.sort(SX2.get_unit_spike_train(id))
        assert np.array_equal(train1, train2)


def check_sorting_properties_features(SX1, SX2):
    # check properties
    print(SX1.__class__)
    print('Properties', sorted(SX1.get_shared_unit_property_names()), sorted(SX2.get_shared_unit_property_names()))
    assert sorted(SX1.get_shared_unit_property_names()) == sorted(SX2.get_shared_unit_property_names())
    for prop in SX1.get_shared_unit_property_names():
        for u in SX1.get_unit_ids():
            if not isinstance(SX1.get_unit_property(u, prop), str):
                assert np.allclose(np.array(SX1.get_unit_property(u, prop)),
                                   np.array(SX2.get_unit_property(u, prop)))
            else:
                assert SX1.get_unit_property(u, prop) == SX2.get_unit_property(u, prop)
    # check features
    print('Features', sorted(SX1.get_shared_unit_spike_feature_names()), sorted(SX2.get_shared_unit_spike_feature_names()))
    assert sorted(SX1.get_shared_unit_spike_feature_names()) == sorted(SX2.get_shared_unit_spike_feature_names())
    for feat in SX1.get_shared_unit_spike_feature_names():
        for u in SX1.get_unit_ids():
            assert np.allclose(np.array(SX1.get_unit_spike_features(u, feat)),
                               np.array(SX2.get_unit_spike_features(u, feat)))


def check_dumping(extractor):
    # dump to dict
    d = extractor.dump_to_dict()
    extractor_loaded = load_extractor_from_dict(d)

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)

    # dump to dict with relative path
    d = extractor.dump_to_dict(relative_to=".")
    extractor_loaded = load_extractor_from_dict(d)

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)

    # dump to json
    # without file_name
    extractor.dump_to_json()

    if 'Recording' in str(type(extractor)):
        extractor_loaded = load_extractor_from_json('spikeinterface_recording.json')
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        extractor_loaded = load_extractor_from_json('spikeinterface_sorting.json')
        check_sortings_equal(extractor, extractor_loaded)

    # with file_name
    extractor.dump_to_json(file_path='test_dumping/test.json')
    extractor_loaded = load_extractor_from_json('test_dumping/test.json')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)

    # dump to pickle
    # without file_name
    extractor.dump_to_pickle()

    if 'Recording' in str(type(extractor)):
        extractor_loaded = load_extractor_from_pickle('spikeinterface_recording.pkl')
        check_recordings_equal(extractor, extractor_loaded)
        check_recording_properties(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        extractor_loaded = load_extractor_from_pickle('spikeinterface_sorting.pkl')
        check_sortings_equal(extractor, extractor_loaded)
        check_sorting_properties_features(extractor, extractor_loaded)

    # dump to json with relative path
    extractor.dump_to_json(file_path='test_dumping/test_rel.json', relative_to=".")
    extractor_loaded = load_extractor_from_json('test_dumping/test_rel.json')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)

    # with file_name
    extractor.dump_to_pickle(file_path='test_dumping/test.pkl')
    extractor_loaded = load_extractor_from_pickle('test_dumping/test.pkl')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
        check_recording_properties(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)
        check_sorting_properties_features(extractor, extractor_loaded)

    # dump to pickle with relative path
    extractor.dump_to_pickle(file_path='test_dumping/test_rel.pkl', relative_to=".")
    extractor_loaded = load_extractor_from_pickle('test_dumping/test_rel.pkl')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)

    shutil.rmtree('test_dumping')
    if Path('spikeinterface_recording.json').is_file():
        os.remove('spikeinterface_recording.json')
    if Path('spikeinterface_sorting.json').is_file():
        os.remove('spikeinterface_sorting.json')
    if Path('spikeinterface_recording.pkl').is_file():
        os.remove('spikeinterface_recording.pkl')
    if Path('spikeinterface_sorting.pkl').is_file():
        os.remove('spikeinterface_sorting.pkl')


def get_default_nwbfile_metadata():
    """
    Returns structure with defaulted metadata values required for a NWBFile.
    """
    metadata = dict(
        NWBFile=dict(
            session_description="no description",
            session_start_time=datetime(1970, 1, 1),
            identifier=str(uuid.uuid4())
        ),
        Ecephys=dict(
            Device=[dict(
                name='Device_ecephys',
                description='no description'
            )],
            ElectrodeGroup=[],
            ElectricalSeries_raw=dict(
                name='raw_traces',
                description='those are the raw traces'
            ),
            ElectricalSeries_processed=dict(
                name='processed_traces',
                description='those are the processed traces'
            ),
            ElectricalSeries_lfp=dict(
                name='lfp_traces',
                description='those are the lfp traces'
            )
        )
    )
    return metadata