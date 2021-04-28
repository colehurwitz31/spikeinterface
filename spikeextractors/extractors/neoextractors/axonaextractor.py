from .neobaseextractor import NeoBaseRecordingExtractor

try:
    import neo

    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'AxonaRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, **kargs):
        NeoBaseRecordingExtractor.__init__(self, **kargs)

        # Read channel groups by tetrode IDs
        self._key_properties['group'] = self.neo_reader.\
            raw_annotations['blocks'][0]['segments'][0]['signals']\
                           [0]['__array_annotations__']['tetrode_id']

        # Enforce 0-based indexing in spikeextractors
        self._key_properties['group'] = [el-1 for el in self._key_properties['group']]

        header_channels = self.neo_reader.header['signal_channels'][slice(None)]

        gains = header_channels['gain'] * self.additional_gain[0]
        self.set_channel_gains(gains=gains, channel_ids=self._channel_ids)

        names = header_channels['name']
        for i, ind in enumerate(self._channel_ids):
            self.set_channel_property(channel_id=ind, property_name='name', value=names[i])
