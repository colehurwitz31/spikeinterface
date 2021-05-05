import json
from pathlib import Path
import importlib
import numpy as np
import datetime
from copy import deepcopy
import tempfile
import pickle
import shutil

from .exceptions import NotDumpableExtractorError


class BaseExtractor:

    # To be specified in concrete sub-classes
    # The default filename (extension to be added by corresponding method)
    # to be used if no file path is provided
    _default_filename = None

    def __init__(self):
        self._kwargs = {}
        self._tmp_folder = None
        self._key_properties = {}
        self._properties = {}
        self._annotations = {}
        self._memmap_files = []
        self._features = {}
        self._epochs = {}
        self._times = None
        self.is_dumpable = True
        self.id = np.random.randint(low=0, high=9223372036854775807, dtype='int64')

    def __del__(self):
        # close memmap files (for Windows)
        for memmap_obj in self._memmap_files:
            self.del_memmap_file(memmap_obj)
        if self._tmp_folder is not None and len(self._memmap_files) > 0:
            try:
                shutil.rmtree(self._tmp_folder)
            except Exception as e:
                print('Impossible to delete temp file:', self._tmp_folder, 'Error', e)

    def del_memmap_file(self, memmap_file):
        """
        Safely deletes instantiated memmap file.

        Parameters
        ----------
        memmap_file: str or Path
            The memmap file to delete
        """
        if isinstance(memmap_file, np.memmap):
            memmap_file = memmap_file.filename
        else:
            memmap_file = Path(memmap_file)

        existing_memmap_files = [Path(memmap.filename) for memmap in self._memmap_files]
        if memmap_file in existing_memmap_files:
            try:
                memmap_idx = existing_memmap_files.index(memmap_file)
                memmap_obj = self._memmap_files[memmap_idx]
                if not memmap_obj._mmap.closed:
                    memmap_obj._mmap.close()
                    del memmap_obj
                memmap_file.unlink()
                del self._memmap_files[memmap_idx]
            except Exception as e:
                raise Exception(f"Error in deleting {memmap_file.name}: Error {e}")

    def make_serialized_dict(self, relative_to=None):
        """
        Makes a nested serialized dictionary out of the extractor. The dictionary be used to re-initialize an
        extractor with spikeextractors.load_extractor_from_dict(dump_dict)

        Parameters
        ----------
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path

        Returns
        -------
        dump_dict: dict
            Serialized dictionary
        """
        class_name = str(type(self)).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        try:
            version = imported_module.__version__
        except AttributeError:
            version = 'unknown'

        if self.is_dumpable:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': self._kwargs,
                         'key_properties': self._key_properties, 'annotations': self._annotations,
                         'version': version, 'dumpable': True}
        else:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': {}, 'key_properties': self._key_properties,
                         'annotations': self._annotations, 'version': imported_module.__version__,
                         'dumpable': False}

        if relative_to is not None:
            relative_to = Path(relative_to).absolute()
            assert relative_to.is_dir(), "'relative_to' must be an existing directory"

            dump_dict = _make_paths_relative(dump_dict, relative_to)

        return dump_dict

    def dump_to_dict(self, relative_to=None):
        """
        Dumps recording to a dictionary.
        The dictionary be used to re-initialize an
        extractor with spikeextractors.load_extractor_from_dict(dump_dict)

        Parameters
        ----------
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path

        Returns
        -------
        dump_dict: dict
            Serialized dictionary
        """
        return self.make_serialized_dict(relative_to)

    def _get_file_path(self, file_path, extensions):
        """
        Helper to be used by various dump_to_file utilities.

        Returns default file_path (if not specified), assures that target
        directory exists, adds correct file extension if none, and assures
        that provided file extension is one of the allowed.

        Parameters
        ----------
        file_path: str or None
        extensions: list or tuple
            First provided is used as an extension for the default file_path.
            All are tested against

        Returns
        -------
        Path
            Path object with file path to the file

        Raises
        ------
        NotDumpableExtractorError
        """
        ext = extensions[0]
        if self.check_if_dumpable():
            if file_path is None:
                file_path = self._default_filename + ext
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            folder_path = file_path.parent
            if Path(file_path).suffix == '':
                file_path = folder_path / (str(file_path) + ext)
            assert file_path.suffix in extensions, \
                "'file_path' should have one of the following extensions:" \
                " %s" % (', '.join(extensions))
            return file_path
        else:
            raise NotDumpableExtractorError(
                f"The extractor is not dumpable to {ext}")

    def dump_to_json(self, file_path=None, relative_to=None):
        """
        Dumps recording extractor to json file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path

        """
        dump_dict = self.make_serialized_dict(relative_to)
        self._get_file_path(file_path, ['.json'])\
            .write_text(
                json.dumps(_check_json(dump_dict), indent=4),
                encoding='utf8'
            )

    def dump_to_pickle(self, file_path=None, include_properties=True, include_features=True,
                       relative_to=None):
        """
        Dumps recording extractor to a pickle file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        file_path: str
            Path of the json file
        include_properties: bool
            If True, all properties are dumped
        include_features: bool
            If True, all features are dumped
        relative_to: str, Path, or None
            If not None, file_paths are serialized relative to this path
        """
        file_path = self._get_file_path(file_path, ['.pkl', '.pickle'])

        # Dump all
        dump_dict = {'serialized_dict': self.make_serialized_dict(relative_to)}
        if include_properties:
            if len(self._properties.keys()) > 0:
                dump_dict['properties'] = self._properties
        if include_features:
            if len(self._features.keys()) > 0:
                dump_dict['features'] = self._features
        # include times
        dump_dict["times"] = self._times

        file_path.write_bytes(pickle.dumps(dump_dict))

    def get_tmp_folder(self):
        """
        Returns temporary folder associated to the extractor

        Returns
        -------
        temp_folder: Path
            The temporary folder
        """
        if self._tmp_folder is None:
            self._tmp_folder = Path(tempfile.mkdtemp())
        return self._tmp_folder

    def set_tmp_folder(self, folder):
        """
        Sets temporary folder of the extractor

        Parameters
        ----------
        folder: str or Path
            The temporary folder
        """
        self._tmp_folder = Path(folder)

    def allocate_array(self, memmap, shape=None, dtype=None, name=None, array=None):
        """
        Allocates a memory or memmap array

        Parameters
        ----------
        memmap: bool
            If True, a memmap array is created in the sorting temporary folder
        shape: tuple
            Shape of the array. If None array must be given
        dtype: dtype
            Dtype of the array. If None array must be given
        name: str or None
            Name (root) of the file (if memmap is True). If None, a random name is generated
        array: np.array
            If array is given, shape and dtype are initialized based on the array. If memmap is True, the array is then
            deleted to clear memory

        Returns
        -------
        arr: np.array or np.memmap
            The allocated memory or memmap array
        """
        if memmap:
            tmp_folder = self.get_tmp_folder()
            if array is not None:
                shape = array.shape
                dtype = array.dtype
            else:
                assert shape is not None and dtype is not None, "Pass 'shape' and 'dtype' arguments"
            if name is None:
                tmp_file = tempfile.NamedTemporaryFile(suffix=".raw", dir=tmp_folder).name
            else:
                if Path(name).suffix == '':
                    tmp_file = tmp_folder / (name + '.raw')
                else:
                    tmp_file = tmp_folder / name
            raw_tmp_file = r'{}'.format(str(tmp_file))

            # make sure any open memmap files with same path are deleted
            self.del_memmap_file(raw_tmp_file)
            arr = np.memmap(raw_tmp_file, mode='w+', shape=shape, dtype=dtype)
            if array is not None:
                arr[:] = array
                del array
            else:
                arr[:] = 0
            self._memmap_files.append(arr)
        else:
            if array is not None:
                arr = array
            else:
                arr = np.zeros(shape, dtype=dtype)
        return arr

    def annotate(self, annotation_key, value, overwrite=False):
        """This function adds an entry to the annotations dictionary.

        Parameters
        ----------
        annotation_key: str
            An annotation stored by the Extractor
        value:
            The data associated with the given property name. Could be many
            formats as specified by the user
        overwrite: bool
            If True and the annotation already exists, it is overwritten
        """
        if annotation_key not in self._annotations.keys():
            self._annotations[annotation_key] = value
        else:
            if overwrite:
                self._annotations[annotation_key] = value
            else:
                print(f"{annotation_key} is already an annotation key. Use 'overwrite=True' to overwrite it")

    def get_annotation(self, annotation_name):
        """This function returns the data stored under the annotation name.

        Parameters
        ----------
        annotation_name: str
            A property stored by the Extractor

        Returns
        ----------
        annotation_data
            The data associated with the given property name. Could be many
            formats as specified by the user
        """
        if annotation_name not in self._annotations.keys():
            print(f"{annotation_name} is not an annotation")
            return None
        else:
            return deepcopy(self._annotations[annotation_name])

    def get_annotation_keys(self):
        """This function returns a list of stored annotation keys

        Returns
        ----------
        property_names: list
            List of stored annotation keys
        """
        return list(self._annotations.keys())

    def copy_annotations(self, extractor):
        """Copy object properties from another extractor to the current extractor.

        Parameters
        ----------
        extractor: Extractor
            The extractor from which the annotations will be copied
        """
        self._annotations = deepcopy(extractor._annotations)

    def add_epoch(self, epoch_name, start_frame, end_frame):
        """This function adds an epoch to your extractor that tracks
        a certain time period. It is stored in an internal
        dictionary of start and end frame tuples.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be added
        start_frame: int
            The start frame of the epoch to be added (inclusive)
        end_frame: int
            The end frame of the epoch to be added (exclusive). If set to None, it will include the entire
            sorting after the start_frame
        """
        if isinstance(epoch_name, str):
            start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
            self._epochs[epoch_name] = {'start_frame': start_frame, 'end_frame': end_frame}
        else:
            raise TypeError("epoch_name must be a string")

    def remove_epoch(self, epoch_name):
        """This function removes an epoch from your extractor.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be removed
        """
        if isinstance(epoch_name, str):
            if epoch_name in list(self._epochs.keys()):
                del self._epochs[epoch_name]
            else:
                raise ValueError("This epoch has not been added")
        else:
            raise ValueError("epoch_name must be a string")

    def get_epoch_names(self):
        """This function returns a list of all the epoch names in the extractor

        Returns
        ----------
        epoch_names: list
            List of epoch names in the recording extractor
        """
        epoch_names = list(self._epochs.keys())
        if not epoch_names:
            pass
        else:
            epoch_start_frames = []
            for epoch_name in epoch_names:
                epoch_info = self.get_epoch_info(epoch_name)
                start_frame = epoch_info['start_frame']
                epoch_start_frames.append(start_frame)
            epoch_names = [epoch_name for _, epoch_name in sorted(zip(epoch_start_frames, epoch_names))]
        return epoch_names

    def get_epoch_info(self, epoch_name):
        """This function returns the start frame and end frame of the epoch
        in a dict.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        ----------
        epoch_info: dict
            A dict containing the start frame and end frame of the epoch
        """
        # Default (Can add more information into each epoch in subclass)
        if isinstance(epoch_name, str):
            if epoch_name in list(self._epochs.keys()):
                epoch_info = self._epochs[epoch_name]
                return epoch_info
            else:
                raise ValueError("This epoch has not been added")
        else:
            raise ValueError("epoch_name must be a string")

    def copy_epochs(self, extractor):
        """Copy epochs from another extractor.

        Parameters
        ----------
        extractor: BaseExtractor
            The extractor from which the epochs will be copied
        """
        for epoch_name in extractor.get_epoch_names():
            epoch_info = extractor.get_epoch_info(epoch_name)
            self.add_epoch(epoch_name, epoch_info["start_frame"], epoch_info["end_frame"])

    def _cast_start_end_frame(self, start_frame, end_frame):
        from .extraction_tools import cast_start_end_frame
        return cast_start_end_frame(start_frame, end_frame)

    @staticmethod
    def load_extractor_from_json(json_file):
        """
        Instantiates extractor from json file

        Parameters
        ----------
        json_file: str or Path
            Path to json file

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        """
        json_file = Path(json_file)
        with open(str(json_file), 'r') as f:
            d = json.load(f)
            extractor = _load_extractor_from_dict(d)
        return extractor

    @staticmethod
    def load_extractor_from_pickle(pkl_file):
        """
        Instantiates extractor from pickle file.

        Parameters
        ----------
        pkl_file: str or Path
            Path to pickle file

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        """
        pkl_file = Path(pkl_file)
        with open(str(pkl_file), 'rb') as f:
            d = pickle.load(f)
        extractor = _load_extractor_from_dict(d['serialized_dict'])
        if 'properties' in d.keys():
            extractor._properties = d['properties']
        if 'features' in d.keys():
            extractor._features = d['features']
        if 'times' in d.keys():
            extractor._times = d['times']
        return extractor

    @staticmethod
    def load_extractor_from_dict(d):
        """
        Instantiates extractor from dictionary

        Parameters
        ----------
        d: dictionary
            Python dictionary

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        """
        extractor = _load_extractor_from_dict(d)
        return extractor

    def check_if_dumpable(self):
        return _check_if_dumpable(self.make_serialized_dict())


def _make_paths_relative(d, relative):
    dcopy = deepcopy(d)
    if "kwargs" in dcopy.keys():
        relative_kwargs = _make_paths_relative(dcopy["kwargs"], relative)
        dcopy["kwargs"] = relative_kwargs
        return dcopy
    else:
        for k in d.keys():
            # in SI, all input paths have the "path" keyword
            if "path" in k:
                d[k] = str(Path(d[k]).relative_to(relative))
        return d


def _load_extractor_from_dict(dic):
    cls = None
    class_name = None
    probe_file = None
    kwargs = deepcopy(dic['kwargs'])
    if np.any([isinstance(v, dict) for v in kwargs.values()]):
        # nested
        for k in kwargs.keys():
            if isinstance(kwargs[k], dict):
                if 'module' in kwargs[k].keys() and 'class' in kwargs[k].keys() and 'version' in kwargs[k].keys():
                    extractor = _load_extractor_from_dict(kwargs[k])
                    class_name = dic['class']
                    cls = _get_class_from_string(class_name)
                    kwargs[k] = extractor
                    break
    elif np.any([isinstance(v, list) and isinstance(v[0], dict) for v in kwargs.values()]):
        # multi
        for k in kwargs.keys():
            if isinstance(kwargs[k], list) and isinstance(kwargs[k][0], dict):
                extractors = []
                for kw in kwargs[k]:
                    if 'module' in kw.keys() and 'class' in kw.keys() and 'version' in kw.keys():
                        extr = _load_extractor_from_dict(kw)
                        extractors.append(extr)
                class_name = dic['class']
                cls = _get_class_from_string(class_name)
                kwargs[k] = extractors
                break
    else:
        class_name = dic['class']
        cls = _get_class_from_string(class_name)

    assert cls is not None and class_name is not None, "Could not load spikeinterface class"
    if not _check_same_version(class_name, dic['version']):
        print('Versions are not the same. This might lead to errors. Use ', class_name.split('.')[0],
              'version', dic['version'])

    if 'probe_file' in kwargs.keys():
        probe_file = kwargs.pop('probe_file')

    # instantiate extrator object
    extractor = cls(**kwargs)

    # load probe file
    if probe_file is not None:
        assert 'Recording' in class_name, "Only recording extractors can have probe files"
        extractor = extractor.load_probe_file(probe_file=probe_file)

    # load properties and features
    if 'key_properties' in dic.keys():
        extractor._key_properties = dic['key_properties']

    if 'annotations' in dic.keys():
        extractor._annotations = dic['annotations']

    return extractor


def _get_class_from_string(class_string):
    class_name = class_string.split('.')[-1]
    module = '.'.join(class_string.split('.')[:-1])
    imported_module = importlib.import_module(module)

    try:
        imported_class = getattr(imported_module, class_name)
    except:
        imported_class = None

    return imported_class


def _check_same_version(class_string, version):
    module = class_string.split('.')[0]
    imported_module = importlib.import_module(module)

    try:
        return imported_module.__version__ == version
    except AttributeError:
        return 'unknown'


def _check_if_dumpable(d):
    kwargs = d['kwargs']
    if np.any([isinstance(v, dict) and 'dumpable' in v.keys() for (k, v) in kwargs.items()]):
        for k, v in kwargs.items():
            if 'dumpable' in v.keys():
                return _check_if_dumpable(v)
    else:
        return d['dumpable']


def _check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _check_json(v)
        elif isinstance(v, Path):
            d[k] = str(v.absolute())
        elif isinstance(v, bool):
            d[k] = bool(v)
        elif isinstance(v, (np.int, np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v, (np.float, np.float32, np.float64)):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.ndarray, list)):
            if len(v) > 0:
                if isinstance(v[0], dict):
                    # these must be extractors for multi extractors
                    d[k] = [_check_json(v_el) for v_el in v]
                else:
                    v_arr = np.array(v)
                    if len(v_arr.shape) == 1:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [int(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [float(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif isinstance(v_arr[0], str):
                            v_arr = [str(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 1D arrays of int, float, or str types can be serialized')
                    elif len(v_arr.shape) == 2:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [[int(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [[float(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        elif 'bool' in str(v_arr.dtype):
                            v_arr = [[bool(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 2D arrays of int or float type can be serialized')
                    else:
                        print(f"Skipping field {k}: only 1D and 2D arrays can be serialized")
            else:
                d[k] = list(v)
    return d
