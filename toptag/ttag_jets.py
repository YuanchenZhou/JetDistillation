r"""## Top tagging jets 

To be documented...

"""
# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2022 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from energyflow.utils.data_utils import _get_filepath, _pad_events_axis1

__all__ = ['load']

NUM_PER_FILE = 100000
MAX_NUM_FILES = 10
URLS = {
    'pythia': {
        'dropbox': [
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_0.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_1.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_2.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_3.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_4.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_5.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_6.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_7.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_8.npz?dl=1',
            'https://www.dropbox.com/s/fclsl7pukcpobsb/top_qcd_9.npz?dl=1'
        ],
    },

    'herwig': {
        'dropbox': [
            'https://www.dropbox.com/s/xizexr2tjq2bm59/herwig/top_qcd_0.npz?dl=1',
            'https://www.dropbox.com/s/ym675q2ui3ik3n9/herwig/top_qcd_1.npz?dl=1',
            'https://www.dropbox.com/s/qic6ejl27y6vpqj/herwig/top_qcd_2.npz?dl=1',
            'https://www.dropbox.com/s/ea5a9wruo7sf3zy/herwig/top_qcd_3.npz?dl=1',
            'https://www.dropbox.com/s/5iz5q2pjcys74tb/herwig/top_qcd_4.npz?dl=1',
            'https://www.dropbox.com/s/6zha7fka0dl7t30/herwig/top_qcd_5.npz?dl=1',
            'https://www.dropbox.com/s/vljp5nhoocv2zmf/herwig/top_qcd_6.npz?dl=1',
            'https://www.dropbox.com/s/vzzl5yv9esro811/herwig/top_qcd_7.npz?dl=1',
            'https://www.dropbox.com/s/74u8y4afe1jqiyw/herwig/top_qcd_8.npz?dl=1',
            'https://www.dropbox.com/s/ra7hdq23qy7lgia/herwig/top_qcd_9.npz?dl=1',
        ]
    }
}

# Hashes NOT UPDATED YET, these are for the q/g files -- MLB
HASHES = {
    'pythia': {
        'sha256': [
            '09f7b16fa7edb312c0f652bb8504de45f082c4193df65204d693155017272fe9',
            '7dc9a50bb38e9f6fc1f11db18f9bd04f72823c944851746b848dee0bba808537',
            '3e6217aad8e0502f5ce3b6371c61396dfc48a6cf4f26ee377cc7b991b1d2b543',
            'b5b7d742b2599bcbe1d7a639895bca64c28da513dc3620b0e5bbb5801f8c88fd',
            '7d31bc48c15983401e0dbe8fd5ee938c3809d9ee3c909f4adab6daf8b73c14f1',
            'cec0d7b2afa9d955543c597f9b7f3b3767812a68b2401ec870caf3a2ceb98401',
            'e984620f57abe06fc5d0b063f9f84ba54bd3e8c295d2b2419a7b1c6175079ed4',
            '6e3b69196995d6eb3b8e7af874e2b9f93d904624f7a7a73b8ff39f151e3bd189',
        ],
        'md5': [
            '2628367c57ba598f4473c870d1381041',
            'dd3ad998b0a1bd9acea2ecf029a8a921',
            'a56d6bb98361b55382aa8c06225e05d8',
            '266c688e9e6ff1cd20840692d45eaaf8',
            '95a9f7e555fb7b1073967056b9030b11',
            '4ae72aaabe121bd489532c99a6bdde95',
            'a2b80bd4199468fde4f302d346a8c9d8',
            '1157cbace488c70c9dcfc250f3345b06',
            '4b424b553e1e7f852e47ea9904bc2dcf',
        ],
    },
    
    'herwig': {
        'sha256': [
            '0527349778c0ab2f7da268975fb9e7c0705c88d60f2c478401d941b9913f4d44',
            '65ef3b4cced4e2618c2bf8f3c66ef707dbd7a9740825f93549732d64e60d7ea8',
            'f13dab1937e40d0c05b97e9813fb4dda5156a8f6b4e41a89cc13821d02a60f58',
            '7b55e26262f2c156b15014b796d0a7e7a5254a982170f45cf2d9857b1f23b5f7',
            '3a5006da4a05192636a74fc818256fce215970c626719738cae9f82e3f068646',
            '2601564aee41aa5851392d6b3d12021f978fa17199e42dde004e35d1175055ea',
            '2c1fc34e99816a0bb5a84f68fa42f6314252521f6b32384a118cdec872ea97a1',
            '4b05f17acb046ad50232987003b89a91800cc713eefd81142ffeb42259369fb2',
            '150cbe132a2ee3178ba3a93a6b1733b3498b728db91f572295b6213d287ec1f7',
            '7d74c90843c751ade4cac47f6c2505da8bcbaf8645bc3f9870bdca481ff805fd',
        ],
        'md5': [
            'a9de310c35c5a83ea592ef93070ff2f3',
            'd6ff8cc5c6192309fba915114fdc8358',
            '625bc4a0619b5b2551e273be493c6092',
            '821b293d4e68db8b2bd40a1732d1d865',
            '415dded70fca2ae5e555cdee776724d8',
            '242e23df1b837b9afac880383157a161',
            '068eb955146f773c1b5815dd3424c434',
            'e0212b768f57344ae60df7783ba5ba25',
            'd1a082794c84d2b0cc159034cf4d44b6',
            'f2e1c99033a2ff7d97d9968d394333ba',
            '5eab363df8bdff106f53858e60fe7ed1',
        ],
    }
}

GENERATORS = frozenset(URLS.keys())
SOURCES = ['dropbox']

# load(num_data=100000, pad=True, ncol=4, generator='pythia',
#      with_bc=False, cache_dir='~/.energyflow')
def load(num_data=100000, pad=True, ncol=4, 
         generator='pythia', with_bc=False, cache_dir='~/.energyflow/'):
    """Loads samples from the dataset.
    Any file that is needed that has not been cached will be 
    automatically downloaded. Downloading a file causes it to be cached for
    later use. Basic checksums are performed.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all
        events.
    - **pad** : _bool_
        - Whether to pad the events with zeros to make them the same length.
        Note that if set to `False`, the returned `X` array will be an object
        array and not a 3-d array of floats.
    - **ncol** : _int_
        - Number of columns to keep in each event.
    - **generator** : _str_
        - Specifies which Monte Carlo generator the events should come from.
        Currently, the options are `'pythia'` and `'herwig'`.
    - **with_bc** : _bool_
        - Whether to include jets coming from bottom or charm quarks. Changing
        this flag does not mask out these jets but rather accesses an entirely
        different dataset. The datasets with and without b and c quarks should
        not be combined.
    - **cache_dir** : _str_
        - The directory where to store/look for the files. Note that 
        `'datasets'` is automatically appended to the end of this path.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above. If
        `pad` is `False` then these will be object arrays holding the events,
        each of which is a 2-d ndarray.
    """

    # check for valid options
    if generator not in GENERATORS:
        raise ValueError("'generator' must be in " + str(GENERATORS))

    # get number of files we need
    num_files = int(np.ceil(num_data/NUM_PER_FILE)) if num_data > -1 else MAX_NUM_FILES
    if num_files > MAX_NUM_FILES:
        warnings.warn('More data requested than available. Providing the full dataset.')
        num_files = MAX_NUM_FILES
        num_data = -1

    # index into global variables
    urls = URLS[generator]
    hashes = HASHES[generator]

    # obtain files
    Xs, ys = [], []
    for i in range(num_files):
        for j,source in enumerate(SOURCES):
            try:
                url = urls[source][i]
                filename = url.split('/')[-1].split('?')[0]

                #fpath = _get_filepath(filename, url, cache_dir, file_hash=hashes['sha256'][i])
                fpath = _get_filepath(filename, url, cache_dir)
                #print(filename, url, cache_dir, source)
                
                # we succeeded, so don't continue trying to download this file
                break

            except Exception as e:
                print(str(e))

                # if this was our last source, raise an error
                if j == len(SOURCES) - 1:
                    m = 'Failed to download {} from any source.'.format(filename)
                    #raise RuntimeError(m)

                # otherwise indicate we're trying again
                else:
                    print("Failed to download {} from source '{}', trying next source...".format(filename, source))

        # load file and append arrays
        with np.load(fpath) as f:
            #Xs.append(f['X'])
            #ys.append(f['y'])
            Xs.append(f['data'])
            ys.append(f['labels'])

    # get X array
    if pad:
        max_len_axis1 = max([X.shape[1] for X in Xs])
        X = np.vstack([_pad_events_axis1(x[...,:ncol], max_len_axis1) for x in Xs])
    else:
        X = np.asarray([x[x[:,0]>0,:ncol] for X in Xs for x in X], dtype='O')

    # get y array
    y = np.concatenate(ys)

    # chop down to specified amount of data
    if num_data > -1:
        X, y = X[:num_data], y[:num_data]

    return X, y
    
# Put this in virtual_environment/lib/python3.9/site-packages/energyflow/datasets/
