# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
import os
import sys
import json
from smtk.sm_database import GroundMotionDatabase
from openquake.hazardlib import __version__


if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

def load_database(directory):
    """
    Wrapper function to load the metadata of a strong motion database
    according to the filetype
    """
    metadata_file = None
    filetype = None
    fileset = os.listdir(directory)
    for ftype in ["pkl", "json"]:
        if ("metadatafile.%s" % ftype) in fileset:
            metadata_file = "metadatafile.%s" % ftype
            filetype = ftype
            break
    if not metadata_file:
        raise IOError("Expected metadata file of supported type not found in %s"
                      % directory)
    metadata_path = os.path.join(directory, metadata_file)
    if filetype == "json":
        # json metadata filetype
        return GroundMotionDatabase.from_json(metadata_path)
    elif filetype == "pkl":
        # pkl file type
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Metadata filetype %s not supported" % ftype)
