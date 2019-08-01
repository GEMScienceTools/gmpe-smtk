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
import pickle
import subprocess
from smtk.sm_database import GroundMotionDatabase


def git_suffix(fname):
    """
    :returns: `<short git hash>` if Git repository found
    """
    # we assume that the .git folder is one levels above
    git_path = os.path.join(os.path.dirname(fname), '..', '.git')

    # macOS complains if we try to execute git and it's not available.
    # Code will run, but a pop-up offering to install bloatware (Xcode)
    # is raised. This is annoying in end-users installations, so we check
    # if .git exists before trying to execute the git executable
    if os.path.isdir(git_path):
        try:
            gh = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=open(os.devnull, 'w'),
                cwd=os.path.dirname(git_path)).strip()
            gh = "-git" + gh.decode() if gh else ''
            return gh
        except Exception:
            # trapping everything on purpose; git may not be installed or it
            # may not work properly
            pass

    return ''


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
        raise IOError(
            "Expected metadata file of supported type not found in %s"
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


# version is used by the setup.py
__version__ = '0.9.0'
__version__ += git_suffix(__file__)
#
