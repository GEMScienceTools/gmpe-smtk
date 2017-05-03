#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
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

"""
Tool for creating visualisation of database information
"""
from sets import Set
import numpy as np
import matplotlib.pyplot as plt
from smtk.sm_utils import _save_image
from smtk.strong_motion_selector import SMRecordSelector 

DISTANCES = {"repi": lambda rec: rec.distance.repi,
             "rhypo": lambda rec: rec.distance.rhypo,
             "rjb": lambda rec: rec.distance.rjb,
             "rrup": lambda rec: rec.distance.rrup,
             "r_x": lambda rec: rec.distance.r_x,
             }

DISTANCE_LABEL = {"repi": "Epicentral Distance (km)",
                  "rhypo": "Hypocentral Distance (km)",
                  "rjb": "Joyner-Boore Distance (km)",
                  "rrup": "Rupture Distance (km)",
                  "r_x": "R-x Distance (km)"}

def get_magnitude_distances(db1, dist_type):
    """
    From the Strong Motion database, returns lists of magnitude and distance
    pairs
    """
    mags = []
    dists = []
    for record in db1.records:
        mags.append(record.event.magnitude.value)
        if dist_type == "rjb":
            rjb = DISTANCES[dist_type](record)
            if rjb:
                dists.append(rjb)
            else:
                dists.append(DISTANCES["repi"](record))
        elif dist_type == "rrup":
            rrup = DISTANCES[dist_type](record)
            if rrup:
                dists.append(rrup)
            else:
                dists.append(DISTANCES["rhypo"](record))
        else:
            dists.append(DISTANCES[dist_type](record))
    return mags, dists

def db_magnitude_distance(db1, dist_type, figure_size=(7, 5),
        figure_title=None,filename=None, filetype="png", dpi=300):
    """
    Creates a plot of magnitude verses distance for a strong motion database
    """
    plt.figure(figsize=figure_size)
    mags, dists = get_magnitude_distances(db1, dist_type)
    plt.semilogx(np.array(dists), np.array(mags), "o")
    plt.xlabel(DISTANCE_LABEL[dist_type], fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    if figure_title:
        plt.title(figure_title, fontsize=18)
    _save_image(filename, filetype, dpi)
    plt.grid()
    plt.show()

NEHRP_BOUNDS = {"A": (1500.0, np.inf),
                "B": (760.0, 1500.0),
                "C": (360.0, 760.),
                "D": (180., 360.),
                "E": (0., 180.)}

EC8_BOUNDS = {"A": (800., np.inf),
              "B": (360.0, 800.),
              "C": (180.0, 360.),
              "D": (0., 360)}

def _site_selection(db1, site_class, classifier):
    """
    Select records within a particular site class and/or vs30 range
    """
    idx = []
    for iloc, rec in enumerate(db1.records):
        if classifier == "NEHRP":
            if rec.site.nehrp and (rec.site.nerhp == site_class):
                idx.append(iloc)
                continue
              
            if rec.site.vs30:
                if (rec.site.vs30 >= NEHRP_BOUNDS[site_class][0]) and\
                    (rec.site.vs30 < NEHRP_BOUNDS[site_class][1]):
                    idx.append(iloc)
        elif classifier == "EC8":
            if rec.site.ec8 and (rec.site.ec8 == site_class):
                idx.append(iloc)
                continue
              
            if rec.site.vs30:
                if (rec.site.vs30 >= EC8_BOUNDS[site_class][0]) and\
                    (rec.site.vs30 < EC8_BOUNDS[site_class][1]):
                    idx.append(iloc)
        else:
            raise ValueError("Unrecognised Site Classifier!")
            
    return idx

def db_magnitude_distance_by_site(db1, dist_type, classification="NEHRP",
        figure_size=(7, 5), filename=None, filetype="png", dpi=300):
    """
    Plot magnitude-distance comparison by site NEHRP or Eurocode 8 Site class   
    """ 
    if classification == "NEHRP":
        site_bounds = NEHRP_BOUNDS
    elif classification == "EC8":
        site_bounds = EC8_BOUNDS
    else:
        raise ValueError("Unrecognised Site Classifier!")
    selector = SMRecordSelector(db1)
    plt.figure(figsize=figure_size)
    total_idx = []
    for site_class in site_bounds.keys():
        site_idx = _site_selection(db1, site_class, classification)
        site_db = selector.select_records(site_idx, as_db=True)
        mags, dists = get_magnitude_distances(site_db, dist_type)
        plt.plot(np.array(dists), np.array(mags), "o",
                 label="Site Class %s" % site_class)
        total_idx.extend(site_idx)
    unc_idx = Set(range(db1.number_records())).difference(Set(total_idx))
    unc_db = selector.select_records(unc_idx, as_db=True)
    mag, dists = get_magnitude_distances(site_db, dist_type)
    plt.semilogx(np.array(dists), np.array(mags), "o", mfc="None",
                 label="Unclassified", zorder=0)
    plt.xlabel(DISTANCE_LABEL[dist_type], fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.grid()
    plt.legend(ncol=2,loc="lower right", numpoints=1)
    plt.title("Magnitude vs Distance (by %s Site Class)" % classification,
              fontsize=18)
    _save_image(filename, filetype, dpi)
    plt.show()
