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
Strong motion record selection tools
"""
import numpy as np
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from smtk.sm_database import GroundMotionRecord, GroundMotionDatabase

def rank_sites_by_record_count(database, threshold=0):
    """
    Function to determine count the number of records per site and return
    the list ranked in descending order
    """
    name_id_list = [(rec.site.id, rec.site.name) for rec in database.records]
    name_id = dict([])
    for name_id_pair in name_id_list:
        if name_id_pair[0] in name_id:
            name_id[name_id_pair[0]]["Count"] += 1
        else:
            name_id[name_id_pair[0]] = {"Count": 1, "Name": name_id_pair[1]}
    counts = np.array([name_id[key]["Count"] for key in name_id])
    sort_id = np.flipud(np.argsort(counts))

    key_vals = list(name_id)
    output_list = []
    for idx in sort_id:
        if name_id[key_vals[idx]]["Count"] >= threshold:
            output_list.append((key_vals[idx], name_id[key_vals[idx]]))
    return OrderedDict(output_list)


class SMRecordSelector(object):
    """
    General class to hold methods for selecting and querying a strong
    motion database
    """
    def __init__(self, database):
        """
        """
        self.database = database
        self.record_ids = self._get_record_ids()
        self.event_ids = self.database._get_event_id_list()
        self.event_ids = self.event_ids.tolist()
        self.site_ids = [record.site.id for record in self.database.records]

    def _get_record_ids(self):
        """
        Returns a list of record IDs
        """
        return [record.id for record in self.database.records]
    
    def select_records(self, idx, as_db=False):
        """
        Selects records from a list of pointers:
        :param list idx:
            List of pointers to selected records
        :param bool as_db:
            Returns the selection as an instance of :class:
            GroundMotionDatabase (True) or as a list of selected records
            (False)
        """
        selected_records = [self.database.records[iloc] for iloc in idx]
        if as_db:
            # Returns the selection as a new instance of the database class
            if len(selected_records) == 0:
                return None
            else:
                return GroundMotionDatabase(self.database.id,
                                            self.database.name,
                                            self.database.directory,
                                            selected_records)
        else:
            return selected_records 

    def select_from_record_id(self, record_id):
        """
        Selects a record according to its waveform id
        """
        if record_id in self.record_ids:
            return self.database.records[self.record_ids.index(record_id)]
        else:
            raise ValueError(
                "Record {:s} is not in database".format(record_id))

    def select_from_record_ids(self, record_ids, as_db=False):
        """
        Selects records from a list of IDs
        """
        idx = []
        for record_id in record_ids:
            if not record_id in self.record_ids:
                print("Record {:s} is not in database".format(record_id))
        for iloc, record in enumerate(self.database.records):
            if record.id in record_ids:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_from_site_id(self, site_id, as_db=False):
        """
        Select records corresponding to a particular site ID
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.site.id == site_id:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_from_site_ids(self, site_ids, as_db=False):
        """
        Selects records corresponding to a set of site IDs
        """
        for site_id in site_ids:
            if not site_id in self.site_ids:
                print("Site {:w is not in database" % record_id)
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.site.id in site_ids:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_from_event_id(self, event_id, as_db=False):
        """
        Returns a set of records from a common event
        """
        if not event_id in self.event_ids:
            raise ValueError("Event %s not found in database" % event_id)
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.event.id == event_id:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_from_event_ids(self, event_ids, as_db=False):
        """
        Returns records from events whose IDs are found in the list
        """
        for event_id in event_ids:
            if not event_id in self.event_ids:
                print("Event {:s} not found in database".format(event_id))
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.event.id in event_ids:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_within_time(self, start_time=None, end_time=None, as_db=False):
        """
        Selects records within a specific time
        :param start_time:
            Earliest time as instance of :class: datetime.datetime
        :param end_time:
            Latest time as instance of :class: datetime.datetime
        """
        if start_time:
            assert isinstance(start_time, datetime)
        else:
            start_time = datetime.min

        if end_time:
            assert isinstance(end_time, datetime)
        else:
            end_time = datetime.now()
        idx = []
        for iloc, record in enumerate(self.database.records):
            if (record.event.datetime >= start_time) and\
                (record.event.datetime <= end_time):
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_within_depths(self, upper_depth=None, lower_depth=None,
            as_db=False):
        """
        Selects records corresponding to events within a specific depth range
        :param float upper_depth:
            Upper event depth (km)    
        :param float lower_depth:
            Lower event depth (km)
        """
        if not upper_depth:
            upper_depth = 0.0
        if not lower_depth:
            lower_depth = np.inf
        assert (lower_depth >= upper_depth)
        idx = []
        for iloc, record in enumerate(self.database.records):
            if (record.event.depth >= upper_depth) and\
                (record.event.depth <= lower_depth):
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_within_magnitude(self, lower=None, upper=None, as_db=False):
        """
        Select records corresponding to events within a magnitude range
        :param float lower:
            Lower bound magnitude
        :param float upper:
            Upper bound magnitude
        """
        if not lower:
            lower = -np.inf
        if not upper:
            upper = np.inf
        assert (upper >= lower)
        idx = []
        for iloc, record in enumerate(self.database.records):
            if (record.event.magnitude.value >= lower) and\
                (record.event.magnitude.value <= upper):
                idx.append(iloc)
        return self.select_records(idx, as_db)

    # Site based selections
    def select_by_station_country(self, country, as_db=False):
        """
        Returns the records within a specific country
        :param str country:
            Country name
        """
        return self.select_by_site_attribute("country", country, as_db)

    def select_by_site_attribute(self, attribute, value, as_db=False):
        """
        Select records corresponding to a particular site attribute
        :param str attribute:
            Attribute name
        :param value:
            Value of the specific attribute
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if getattr(record.site, attribute) == value:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_within_vs30_range(self, lower_vs30, upper_vs30, as_db=False):
        """
        Select records within a given Vs30 range
        :param float lower_vs30:
            Lowest Vs30 (m/s)
        :param float uper_vs30:
            Upper Vs30 (m/s)
        """
        if not lower_vs30:
            lower_vs30 = -np.inf
        if not upper_vs30:
            upper_vs30 = np.inf
        idx = []
        for iloc, record in enumerate(self.database.records):
            if not record.site.vs30:
                continue
            if (record.site.vs30 >= lower_vs30) and\
                (record.site.vs30 <= upper_vs30):
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_stations_within_distance(self, location, distance, as_db=False):
        """
        Selects stations within a distance of a specified location
        :param location:
            Location as instance of :class: openquake.hazardlib.geo.point.Point
        :param float distance:
            Distance (kme
        """
        assert isinstance(location, Point)
        idx = []
        for iloc, record in enumerate(self.database.records):
            is_close = location.closer_than(
                Mesh(np.array([record.site.longitude]),
                     np.array([record.site.latitude]),
                     None),
                distance)
            if is_close[0]:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_stations_within_region(self, region, as_db=False):
        """
        Selects station inside of a specified region
        :param region:
             Region as instance of :class:
             openquake.hazardlib.geo.polygon.Polygon
        """
        assert isinstance(region, Polygon)
        idx = []
        
        for iloc, record in enumerate(self.database.records):
            site_loc = Mesh(np.array([record.site.longitude]),
                            np.array([record.site.latitude]),
                            None)
            if region.intersects(site_loc)[0]:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    # Distance based selection
    def select_within_distance_range(self, distance_type, shortest, furthest,
            alternative=False, as_db=False):
        """
        Select records based on a distance range
        :param str distance_type:
            Distance type
        :param float shortest:
            Shortest distance limit (default to 0.0 if None)
        :param float furthest:
            Furthest distance limit (default to inf if None)
        :param tuple alternative:
            If no value is found for a given distance metric it is possible
            to specify an alternative distance metric and corresponding limits
            as a tuple of (distance_type, shortest, furthest)
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            value = getattr(record.distance, distance_type)
            if value:
                if (value >= shortest) and (value <= furthest):
                    idx.append(iloc)
                else:
                    continue
            elif alternative and isinstance(alternative, tuple):
                alt_value = getattr(record.distance, alternative[0])
                if alt_value and (alt_value >= alternative[1]) and\
                    (alt_value <= alternative[2]):
                    idx.append(iloc)
#                else:
#                    raise ValueError("Record %s is missing selected distance "
#                        "metric and alternative metric" % record.id)
            else:
                print("Record {:s} is missing selected distance metric".format(
                      record.id))
        return self.select_records(idx, as_db)

    # Event-based selection
    def select_mechanism_type(self, mechanism_type, as_db=False):
        """
        Select records based on event mechanism type
        :param str mechanism_type:
            Mechanism type
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.event.mechanism.mechanism_type == mechanism_type:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_trt_type(self, trt_type, as_db=False):
        """
        Select records based on tectonic region type
        :param str trt_type:
            Tectonic Region
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.event.tectonic_region == trt_type:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_event_country(self, country, as_db=False):
        """
        Select records corresponding to events within a specific country
        :param str country
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.event.mechanism.country == country:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_epicentre_within_distance_from_point(self, location, distance,
            as_db=False):
        """
        Selects the records whose epicentres are a given distance from a
        point location
        """
        assert isinstance(location, Point)
        idx = []
        for iloc, record in enumerate(self.database.records):
            is_close = location.closer_than(
                Mesh(np.array([record.event.longitude]),
                     np.array([record.event.latitude]),
                     None),
                distance)
            if isclose[0]:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_epicentre_within_region(self, region, as_db=False):
        """
        Selects records from event inside the specified region
        """
        assert isinstance(region, Polygon)
        idx = []
        
        for iloc, record in enumerate(self.database.records):
            epi_loc = Mesh(np.array([record.event.longitude]),
                            np.array([record.event.latitude]),
                            None)
            if region.intersects(epi_loc)[0]:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_epicentre_within_bounding_box(self, llon, llat, ulon, ulat,
                                             as_db=False):
        """
        Selects earthquakes whose epicentres are within a longitude
        and latitude bounding box
        """
        assert ulon >= llon and ulat >= llat
        idx = []
        for iloc, record in enumerate(self.database.records):
            bbox_check = record.event.longitude >= llon and\
                record.event.longitude <= ulon and\
                record.event.latitude >= llat and\
                record.event.latitude <= ulat
            if bbox_check:
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_longest_usable_period(self, lup, as_db=False):
        """
        Selects records with a longest usable period > lup

        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if record.average_lup and (record.average_lup >= lup):
                idx.append(iloc)
        return self.select_records(idx, as_db)

    def select_backarc_forearc(self, forearc=True, as_db=False):
        """
        Select sites depending on whether they are backarc or forearc
        """
        idx = []
        for iloc, record in enumerate(self.database.records):
            if forearc:
                if not record.site.backarc:
                    idx.append(iloc)
            else:
                if record.site.backarc:
                    idx.append(iloc)
        if len(idx):
            return self.select_records(idx, as_db)
        else:
            if forearc:
                print("No forearc sites found in database")
            else:
                print("No backarc sites found in database")
            return None

    def select_event_within_fault_distance(self, faults, distance, as_db=True):
        """
        Selects records whose hypocentes lie within a given distance of a
        fault or set of faults

        :param list faults:
            Faults as a list of instances of 
            :class:`openquake.hazardlib.geo.SimpleFaultSurface`

        :param float distance:
            Distance (nearest distance to fault) (km) for selection
        """
        # Render record event locations to openquake.hazardlib.geo.Mesh
        lons, lats, depths = [], [], []
        for rec in self.database.records:
            lons.append(rec.event.longitude)
            lats.append(rec.event.latitude)
            depths.append(rec.event.depth)
        db_mesh = Mesh(np.array(lons), np.array(lats), np.array(depths))
        # Initially no event is allocated to any faults
        idx = np.zeros(len(self.database), dtype=int)
        for fault in faults:
            rrup = fault.get_min_distance(db_mesh)
            within_dist = rrup < distance
            # If within fault distance add 1 - no record of which fault is
            # kept
            idx[within_dist] += 1
        # Find the events within the selected distance of any fault
        idx = idx > 0
        if np.any(idx):
            idx = np.where(idx)[0]
            return self.select_records(idx, as_db)
        else:
            None

    def select_within_interface_distances(self, surfaces, sp_distance,
                                          sf_distance, crustal_thickness,
                                          as_db=True):
        """
        Selects records whose events are within a distance of a set of
        interfaces. A two-stage selection process is applied here. First
        the records within a distance of the surface projection of the
        interface are selected. From those then events within the proposed
        crustal thickness are discarded. In the second step those within
        the specified distance of the interface of are selected providing they
        are within the crust.

        :param list surfaces:
            Subduction interfaces as list of instances of SimpleFaultSurface
            or ComplexFautSurface
        :param float sp_distance:
            Distance from surface projection (km) for selection
        :param float sf_distance:
            Distance from interface (km) for selection
        :param float crustal_thickness:
            Thickness of crust
        """
        lons, lats, depths = [], [], []
        for rec in self.database.records:
            lons.append(rec.event.longitude)
            lats.append(rec.event.latitude)
            depths.append(rec.event.depth)
        db_mesh = Mesh(np.array(lons), np.array(lats), np.array(depths))
        idx = np.zeros(len(self.database), dtype=int)
        for surface in surfaces:
            # Get events falling within a distance of the surface projection
            # But below the crust
            rjb = surface.get_joyner_boore_distance(db_mesh)
            surf_idx = np.logical_and(rjb <= sp_distance,
                                      db_mesh.depths >= crustal_thickness)
            idx[surf_idx] += 1
            # Now find those events in the crust within a sf_distance of the fault
            rrup = surface.get_min_distance(db_mesh)
            sub_idx = np.logical_and(db_mesh.depths <= crustal_thickness,
                                     rrup <= sf_distance)
            idx[sub_idx] += 1
        idx = idx > 0
        if np.any(idx):
            idx = np.where(idx)[0]
            return self.select_records(idx, as_db)
        else:
            None
