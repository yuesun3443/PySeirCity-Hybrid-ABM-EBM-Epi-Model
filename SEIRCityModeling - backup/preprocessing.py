from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

from FacilityTimeWorkflow import get_baseline_time_and_mobility_change, convert_dfs_to_dict
from StationaryDistributionWorkflow import create_tm_sd_for_all_homeblock_travelertype_pair
from FacilityPatronageWorkflow import (reoranize_agent_trip_records, 
                                       map_agent_to_traveler_and_traveler_to_agent, 
                                       create_facility_visit_urbano_agent_record_parallel_computing, 
                                       create_facility_visit_urbano_agent_record)

class SimulationPeriodBasicInfo:
    """
    Contain info that does not change as time passes. 
    Such as facility visited record, traveler's trips.
    """
    def __init__(self, 
                 dates: List[datetime],
                 facility_date_time_step_urbano_agent_dic:dict,
                 urbano_agent_date_time_step_facility_dic:dict,

                 urbano_agents_travelers_mapping: dict,
                 traveler_urbano_agent_mapping: dict,
                 baseline_time_use_dict: dict,
                 county_mobility_changes_dict: dict, 
                 stationary_distributions: dict,
                 facility_to_index : dict):
        self.Dates = dates
        # TODO: if time span in each time step is not the same, adjust this dic
        self.time_span_of_each_time_step_dic = {ts: 1/6 for ts in range(1,7)} 

        # format: {(facility_name, visit_time_step): {date: {time_step: [(home_block, ua_id, ua_type)]}}}
        self.facility_date_time_step_urbano_agent_dic = facility_date_time_step_urbano_agent_dic
        # format: {(home_block, ua_id, ua_type): {date: set((facility_name, visit_time_step))}}
        self.urbano_agent_date_time_step_facility_dic = urbano_agent_date_time_step_facility_dic

        # format: {(home_block, ua_id, ua_type):{traveler id}}
        self.urbano_agents_travelers_mapping = urbano_agents_travelers_mapping
        # format: {traveler_id: ((home_block, ua_id, ua_type))}
        self.traveler_urbano_agent_mapping = traveler_urbano_agent_mapping
        # format: {((home_block, ua_type)): [(home_block, ua_id, ua_type)]}
        self.mega_agent_Urbano_mapping = SimulationPeriodBasicInfo.get_MegaAgent_Urbano_mapping(list(urbano_agents_travelers_mapping.keys()))
        # the dict is in the format of: {(home_block, agent_id, agent_type): [travelers' IDs]}
        self.MegaAgent_Travelers, self.AllTravelers = SimulationPeriodBasicInfo.get_travelers_in_MegaAgents(urbano_agents_travelers_mapping,
                                                                                                            list(self.mega_agent_Urbano_mapping))

        # self.AllMegaAgentsNames = all_HomeBlock_TravelerType_pairs
        # all the unique pairs of activity_type and home_block, and its corresponding
        # index, eg:("other",16): 43
        # some facilities may not ever be visited.
        self.FacilityToIndex = facility_to_index
        self.IndexToFacilityObject = {}
        self.AllFacilityNames = list(self.FacilityToIndex.keys())
        # the below two dict are for retrieving facility visit probability. 
        # should be used paired with all_facilities
        self.StationaryDistributions = stationary_distributions
        # the dict is in the format of: {date: {facilityname: time_use}}
        self.AllDatesFacilityTimeUse = SimulationPeriodBasicInfo.get_facility_time_use_for_all_dates(dates, 
                                                                                                    self.AllFacilityNames, 
                                                                                                    baseline_time_use_dict, 
                                                                                                    county_mobility_changes_dict)

        # date_urbano_agent_timeStep_FacilityTypeVisitCount_dic format: {date: {(home_block, ua_id, ua_type): {facility_type: {time_step: visit_count}}}}
        # date_urbano_agent_timeStep_AllFacilityVisitCount_dic format: {date: {(home_block, ua_id, ua_type): {time_step: all_visited_facility_count}}}
        self.date_urbano_agent_timeStep_FacilityTypeVisitCount_dic, self.date_urbano_agent_timeStep_AllFacilityVisitCount_dic = self.get_urbano_agent_facilityVisitCount(dates, urbano_agents_travelers_mapping)


    @staticmethod
    def get_facility_time_use_for_all_dates(dates: List[datetime], 
                                            allFacilityNames, 
                                            baselineTimeUse_dict:dict, 
                                            countyMobilityChanges_dict:dict) -> dict:
        time_use_dict = {}
        for d in dates:
            time_use_dict[d] = {}
            for facilityName in allFacilityNames:
                activity = facilityName[0]
                if_weekday = d.weekday() < 5
                activity_time_use_dict = None
                if if_weekday:
                    activity_time_use_dict = baselineTimeUse_dict["Avg prcntg/day, civilian pop, Weekdays"]
                else:
                    activity_time_use_dict = baselineTimeUse_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]
                
                base_line_time = activity_time_use_dict[activity]
                time_change_percentage = countyMobilityChanges_dict[d][activity]
                
                time_use_dict[d][facilityName] = base_line_time * (1 + time_change_percentage/100)
        return time_use_dict


    @staticmethod
    def get_travelers_in_MegaAgents(urbano_agents_travelers_mapping: dict, 
                                    all_HomeBlock_TravelerType_pairs: List[tuple]) -> dict:
        """
        groups_travelers_mapping is in the format of 
        {(home_block, agent_id, agent_type): [travelers' IDs]}. Convert the format into
        {(home_block, agent_type): [travelers' IDs]}
        """
        homeblock_travelertype_travelers = {}
        all_travelers = set()
        for pair in all_HomeBlock_TravelerType_pairs:
            homeblock_travelertype_travelers[pair] = set()

        for group_info, travelers in urbano_agents_travelers_mapping.items():
            # group_info is in the format of: (home_block, agent_id, agent_type)
            home_block = group_info[0]
            agent_type = group_info[2]
            mega_agent_name = (home_block, agent_type)
            homeblock_travelertype_travelers[mega_agent_name].update(travelers)
            all_travelers.update(travelers)
        return homeblock_travelertype_travelers, all_travelers

    
    @staticmethod
    def get_MegaAgent_Urbano_mapping(urbano_agent_list):
        MegaAgent_Urbano_map = dict()
        for (home_block, ua_id, ua_type) in urbano_agent_list:
            if (home_block, ua_type) not in MegaAgent_Urbano_map:
                MegaAgent_Urbano_map[(home_block, ua_type)] = set()
            MegaAgent_Urbano_map[(home_block, ua_type)].add(((home_block, ua_id, ua_type)))
        return MegaAgent_Urbano_map


    def get_urbano_agent_facilityVisitCount(self, dates, urbano_agents_travelers_mapping) -> None:
            date_urbano_agent_timeStep_FacilityTypeVisitCount_dic = dict()
            date_urbano_agent_timeStep_AllFacilityVisitCount_dic = dict()

            for date in dates:
                date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date] = dict()
                date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date] = dict()

                for urbano_agent, residents in urbano_agents_travelers_mapping.items():
                    if urbano_agent not in date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date]:
                        date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent] = dict()
                    if urbano_agent not in date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date]:
                        date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent] = dict()

                    visit_facilities = self.urbano_agent_date_time_step_facility_dic[urbano_agent][date]
                    for facility_name, visit_time_step in visit_facilities:
                        facility_type = facility_name[0]
                        if facility_type not in date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent]:
                            date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type] = dict()
                        if visit_time_step not in date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type]:
                            date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type][visit_time_step] = 0
                        date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type][visit_time_step] += 1                        
                            
                        if visit_time_step not in date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent]:
                            date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent][visit_time_step] = 0
                        date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent][visit_time_step] += 1
                
            return date_urbano_agent_timeStep_FacilityTypeVisitCount_dic, date_urbano_agent_timeStep_AllFacilityVisitCount_dic


class TimeUseDataCreator:
    """
    Create time_use_baseline_dict and county_mobility_changes_dict.
    """
    @staticmethod
    def create(facility_time_use_file: str,
               mobility_change_file: str,
               county: str):
        (time_use_baseline_df, 
         county_mobility_changes_df) = get_baseline_time_and_mobility_change(facility_time_use_file,
                                                                             mobility_change_file,
                                                                             county=county)
        (time_use_baseline_dict, 
         county_mobility_changes_dict) = convert_dfs_to_dict(time_use_baseline_df, 
                                                             county_mobility_changes_df)
        return time_use_baseline_dict, county_mobility_changes_dict

class StationaryDistributionCreator:
    """
    Create facility_to_index and stationary_distributions of visiting facilities.
    stationary_distributions should be updated on daily basis.
    """
    @staticmethod
    def create(TotalBlocks_count: int,
               urbano_trip_volumn_file: str):
        """
        Create stationary distribution of one day.
        """
        (stationary_distributions, 
         facility_to_index) = create_tm_sd_for_all_homeblock_travelertype_pair(urbano_trip_volumn_file,
                                                                               TotalBlocks_count)
        return stationary_distributions, facility_to_index


class TripDataCreator:
    """
    Create urbano_agents_travelers_mapping, traveler_urbano_agent_mapping, 
    facility_date_time_step_urbano_agent_dic and urbano_agent_date_time_step_facility_dic.
    """

    # not being used, too slow
    @staticmethod
    def create0(start_date_string: str, 
               simulation_date_count: int,
               urbano_trip_volumn_files: List[str], 
               all_facilities: List[tuple]):
        if simulation_date_count != len(urbano_trip_volumn_files):
            raise ValueError("number of days should be equal to number of urbano_trip_volumn_file")

        start_date_time = datetime.strptime(start_date_string, '%Y-%m-%d')
        dates = [start_date_time + timedelta(i) for i in range(simulation_date_count)]
        
        facility_date_time_step_urbano_agent_dic = {}
        urbano_agent_date_time_step_facility_dic = {}
        agents_population_mapping = None

        ################################################ try parallel computing
        # create data for each day
        for i in range(simulation_date_count):
            urbano_trip_volumn_file = urbano_trip_volumn_files[i]
            date = dates[i]

            (agents_trips_record, 
            agents_population_mapping) = reoranize_agent_trip_records(urbano_trip_volumn_file)

            (facility_time_step_urbano_agent_dic, 
            urbano_agent_time_step_facility_dic) = create_facility_visit_urbano_agent_record_parallel_computing(agents_trips_record, all_facilities)

            for facility_name, time_step_urbano_agent_dic in facility_time_step_urbano_agent_dic.items():
                if facility_name not in facility_date_time_step_urbano_agent_dic:
                    facility_date_time_step_urbano_agent_dic[facility_name] = dict()
                facility_date_time_step_urbano_agent_dic[facility_name][date] = time_step_urbano_agent_dic
            
            for urbano_agent, time_step_facilities in urbano_agent_time_step_facility_dic.items():
                if urbano_agent not in urbano_agent_date_time_step_facility_dic:
                    urbano_agent_date_time_step_facility_dic[urbano_agent] = dict()
                urbano_agent_date_time_step_facility_dic[urbano_agent][date] = time_step_facilities
        #################################################

        (urbano_agents_travelers_mapping, 
         traveler_urbano_agent_mapping, 
         total_population) = map_agent_to_traveler_and_traveler_to_agent(agents_population_mapping)
                                                                                                             
        return (urbano_agents_travelers_mapping, 
                traveler_urbano_agent_mapping, 
                facility_date_time_step_urbano_agent_dic, 
                urbano_agent_date_time_step_facility_dic)


    @staticmethod
    def create(start_date_string: str, 
               simulation_date_count: int,
               urbano_trip_volumn_files: List[str], 
               all_facilities: List[tuple], 
               max_processor_num=8):
        if simulation_date_count != len(urbano_trip_volumn_files):
            raise ValueError("number of days should be equal to number of urbano_trip_volumn_file")

        start_date_time = datetime.strptime(start_date_string, '%Y-%m-%d')
        dates = [start_date_time + timedelta(i) for i in range(simulation_date_count)]
        
        _, agents_population_mapping = reoranize_agent_trip_records(urbano_trip_volumn_files[-1])
        (urbano_agents_travelers_mapping, 
         traveler_urbano_agent_mapping, 
         _) = map_agent_to_traveler_and_traveler_to_agent(agents_population_mapping)

        facility_date_time_step_urbano_agent_dic = {}
        urbano_agent_date_time_step_facility_dic = {}

        with ProcessPoolExecutor(max_workers=max_processor_num) as executor:
            futures = [executor.submit(TripDataCreator.process_date, i, dates, urbano_trip_volumn_files, all_facilities) for i in range(simulation_date_count)]
            for future in futures:
                date, facility_time_step_urbano_agent_dic_part, urbano_agent_time_step_facility_dic_part = future.result()

                # Merge facility_time_step_urbano_agent_dic
                for facility_name, time_step_urbano_agent_dic in facility_time_step_urbano_agent_dic_part.items():
                    if facility_name not in facility_date_time_step_urbano_agent_dic:
                        facility_date_time_step_urbano_agent_dic[facility_name] = {}
                    facility_date_time_step_urbano_agent_dic[facility_name][date] = time_step_urbano_agent_dic

                # Merge urbano_agent_time_step_facility_dic
                for urbano_agent, time_step_facilities in urbano_agent_time_step_facility_dic_part.items():
                    if urbano_agent not in urbano_agent_date_time_step_facility_dic:
                        urbano_agent_date_time_step_facility_dic[urbano_agent] = {}
                    urbano_agent_date_time_step_facility_dic[urbano_agent][date] = time_step_facilities
                                                                       
        return (urbano_agents_travelers_mapping, 
                traveler_urbano_agent_mapping, 
                facility_date_time_step_urbano_agent_dic, 
                urbano_agent_date_time_step_facility_dic)
    
    @staticmethod
    def process_date(i, dates, urbano_trip_volumn_files, all_facilities):
        urbano_trip_volumn_file = urbano_trip_volumn_files[i]
        date = dates[i]

        (agents_trips_record, 
        agents_population_mapping) = reoranize_agent_trip_records(urbano_trip_volumn_file)

        (facility_time_step_urbano_agent_dic, 
        urbano_agent_time_step_facility_dic) = create_facility_visit_urbano_agent_record(agents_trips_record, all_facilities)
        return date, facility_time_step_urbano_agent_dic, urbano_agent_time_step_facility_dic