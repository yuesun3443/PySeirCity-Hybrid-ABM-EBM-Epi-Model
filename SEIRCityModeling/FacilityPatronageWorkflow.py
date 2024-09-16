import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Union
import concurrent.futures

# This code is to find visitors corresponding to each facility, and performs the following steps: 
# 1.Loads trip information into a DataFrame from an Excel file.
# 2.Defines city and facility parameters.
# 3.Identifies unique agents based on combinations of TravelerHomeBlock and TravelerAgentID.\
# 4.Records each agent's trips and the number of travelers they represent.
# 5.Assigns a unique ID to individual travelers represented by each agent.
# 6.Initializes and populates records of facilities visited by each agent.
# 7.Converts agent-based facility visit records to individual traveler-based records, aggregating all traveler IDs visiting each facility.



def getTotalPopulation(date_facility_visit_travelers_dict):
    ts = set()
    first_date = list(date_facility_visit_travelers_dict.keys())[0]
    for f, visitors in date_facility_visit_travelers_dict[first_date].items():
        ts.update(visitors)
    return ts


def reoranize_agent_trip_records(urbano_trip_file):
    # Load the data into a pandas DataFrame
    urbano_trip_df = pd.read_excel(urbano_trip_file, sheet_name='TripInformation')

    # Unique pairs of TravelerHomeBlock and TravelerID and TravelerType
    # Each pair indicates a unique agent
    # Note that one agent might represent a group of travelers with the exactly same schedules
    unique_pairs = urbano_trip_df[['TravelerHomeBlock', 'TravelerAgentID', 'TravelerType']].drop_duplicates()

    # Initialize dictionaries for storing trips and population sizes by agent, and travelers' IDs by agent
    urbano_agents_trips_record = {} # format => Traveler Agent ID: dataframe of trips
    agents_population_mapping = {} # format => Traveler Agent ID: agent population

    for _, row in unique_pairs.iterrows():
        home_block = row["TravelerHomeBlock"]
        agent_id = row["TravelerAgentID"]
        agent_type = row['TravelerType']
        agent_df = urbano_trip_df[(urbano_trip_df['TravelerAgentID'] == agent_id)&
                                  (urbano_trip_df['TravelerHomeBlock'] == home_block)&
                                  (urbano_trip_df['TravelerType'] == agent_type)]

        agent_homeblock_id_type = (home_block, agent_id, agent_type)
        urbano_agents_trips_record[agent_homeblock_id_type] = agent_df  
        agents_population_mapping[agent_homeblock_id_type] = agent_df.iloc[0]["AsPopulation"]
    return urbano_agents_trips_record, agents_population_mapping


def map_agent_to_traveler_and_traveler_to_agent(agents_population_mapping: dict):
    # Initialize a dictionary for tracking each agent's travelers' IDs
    # Meanwhile, create mapping between individual traveler and its belonging agent
    agents_travelers_mapping = {} # format => Agent ID: [travelers' id]
    traveler_agent_mapping = {} # format => Trvaler's ID: Agent ID
    id_iter = 0 # Unique ID counter for individual travelers
    
    for agent_info, population in agents_population_mapping.items():
        traveler_ids = set()
        # Assign unique IDs to travelers represented by this agent
        for _ in range(population):
            traveler_id = id_iter
            traveler_ids.add(traveler_id)
            traveler_agent_mapping[traveler_id] = agent_info

            id_iter += 1
        agents_travelers_mapping[agent_info] = traveler_ids

    total_population = id_iter
    return agents_travelers_mapping, traveler_agent_mapping, total_population


# End up not being used
def create_facility_visit_travelers_record(urbano_agents_trips_record: dict,
                                           agents_travelers_mapping: dict,
                                           all_facilities: List[tuple]):
    """
    A dictionary to contain visiting records for each facility by date.
    The dictionary is in the format of: {date:{facilities:[travelers]}}
    """
    # Initialize a dictionary to store facilities and the agents who visited them
    facility_visit_agent_record = {facility: {time_step: set() for time_step in range(1, 7)} for facility in all_facilities}
    
    # Iterate through each agent's trip records to populate the facility visitation records
    for agent_info, trips_df in urbano_agents_trips_record.items():  
        AgentHomeBlock,AgentID,AgentType = agent_info[0], agent_info[1], agent_info[2]
        trips_df = adjust_urbano_agent_trip_record(trips_df, AgentHomeBlock, AgentID, AgentType)

        for time_step in range(1,7):
            time_step_trips_df = trips_df[trips_df['TimeOfDay']==time_step]

            # Iterate through the trips DataFrame for this agent
            for _, trip_row in time_step_trips_df.iterrows():
                # Identify facilities involved in each trip (both departure and arrival)
                # from_facility = (trip_row['FromActivity'], trip_row['OriginBlockID'])
                to_facility = (trip_row['ToActivity'], trip_row['DestBlockID'])

                # Update facility visit records with this agent's ID
                # facility_visit_agent_record[from_facility][time_step].add(agent_info)
                facility_visit_agent_record[to_facility][time_step].add(agent_info)

    # Map individual travelers to facilities based on the agents who visited them
    facility_visit_traveler_record = {facility: {} for facility in all_facilities}
    # Map facilities to travelers
    traveler_visit_facility_record = {}

    for facility, timeStep_agents_dict in facility_visit_agent_record.items():
        for time_step, agents in timeStep_agents_dict.items():
            all_visit_travelers = set()
            # Aggregate all individual traveler IDs for this facility based on visiting agents
            for agent_info in agents:
                travelers = agents_travelers_mapping.get(agent_info, [])
                all_visit_travelers.update(travelers)
            facility_visit_traveler_record[facility][time_step] = all_visit_travelers

            for traveler in all_visit_travelers:
                if traveler not in traveler_visit_facility_record:
                    traveler_visit_facility_record[traveler] = set()
                traveler_visit_facility_record[traveler].add((facility, time_step))

    return facility_visit_traveler_record, traveler_visit_facility_record


def create_facility_visit_urbano_agent_record(urbano_agents_trips_record: dict,
                                              all_facilities: List[tuple]):
    """
    A dictionary to contain visiting records for each facility by date.
    The dictionary is in the format of: {date:{facilities:[travelers]}}
    """ 
    # Initialize a dictionary to store facilities and the agents who visited them
    facility_time_step_urbano_agent_dic = {facility: 
                                            {time_step: set() for time_step in range(1, 7)} 
                                            for facility in all_facilities
                                          }
    
    # Iterate through each agent's trip records to populate the facility visitation records
    for agent_info, trips_df in urbano_agents_trips_record.items():  
        AgentHomeBlock,AgentID,AgentType = agent_info[0], agent_info[1], agent_info[2]
        trips_df = adjust_urbano_agent_trip_record(trips_df, AgentHomeBlock, AgentID, AgentType)

        for time_step in range(1,7):
            time_step_trips_df = trips_df[trips_df['TimeOfDay']==time_step]

            # Iterate through the trips DataFrame for this agent
            for _, trip_row in time_step_trips_df.iterrows():
                # Identify facilities involved in each trip (both departure and arrival)
                # from_facility = (trip_row['FromActivity'], trip_row['OriginBlockID'])
                to_facility = (trip_row['ToActivity'], trip_row['DestBlockID'])

                # Update facility visit records with this agent's ID
                # facility_visit_agent_record[from_facility][time_step].add(agent_info)
                facility_time_step_urbano_agent_dic[to_facility][time_step].add(agent_info)

    # Map facilities to urbano_agent
    urbano_agent_time_step_facility_dic = dict()

    for facility, timeStep_agents_dict in facility_time_step_urbano_agent_dic.items():
        for time_step, agents in timeStep_agents_dict.items():
            # urbano_agent is (home_block, ua_id, ua_type)
            for urbano_agent in agents:
                if urbano_agent not in urbano_agent_time_step_facility_dic:
                    urbano_agent_time_step_facility_dic[urbano_agent] = set()
                urbano_agent_time_step_facility_dic[urbano_agent].add((facility, time_step))
    return facility_time_step_urbano_agent_dic, urbano_agent_time_step_facility_dic


# End up not being used
def create_facility_visit_travelers_record_parallel_computing(urbano_agents_trips_record: dict,
                                                              agents_travelers_mapping: dict,
                                                              all_facilities: List[tuple]):
    """
    A dictionary to contain visiting records for each facility by date.
    The dictionary is in the format of: {date:{facilities:[travelers]}}
    """    
    # Iterate through each agent's trip records to populate the facility visitation records
    # Convert the dictionary items to a list to be compatible with executor.map
    items = list(urbano_agents_trips_record.items())

    # Using ProcessPoolExecutor to parallelize processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = dict(executor.map(supplement_agent_trips_one_by_one, items))
    # Wait for all tasks to complete (implicit in map)

    # recreate the dictionary with processed data
    items = list(results.items())

    # Using ProcessPoolExecutor to parallelize processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_facility_visit_agent_record, items))

    # Initialize a dictionary to store facilities and the agents who visited them
    facility_visit_agent_record = {facility: {time_step: set() for time_step in range(1, 7)} for facility in all_facilities}
    # Iterate through the list of dictionaries and merge the lists
    for dictionary in results:
        for facility, time_step_agents in dictionary.items():
            for time_step, agents in time_step_agents.items():
                facility_visit_agent_record[facility][time_step].update(agents) # Merge lists for the same keys

    # Map individual travelers to facilities based on the agents who visited them
    facility_visit_traveler_record = {facility: {} for facility in all_facilities}
    # Map facilities to travelers
    traveler_visit_facility_record = {}

    for facility, timeStep_agents_dict in facility_visit_agent_record.items():
        for time_step, agents in timeStep_agents_dict.items():
            all_visit_travelers = set()
            # Aggregate all individual traveler IDs for this facility based on visiting agents
            for agent_info in agents:
                travelers = agents_travelers_mapping.get(agent_info, [])
                all_visit_travelers.update(travelers)
            facility_visit_traveler_record[facility][time_step] = all_visit_travelers

            for traveler in all_visit_travelers:
                if traveler not in traveler_visit_facility_record:
                    traveler_visit_facility_record[traveler] = set()
                traveler_visit_facility_record[traveler].add((facility, time_step))

    return facility_visit_traveler_record, traveler_visit_facility_record


def create_facility_visit_urbano_agent_record_parallel_computing(urbano_agents_trips_record: dict,
                                                                 all_facilities: List[tuple]):
    """
    A dictionary to contain visiting records for each facility by date.
    The dictionary is in the format of: {date:{facilities:[travelers]}}
    """    
    # Iterate through each agent's trip records to populate the facility visitation records
    # Convert the dictionary items to a list to be compatible with executor.map
    items = list(urbano_agents_trips_record.items())

    # Using ProcessPoolExecutor to parallelize processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = dict(executor.map(supplement_agent_trips_one_by_one, items))
    # Wait for all tasks to complete (implicit in map)

    # recreate the dictionary with processed data
    items = list(results.items())

    # Using ProcessPoolExecutor to parallelize processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_facility_visit_agent_record, items))

    # Initialize a dictionary to store facilities and the agents who visited them
    facility_time_step_urbano_agent_dic = {facility: 
                                            {time_step: set() for time_step in range(1, 7)}
                                            for facility in all_facilities
                                          }
    # Iterate through the list of dictionaries and merge the lists
    for dictionary in results:
        for facility, time_step_agents in dictionary.items():
            for time_step, agents in time_step_agents.items():
                facility_time_step_urbano_agent_dic[facility][time_step].update(agents) # Merge lists for the same keys

    # Map facilities to urbano_agent
    urbano_agent_time_step_facility_dic = dict()

    for facility, timeStep_agents_dict in facility_time_step_urbano_agent_dic.items():
        for time_step, agents in timeStep_agents_dict.items():
            # urbano_agent is (home_block, ua_id, ua_type)
            for urbano_agent in agents:
                if urbano_agent not in urbano_agent_time_step_facility_dic:
                    urbano_agent_time_step_facility_dic[urbano_agent] = set()
                urbano_agent_time_step_facility_dic[urbano_agent].add((facility, time_step))
    return facility_time_step_urbano_agent_dic, urbano_agent_time_step_facility_dic


def adjust_urbano_agent_trip_record(trip_df, 
                                    AgentHomeBlock, 
                                    AgentID, 
                                    AgentType):
    # urbano agent is in the format of ('TravelerHomeBlock', 'TravelerAgentID', 'TravelerType')
    for current_time_step in range(1,7):
        time_step_trip_df = trip_df[trip_df["TimeOfDay"]==current_time_step]
        
        if len(time_step_trip_df)==0:
            row_dict = {'TravelerAgentID':AgentID, 
                        'TravelerType':AgentType,
                        'TravelerHomeBlock':AgentHomeBlock,
                        'TimeOfDay':current_time_step, 
                        'TripOrder':1,
                        'FromActivity':'home',
                        'ToActivity':'home', 
                        'OriginBlockID':AgentHomeBlock, 
                        'DestBlockID':AgentHomeBlock, 
                        'AsPopulation':list(trip_df['AsPopulation'])[0]
                    }
            
            if current_time_step==1:
                # handle the situation where there is no trip in the first time_step
                row_df = pd.DataFrame([row_dict]) # Convert the dictionary to a DataFrame with one row
                trip_df = pd.concat([trip_df, row_df], ignore_index=True)
            elif current_time_step!=1 and current_time_step!=6:
                # handle the situation where there is no trip in the 2-5 time_step,
                # if no trip in either time step, have to check the previous visiting facility, and use the last 
                # visiting facility from the previous time step as the visiting facility at the current
                prev_time_step = current_time_step - 1
                # get the trip df of the previous time step
                prev_time_step_trip_df = trip_df[trip_df['TimeOfDay']==prev_time_step]
                # get the last trip order of the previous time step
                prev_time_step_last_trip_order = max(dict(prev_time_step_trip_df[prev_time_step_trip_df['TimeOfDay']==prev_time_step]['TripOrder']).values())
                prev_time_step_last_trip_df = prev_time_step_trip_df[prev_time_step_trip_df['TripOrder']==prev_time_step_last_trip_order]
                # find the last visit facility of the previous time step
                to_facility_type = list(prev_time_step_last_trip_df['ToActivity'])[0]
                to_BlockID = list(prev_time_step_last_trip_df['DestBlockID'])[0]
                # add the facility to the row_dict, add it to FromActivity and ToActivity
                row_dict['FromActivity'] = to_facility_type
                row_dict['ToActivity'] = to_facility_type
                row_dict['OriginBlockID'] = to_BlockID
                row_dict['DestBlockID'] = to_BlockID
                
                row_df = pd.DataFrame([row_dict])
                trip_df = pd.concat([trip_df, row_df], ignore_index=True)
            elif current_time_step==6:
                # handle the situation where there is no trip in the last time_step
                prev_time_step = current_time_step - 1
                # get the trip df of the previous time step
                prev_time_step_trip_df = trip_df[trip_df['TimeOfDay']==prev_time_step]
                # get the last trip order of the previous time step
                prev_time_step_last_trip_order = max(dict(prev_time_step_trip_df[prev_time_step_trip_df['TimeOfDay']==prev_time_step]['TripOrder']).values())
                prev_time_step_last_trip_df = prev_time_step_trip_df[prev_time_step_trip_df['TripOrder']==prev_time_step_last_trip_order]
                # find the last visit facility of the previous time step
                to_facility_type = list(prev_time_step_last_trip_df['ToActivity'])[0]
                to_BlockID = list(prev_time_step_last_trip_df['DestBlockID'])[0]
                # check if the last visit facility of the previous time step is home
                if to_facility_type == 'home':
                    row_df = pd.DataFrame([row_dict])
                    trip_df = pd.concat([trip_df, row_df], ignore_index=True)
                else:
                    # raise RuntimeError('Every traveler should eventually go back home.' +\
                    #                    ' Defect Urbano Agent: ' + str((AgentHomeBlock,AgentID,AgentType)))
                    # add the facility to the row_dict, add it to FromActivity and ToActivity
                    row_dict['FromActivity'] = to_facility_type
                    row_dict['OriginBlockID'] = to_BlockID
                    row_df = pd.DataFrame([row_dict])
                    trip_df = pd.concat([trip_df, row_df], ignore_index=True)
    return trip_df


def supplement_agent_trips_one_by_one(agent_trips_tuple):
    """
    This function is created for parallel computing. It supplements the trips of one agent to 
    make every time step will have at least one trip happens.
    """
    agent_info, trips_df = agent_trips_tuple[0], agent_trips_tuple[1]
    AgentHomeBlock,AgentID,AgentType = agent_info[0], agent_info[1], agent_info[2]
    # Assuming FacilityPatronageWorkflow.adjust_urbano_agent_trip_record is a static method or similarly accessible
    trips_df = adjust_urbano_agent_trip_record(trips_df, AgentHomeBlock, AgentID, AgentType)
    return (agent_info, trips_df)


def get_facility_visit_agent_record(agent_trips_tuple):
    agent_info, trips_df = agent_trips_tuple[0], agent_trips_tuple[1]
    facility_visit_agent_record = dict()
    for time_step in range(1,7):
        time_step_trips_df = trips_df[trips_df['TimeOfDay']==time_step]

        # Iterate through the trips DataFrame for this agent
        for _, trip_row in time_step_trips_df.iterrows():
            # Identify facilities involved in each trip (both departure and arrival)
            # from_facility = (trip_row['FromActivity'], trip_row['OriginBlockID'])
            to_facility = (trip_row['ToActivity'], trip_row['DestBlockID'])

            # Update facility visit records with this agent's ID
            # facility_visit_agent_record[from_facility][time_step].add(agent_info)
            if to_facility not in facility_visit_agent_record:
                facility_visit_agent_record[to_facility] = dict()
            if time_step not in facility_visit_agent_record[to_facility]:
                facility_visit_agent_record[to_facility][time_step] = set()
            facility_visit_agent_record[to_facility][time_step].add(agent_info)
    return facility_visit_agent_record


