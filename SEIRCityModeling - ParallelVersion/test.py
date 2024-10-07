import pickle
import cProfile
import pstats
import os
from threading import Thread
from multiprocessing import Process, Manager
import multiprocessing

from facility import Facility
from megaagent import State, MegaAgent

from seirstat import Statistics
from preprocessing import SimulationPeriodBasicInfo
from parameters import Parameters, VariedInfectionDurationResponse
from interventions import GeneralTesting, ContactTracing
from simulation import Simulation


def reset_facilities_and_MegaAgents(simulation_period_basic_info) -> None:
    for facility_name, facility in all_facilities_objects.items():
        facility.reset_facility(simulation_period_basic_info)
    for mega_agent_name, mega_agent in all_mega_agents.items():
        mega_agent.reset_MegaAgent()

def worker(shared_list, i):
    ag = shared_list[i]
    MegaAgent.labelTravelsAsExposed(*ag)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    current_directory = os.getcwd()  # Get the current working directory
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    print("Parent Directory:", parent_directory)

    # Load serialized data 
    with open(parent_directory+'\\data\\dates.pkl', 'rb') as file:
        dates = pickle.load(file)
        
    with open(parent_directory+'\\data\\facility_date_time_step_urbano_agent_dic.pkl', 'rb') as file:
        facility_date_time_step_urbano_agent_dic = pickle.load(file)
    with open(parent_directory+'\\data\\urbano_agent_date_time_step_facility_dic.pkl', 'rb') as file:
        urbano_agent_date_time_step_facility_dic = pickle.load(file)
        
    with open(parent_directory+'\\data\\urbano_agents_travelers_mapping.pkl', 'rb') as file:
        urbano_agents_travelers_mapping = pickle.load(file)
    with open(parent_directory+'\\data\\traveler_urbano_agent_mapping.pkl', 'rb') as file:
        traveler_urbano_agent_mapping = pickle.load(file)
        
    with open(parent_directory+'\\data\\time_use_baseline_dict.pkl', 'rb') as file:
        time_use_baseline_dict = pickle.load(file)
    with open(parent_directory+'\\data\\county_mobility_changes_dict.pkl', 'rb') as file:
        county_mobility_changes_dict = pickle.load(file)
    with open(parent_directory+'\\data\\stationary_distributions.pkl', 'rb') as file:
        stationary_distributions = pickle.load(file)
    with open(parent_directory+'\\data\\facility_to_index.pkl', 'rb') as file:
        facility_to_index = pickle.load(file)



    repetitions = 1
    initial_exposed_count = 10
    # discount percentage of hazard brought by an asymptomatic individual
    asym_hazard_multiplier = 0.5

    parameters = Parameters(repetitions, initial_exposed_count, asym_hazard_multiplier)
    parameters.infection_duration = VariedInfectionDurationResponse(rate_from_S_to_E=2,
                                                                rate_from_E_to_I=1/2, 
                                                                rate_from_asymI_to_R=1/8, 
                                                                rate_from_symI_to_R=1/8, 
                                                                asym_fraction=0, 
                                                                asym_hazard_multiplier=asym_hazard_multiplier)

    # Create simulation_period_basic_info object
    simulation_period_basic_info = SimulationPeriodBasicInfo(dates,
                                                        facility_date_time_step_urbano_agent_dic,
                                                        urbano_agent_date_time_step_facility_dic,
                                                        urbano_agents_travelers_mapping,
                                                        traveler_urbano_agent_mapping,
                                                        time_use_baseline_dict,
                                                        county_mobility_changes_dict,
                                                        stationary_distributions,
                                                        facility_to_index)
                                                
    # Create all Facility objects
    all_facilities_objects = {}
    for f_name in simulation_period_basic_info.AllFacilityNames:
        facility_object = Facility(f_name, simulation_period_basic_info)
        all_facilities_objects[f_name] = facility_object


    # Create all MegaAgents
    all_mega_agents = {}
    for MegaAgentName in simulation_period_basic_info.mega_agent_Urbano_mapping.keys():
        travelers_in_mega_agent = simulation_period_basic_info.MegaAgent_Travelers[MegaAgentName]   
        if travelers_in_mega_agent==0:
            continue
            
        mega_agent = MegaAgent(MegaAgentName,travelers_in_mega_agent,simulation_period_basic_info)
        all_mega_agents[MegaAgentName] = mega_agent

    # initialize Statistics
    stat = Statistics(all_mega_agents, dates)

    


    for _ in range(repetitions):
        initial_infectious = list(simulation_period_basic_info.AllTravelers)[:parameters.initial_exposed_count]

        for mega_agent_name, mega_agent in all_mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue

            if not mega_agent.MegaAgentState.S_travelers.isdisjoint(initial_infectious):
                initial_infectious_travelers_in_mega_agent = list(mega_agent.MegaAgentState.S_travelers.intersection(initial_infectious))
            else:
                continue

            first_date = dates[0]
            mega_agent.dynamic_time_spent_dic = MegaAgent.get_dynamic_time_spent(first_date,
                                                                                mega_agent.UrbanoAgents,
                                                                                simulation_period_basic_info)

            MegaAgent.initialize_MegaAgent(first_date, 
                                            parameters,
                                            mega_agent.MegaAgentState.S_travelers,
                                            mega_agent.MegaAgentState.Is_travelers_list,
                                            mega_agent.MegaAgentState.Ia_travelers_list,
                                            mega_agent.MegaAgentState.Is,
                                            mega_agent.MegaAgentState.Ia,
                                            mega_agent.new_Is_count,
                                            mega_agent.new_Ia_count,
                                            mega_agent.MegaAgentState.R_travelers,
                                            mega_agent.MegaAgentState.Q_travelers,
                                            mega_agent.MegaAgentState.Qe_travelers,
                                            mega_agent.MegaAgentState.Qa_travelers,
                                            mega_agent.MegaAgentState.Qs_travelers,
                                            initial_infectious_travelers_in_mega_agent)
            mega_agent.record_daily_stat(first_date, stat)    
            mega_agent.if_initialized = True

        for date in dates:
            print(date)

            # On each day, at the first time step, calculate dynamic time spent for each travelers
            for mega_agent_name, mega_agent in all_mega_agents.items():
                if mega_agent.if_initialized == True and date == dates[0]:
                    continue
                mega_agent.dynamic_time_spent_dic = MegaAgent.get_dynamic_time_spent(date,
                                                                                    mega_agent.UrbanoAgents,
                                                                                    simulation_period_basic_info)

            for time_step in range(1,7):
                for f_name, f in all_facilities_objects.items():
                    f.FacilityHazard = Facility.computeFacilityHazard(f_name,
                                                                      f.FaciltyIndex,
                                                                      date,
                                                                      time_step,
                                                                      all_mega_agents, 
                                                                      parameters, 
                                                                      simulation_period_basic_info,
                                                                      f.DateTimeStepAllVisitors)
                
                # manager = Manager()
                # shared_list = manager.list()
                # for mega_agent_name, mega_agent in all_mega_agents.items():
                #     if mega_agent.MegaAgentPopulation == 0:
                #         continue
                #     if mega_agent.if_initialized == True and date == dates[0] and time_step == 1:
                #         continue
                #     shared_list.append((mega_agent_name, 
                #                         date, 
                #                         time_step, 
                #                         all_facilities_objects, 
                #                         mega_agent.UrbanoAgents,
                #                         simulation_period_basic_info,
                #                         mega_agent.dynamic_time_spent_dic,
                #                         mega_agent.MegaAgentState.S_travelers,
                #                         mega_agent.MegaAgentState.E_travelers_list,
                #                         mega_agent.MegaAgentState.V_travelers))

                # ps = []
                # for i in range(len(shared_list)):
                #     p = Process(target=worker, args=(shared_list, i))
                #     ps.append(p)
                #     p.start()
                # for p in ps:
                #     p.join()

                ps = []
                for mega_agent_name, mega_agent in all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue

                    if mega_agent.if_initialized == True and date == dates[0] and time_step == 1:
                        continue
                    else:
                        p = Thread(target=MegaAgent.labelTravelsAsExposed, args = (mega_agent_name,
                                                                                    date, 
                                                                                    time_step,
                                                                                    all_facilities_objects,
                                                                                    mega_agent.UrbanoAgents,
                                                                                    simulation_period_basic_info,
                                                                                    mega_agent.dynamic_time_spent_dic,
                                                                                    mega_agent.MegaAgentState.S_travelers,
                                                                                    mega_agent.MegaAgentState.E_travelers_list,
                                                                                    mega_agent.MegaAgentState.V_travelers))
                        p.start()
                        ps.append(p)
                
                for p in ps: p.join()


                # for mega_agent_name, mega_agent in all_mega_agents.items():
                #     if mega_agent.MegaAgentPopulation == 0:
                #         continue

                #     if mega_agent.if_initialized == True and date == dates[0] and time_step == 1:
                #         continue
                #     else:
                #         MegaAgent.labelTravelsAsExposed(mega_agent_name,
                #                                         date, 
                #                                         time_step,
                #                                         all_facilities_objects,
                #                                         mega_agent.UrbanoAgents,
                #                                         simulation_period_basic_info,
                #                                         mega_agent.dynamic_time_spent_dic,
                #                                         mega_agent.MegaAgentState.S_travelers,
                #                                         mega_agent.MegaAgentState.E_travelers_list,
                #                                         mega_agent.MegaAgentState.V_travelers)
 

            # daily update 
            for mega_agent_name, mega_agent in all_mega_agents.items():
                if mega_agent.MegaAgentPopulation == 0:
                    continue
                if mega_agent.if_initialized == True and date == dates[0]:
                    continue
                                    
                (mega_agent.new_Is_count, mega_agent.new_Ia_count) = MegaAgent.MegaAgent_daily_update(date, 
                                                                                                      parameters,
                                                                                                      mega_agent.MegaAgentState.S_travelers,
                                                                                                      mega_agent.MegaAgentState.E_travelers_list,
                                                                                                      mega_agent.MegaAgentState.Is_travelers_list,
                                                                                                      mega_agent.MegaAgentState.Ia_travelers_list,
                                                                                                      mega_agent.MegaAgentState.Is,
                                                                                                      mega_agent.MegaAgentState.Ia,
                                                                                                      mega_agent.MegaAgentState.R_travelers,
                                                                                                      mega_agent.MegaAgentState.Q_travelers,  
                                                                                                      mega_agent.MegaAgentState.Qe_travelers,
                                                                                                      mega_agent.MegaAgentState.Qa_travelers,
                                                                                                      mega_agent.MegaAgentState.Qs_travelers) 
                mega_agent.record_daily_stat(date, stat)
        reset_facilities_and_MegaAgents(simulation_period_basic_info)


        stat.plot_aggregated_progression(parameters)
        stat.plot_MegaAgents_progressions()
