from datetime import datetime, timedelta
from typing import List
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from parameters import Parameters
from preprocessing import SimulationPeriodBasicInfo
from seirstat import Statistics
from facility import Facility
from megaagent import MegaAgent


class Simulation:
    def __init__(self,                  
                 dates: List[datetime],
                 sim_basic_info: SimulationPeriodBasicInfo, 
                 parameters: Parameters, 
                 facility_names):
        self.dates = dates
        self.TimeSteps = [i for i in range(1, 7)]
        self.parameters = parameters
        self.TotalOperationCount = self.parameters.Iteration * len(self.dates)
        self.sim_basic_info = sim_basic_info
        self.FacilityNames = facility_names

        # Create all Facility objects
        self.all_facilities_objects = {}
        for f_name in self.FacilityNames:
            facility_object = Facility(f_name, self.sim_basic_info)
            self.all_facilities_objects[f_name] = facility_object

        # Create all MegaAgents
        self.all_mega_agents = {}
        for MegaAgentName in self.sim_basic_info.mega_agent_Urbano_mapping.keys():
            travelers_in_mega_agent = self.sim_basic_info.MegaAgent_Travelers[MegaAgentName]        
            mega_agent = MegaAgent(MegaAgentName, 
                                   travelers_in_mega_agent,
                                   self.sim_basic_info)
            self.all_mega_agents[MegaAgentName] = mega_agent
        
        # initialize Statistics
        self.stat = Statistics(self.all_mega_agents, self.dates)


    def run(self, if_plot_seperate: bool=False) -> None:
        """
        Specify the initial exposed count across the city.
        """
        print('Running', self.parameters.Iteration, 'repetitions.')
        print('Initial exposed:', self.parameters.initial_exposed_count)
        comp_time = (self.dates[-1] - self.dates[0]) + timedelta(days=1)
        print('Simulation days:', comp_time)

        # Setup the tqdm progress bar
        with tqdm(total=self.TotalOperationCount, desc="Overall Progress") as pbar:
            for _ in range(self.parameters.Iteration):
                initial_exposure = set(np.random.choice(list(self.sim_basic_info.AllTravelers), 
                                                    replace = False, 
                                                    size = self.parameters.initial_exposed_count))
                
                for mega_agent_name, mega_agent in self.all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue
                    
                    initial_exposed_travelers_in_mega_agent = list(mega_agent.MegaAgentState.S_set & initial_exposure)
                    if len(initial_exposed_travelers_in_mega_agent)==0:
                        continue
                    # self.initial_exposed(mega_agent, initial_exposed_travelers_in_mega_agent)
                    self.initial_infect(mega_agent, initial_exposed_travelers_in_mega_agent)
                   
                #self.run_one_iteration(pbar)
                self.run_one_iteration(pbar)
            
        self.stat.plot_aggregated_progression(self.parameters,
                                              if_plot_seperate=if_plot_seperate)
        self.stat.plot_MegaAgents_progressions() 


    def initial_infect(self, 
                       starter_ma: MegaAgent,
                       initial_infected_travelers: List[int]) -> None:
        first_date = self.dates[0]
        UrbanoAgents = self.sim_basic_info.mega_agent_Urbano_mapping[(starter_ma.MegaAgentName[0], starter_ma.MegaAgentName[1])]
        starter_ma.dynamic_time_spent_dic = MegaAgent.get_dynamic_time_spent(first_date,
                                                                             UrbanoAgents,
                                                                             self.sim_basic_info)

        MegaAgent.initialize_MegaAgent(first_date, 
                                        self.parameters,
                                        starter_ma.MegaAgentState.S_set,
                                        starter_ma.MegaAgentState.Is_dict,
                                        starter_ma.MegaAgentState.Ia_dict,
                                        starter_ma.MegaAgentState.Is_set,
                                        starter_ma.MegaAgentState.Ia_set,
                                        starter_ma.new_Is_count,
                                        starter_ma.new_Ia_count,
                                        starter_ma.MegaAgentState.R_dict,
                                        starter_ma.MegaAgentState.Q_dict,
                                        starter_ma.MegaAgentState.Qe_dict,
                                        starter_ma.MegaAgentState.Qa_dict,
                                        starter_ma.MegaAgentState.Qs_dict,
                                        initial_infected_travelers)
        starter_ma.record_daily_stat(first_date, self.stat)   
        self.parameters.conduct_testing.test(starter_ma, 
                                             first_date, 
                                             self.all_facilities_objects, 
                                             self.sim_basic_info)

        starter_ma.if_initialized = True
    
    def run_one_iteration(self, 
                          progess_bar) -> None:
        for date in self.dates:
            # On each day, at first, we need to calculate dynamic time spent at facility type
            for mega_agent_name, mega_agent in self.all_mega_agents.items():
                if mega_agent.if_initialized == True and date == self.dates[0]:
                    continue 
                UrbanoAgents = self.sim_basic_info.mega_agent_Urbano_mapping[(mega_agent.MegaAgentName[0], mega_agent.MegaAgentName[1])]
                mega_agent.dynamic_time_spent_dic = MegaAgent.get_dynamic_time_spent(date,
                                                                                     UrbanoAgents,
                                                                                     self.sim_basic_info)

            for time_step in self.TimeSteps:
                for f_name, f in self.all_facilities_objects.items():
                    f.FacilityHazard = Facility.computeFacilityHazard(f_name,
                                                                      f.FaciltyIndex,
                                                                      date,
                                                                      time_step,
                                                                      self.all_mega_agents, 
                                                                      self.parameters, 
                                                                      self.sim_basic_info,
                                                                      f.DateTimeStepAllVisitors)

                for mega_agent_name, mega_agent in self.all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue
                        
                    if mega_agent.if_initialized == True and date == self.dates[0] and time_step == 1:
                        continue
                    else:
                        MegaAgent.labelTravelsAsExposed(mega_agent_name,
                                                        date, 
                                                        time_step,
                                                        self.parameters,
                                                        self.all_facilities_objects,
                                                        self.sim_basic_info,
                                                        mega_agent.dynamic_time_spent_dic,
                                                        mega_agent.MegaAgentState.S_set,
                                                        mega_agent.MegaAgentState.E_dict,
                                                        mega_agent.MegaAgentState.V_dict)

            # daily update        
            for mega_agent_name, mega_agent in self.all_mega_agents.items():
                if mega_agent.MegaAgentPopulation == 0:
                    continue
                if mega_agent.if_initialized == True and date == self.dates[0]:
                    continue

                (mega_agent.new_Is_count, mega_agent.new_Ia_count) = MegaAgent.MegaAgent_daily_update(date, 
                                                                                                      self.parameters,
                                                                                                      mega_agent.MegaAgentState.S_set,
                                                                                                      mega_agent.MegaAgentState.E_dict,
                                                                                                      mega_agent.MegaAgentState.Is_dict,
                                                                                                      mega_agent.MegaAgentState.Ia_dict,
                                                                                                      mega_agent.MegaAgentState.Is_set,
                                                                                                      mega_agent.MegaAgentState.Ia_set,
                                                                                                      mega_agent.MegaAgentState.R_dict,
                                                                                                      mega_agent.MegaAgentState.Q_dict,  
                                                                                                      mega_agent.MegaAgentState.Qe_dict,
                                                                                                      mega_agent.MegaAgentState.Qa_dict,
                                                                                                      mega_agent.MegaAgentState.Qs_dict) 
                mega_agent.record_daily_stat(date, self.stat)
                self.parameters.conduct_testing.test(mega_agent,
                                                     date,
                                                     self.all_facilities_objects,
                                                     self.sim_basic_info)
            
            # Update the progress bar after each date is processed
            progess_bar.update(1)
        self.reset_facilities_and_MegaAgents()      

    def reset_facilities_and_MegaAgents(self) -> None:
        for facility_name, facility in self.all_facilities_objects.items():
            facility.reset_facility()
        for mega_agent_name, mega_agent in self.all_mega_agents.items():
            mega_agent.reset_MegaAgent()
