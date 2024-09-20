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
                    
                    initial_exposed_travelers_in_mega_agent = mega_agent.MegaAgentState.S_travelers & initial_exposure
                    if len(initial_exposed_travelers_in_mega_agent)==0:
                        continue
                    self.initial_exposed(mega_agent, initial_exposed_travelers_in_mega_agent)
                   
                self.run_one_iteration(pbar)
                #####################################################parallel computing
                # Simulation.run_one_iteration_parallel(pbar, 
                #                                         self.dates,
                #                                         self.all_mega_agents,
                #                                         self.all_facilities_objects,
                #                                         self.stat,
                #                                         self.sim_basic_info,
                #                                         self.TimeSteps,
                #                                         self.parameters,
                #                                         max_processor_num=4) 
                # self.reset_facilities_and_MegaAgents()
                #####################################################
        
        self.stat.plot_aggregated_progression(self.parameters.Iteration, if_plot_seperate=if_plot_seperate)
        self.stat.plot_MegaAgents_progressions()


    def initial_exposed(self, 
                        starter_ma: MegaAgent,
                        initial_exposed_travelers: List[int]) -> None:
        """
        Being used to kick off the simulation.
        """
        first_date = self.dates[0]
        starter_ma.dynamic_time_spent_dic = starter_ma.get_dynamic_time_spent(first_date)
        starter_ma.labelTravelsAsExposed(first_date, 
                                         1,
                                         self.all_facilities_objects, 
                                         initial_infected=initial_exposed_travelers)
        starter_ma.if_initialized = True


    def run_one_iteration(self, 
                          progess_bar) -> None:
        for date in self.dates:
            # On each day, at first, we need to calculate dynamic time spent at facility type
            for mega_agent_name, mega_agent in self.all_mega_agents.items():
                if mega_agent.if_initialized == True and date == self.dates[0]:
                    continue 
                mega_agent.dynamic_time_spent_dic = mega_agent.get_dynamic_time_spent(date)

            for time_step in self.TimeSteps:
                for f_name, f in self.all_facilities_objects.items():
                    f.FacilityHazard = f.computeFacilityHazard(date,
                                                               time_step,
                                                               self.all_mega_agents, 
                                                               self.parameters, 
                                                               self.sim_basic_info)

                for mega_agent_name, mega_agent in self.all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue
                        
                    if mega_agent.if_initialized == True and date == self.dates[0] and time_step == 1:
                        continue
                    else:
                        mega_agent.labelTravelsAsExposed(date,
                                                         time_step,
                                                         self.all_facilities_objects)

            # daily update        
            for mega_agent_name, mega_agent in self.all_mega_agents.items():
                if mega_agent.MegaAgentPopulation == 0:
                    continue
                mega_agent.MegaAgent_daily_update(date, self.parameters) 
                mega_agent.record_daily_stat(date, self.stat)
                
                self.parameters.conduct_testing.test(mega_agent, 
                                                    date, 
                                                    self.all_facilities_objects, 
                                                    self.parameters.contact_trace_date_length, 
                                                    self.parameters.quarantine_length,
                                                    self.parameters.quarantine_prob)

                self.parameters.contact_tracing.trace(mega_agent, 
                                                      date, 
                                                      mega_agent.contact_trace_roster,
                                                      self.all_facilities_objects)
            
            # Update the progress bar after each date is processed
            progess_bar.update(1)
                    
        self.reset_facilities_and_MegaAgents()        
            

    def reset_facilities_and_MegaAgents(self) -> None:
        for facility_name, facility in self.all_facilities_objects.items():
            facility.reset_facility()
        for mega_agent_name, mega_agent in self.all_mega_agents.items():
            mega_agent.reset_MegaAgent()


    @staticmethod
    def run_one_iteration_parallel(progess_bar, 
                                   dates,
                                   all_mega_agents,
                                   all_facilities_objects,
                                   stat,
                                   sim_basic_info,
                                   TimeSteps,
                                   parameters,
                                   max_processor_num=4) -> None:
        for date in dates:
            # On each day, at first, we need to calculate dynamic time spent for each travelers
            for mega_agent_name, mega_agent in all_mega_agents.items():
                if mega_agent.if_initialized == True and date == dates[0]:
                    continue 
                mega_agent.dynamic_time_spent_dic = mega_agent.get_dynamic_time_spent(date)

            for time_step in TimeSteps:
                # for f_name, f in self.all_facilities_objects.items():
                #     f.FacilityHazard = f.computeFacilityHazard(date,
                #                                                time_step,
                #                                                self.all_mega_agents, 
                #                                                self.parameters, 
                #                                                self.sim_basic_info)
                with ProcessPoolExecutor(max_workers=max_processor_num) as executor:
                    futures = [executor.submit(Facility.computeFacilityHazard, f, date, time_step, all_mega_agents, parameters, sim_basic_info) for f in all_facilities_objects.values()] 
                    for future in futures:
                        result = future.result()
                        for fn,fhazard in result.items():
                            all_facilities_objects[fn].FacilityHazard = fhazard

                for mega_agent_name, mega_agent in all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue
                        
                    if mega_agent.if_initialized == True and date == dates[0] and time_step == 1:
                        continue
                    else:
                        mega_agent.labelTravelsAsExposed(date,
                                                         time_step,
                                                         all_facilities_objects)

            # daily update        
            for mega_agent_name, mega_agent in all_mega_agents.items():
                if mega_agent.MegaAgentPopulation == 0:
                    continue
                mega_agent.MegaAgent_daily_update(date, parameters) 
                mega_agent.record_daily_stat(date, stat)
                
                parameters.conduct_testing.test(mega_agent, 
                                                    date, 
                                                    all_facilities_objects, 
                                                    parameters.contact_trace_date_length, 
                                                    parameters.quarantine_length,
                                                    parameters.quarantine_prob)

                parameters.contact_tracing.trace(mega_agent, 
                                                      date, 
                                                      mega_agent.contact_trace_roster,
                                                      all_facilities_objects)
            
            # Update the progress bar after each date is processed
            progess_bar.update(1)
                    
        # self.reset_facilities_and_MegaAgents()        