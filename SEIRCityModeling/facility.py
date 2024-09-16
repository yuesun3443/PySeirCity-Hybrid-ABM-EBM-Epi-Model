from typing import List, Dict, Any
from datetime import datetime
import concurrent.futures

class Facility:
    def __init__(self, 
                 facility_name: tuple, 
                 whole_simulation_period_bi):
        self.FacilityName = facility_name
        self.FacilityType = self.FacilityName[0]
        self.BlockID = self.FacilityName[1]

        self.whole_simulation_period_bi = whole_simulation_period_bi
        self.FaciltyIndex = whole_simulation_period_bi.FacilityToIndex[self.FacilityName]
        self.whole_simulation_period_bi.IndexToFacilityObject[self.FaciltyIndex] = self
        # the dict is in the format of: {date: {time_step: [travelers]}}
        # Date_TimeStep_Susceptibles is a subset of DateTimeStepAllVisitors
        # DateTimeStepAllVisitors includes all visitors with any epi states, Date_TimeStep_Susceptibles only has Susceptible visitors
        # self.DateTimeStepAllVisitors = {date: {time_step: set(visitor for visitor in visitors) for time_step, visitors in time_step_visitors.items()} for date, time_step_visitors in whole_simulation_period_bi.FacilityDateTimeStepTravelers[self.FacilityName].items()}
        # self.Date_TimeStep_Susceptibles = {date: {time_step: set(traveler for traveler in travelers) for time_step, travelers in time_step_travelers.items()} for date, time_step_travelers in self.DateTimeStepAllVisitors.items()}

        self.DateTimeStepAllVisitors = {
                                        date: {
                                                time_step:[
                                                        traveler 
                                                        for urbano_agent in urano_agents
                                                        for traveler in self.whole_simulation_period_bi.urbano_agents_travelers_mapping[urbano_agent]
                                                        ]  
                                                for time_step, urano_agents in time_step_urbano_agents.items()
                                              } 
                                        for date, time_step_urbano_agents in whole_simulation_period_bi.facility_date_time_step_urbano_agent_dic[self.FacilityName].items()
                                        }
        self.Date_TimeStep_Susceptibles = {
                                            date: {
                                                    time_step: set(traveler for traveler in travelers) 
                                                    for time_step, travelers in time_step_travelers.items()
                                                  } 
                                            for date, time_step_travelers in self.DateTimeStepAllVisitors.items()
                                          }

        self.FacilityHazard = 0


    def reset_facility(self):
        self.FacilityHazard = 0
        self.DateTimeStepAllVisitors = {
                                        date: {
                                                time_step:[
                                                        traveler 
                                                        for urbano_agent in urano_agents
                                                        for traveler in self.whole_simulation_period_bi.urbano_agents_travelers_mapping[urbano_agent]
                                                        ]  
                                                for time_step, urano_agents in time_step_urbano_agents.items()
                                              } 
                                        for date, time_step_urbano_agents in self.whole_simulation_period_bi.facility_date_time_step_urbano_agent_dic[self.FacilityName].items()
                                        }        
        self.Date_TimeStep_Susceptibles = {
                                            date: {
                                                    time_step: set(traveler for traveler in travelers) 
                                                    for time_step, travelers in time_step_travelers.items()
                                                  } 
                                            for date, time_step_travelers in self.DateTimeStepAllVisitors.items()
                                          }        

    
    def computeFacilityHazard(self, 
                              date: datetime,
                              time_step: int, 
                              mega_agents: dict, 
                              parameters, 
                              whole_simulation_period_bi) -> float:
        """
        Calculate hazard of the facility. Should be used in time-step basis.
        """
        visitors_on_the_date_and_time_step = set(self.DateTimeStepAllVisitors[date][time_step])
        Facility_Hazard = 0
        for mega_agent_name, mega_agent in mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue  # Skip this iteration if population is zero
            # if len(mega_agent.MegaAgentState.Is) == 0 and len(mega_agent.MegaAgentState.Ia) == 0:
            #     continue
            if not mega_agent.MegaAgentState.Is and not mega_agent.MegaAgentState.Ia:
                continue

            # Find the intersection
            sympt_visitors_num = 0
            asym_visitors_num = 0
            if mega_agent.MegaAgentState.Is:
                # if not mega_agent.MegaAgentState.Is.isdisjoint(visitors_on_the_date_and_time_step):
                #     sympt_intersection = mega_agent.MegaAgentState.Is.intersection(visitors_on_the_date_and_time_step)
                #     sympt_visitors_num = len(sympt_intersection)
                sympt_intersection = mega_agent.MegaAgentState.Is & visitors_on_the_date_and_time_step
                sympt_visitors_num = len(sympt_intersection)
            if mega_agent.MegaAgentState.Ia:
                # if not mega_agent.MegaAgentState.Ia.isdisjoint(visitors_on_the_date_and_time_step):
                #     asym_intersection = mega_agent.MegaAgentState.Ia.intersection(visitors_on_the_date_and_time_step)
                #     asym_visitors_num = len(asym_intersection)
                asym_intersection = mega_agent.MegaAgentState.Ia & visitors_on_the_date_and_time_step
                asym_visitors_num = len(asym_intersection)
            if sympt_visitors_num == 0 and asym_visitors_num == 0:
                continue

            # Find time spent
            time_spent = whole_simulation_period_bi.AllDatesFacilityTimeUse[date][self.FacilityName]
            # Find the faicitliy vist probability
            # visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][self.FaciltyIndex] * (1/6) # consider 6 time steps
            visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][self.FaciltyIndex] * whole_simulation_period_bi.time_span_of_each_time_step_dic[time_step]
            # Compute hazard brought by the mega agent
            Facility_Hazard += sympt_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E
            Facility_Hazard += asym_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E * parameters.asym_hazard_multiplier
        self.FacilityHazard = Facility_Hazard#####################
        return Facility_Hazard


    @staticmethod
    def computeFacilityHazard0(facility, 
                              date: datetime,
                              time_step: int, 
                              mega_agents: dict, 
                              parameters, 
                              whole_simulation_period_bi) -> None:
        """
        Calculate hazard of the facility. Should be used in time-step basis.
        """
        visitors_on_the_date_and_time_step = set(facility.DateTimeStepAllVisitors[date][time_step])
        Facility_Hazard = 0
        for mega_agent_name, mega_agent in mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue  # Skip this iteration if population is zero
            # if len(mega_agent.MegaAgentState.Is) == 0 and len(mega_agent.MegaAgentState.Ia) == 0:
            #     continue
            if not mega_agent.MegaAgentState.Is and not mega_agent.MegaAgentState.Ia:
                continue

            # Find the intersection
            sympt_visitors_num = 0
            asym_visitors_num = 0
            if mega_agent.MegaAgentState.Is:
                # if not mega_agent.MegaAgentState.Is.isdisjoint(visitors_on_the_date_and_time_step):
                #     sympt_intersection = mega_agent.MegaAgentState.Is.intersection(visitors_on_the_date_and_time_step)
                #     sympt_visitors_num = len(sympt_intersection)
                sympt_intersection = mega_agent.MegaAgentState.Is & visitors_on_the_date_and_time_step
                sympt_visitors_num = len(sympt_intersection)
            if mega_agent.MegaAgentState.Ia:
                # if not mega_agent.MegaAgentState.Ia.isdisjoint(visitors_on_the_date_and_time_step):
                #     asym_intersection = mega_agent.MegaAgentState.Ia.intersection(visitors_on_the_date_and_time_step)
                #     asym_visitors_num = len(asym_intersection)
                asym_intersection = mega_agent.MegaAgentState.Ia & visitors_on_the_date_and_time_step
                asym_visitors_num = len(asym_intersection)
            if sympt_visitors_num == 0 and asym_visitors_num == 0:
                continue

            # Find time spent
            time_spent = whole_simulation_period_bi.AllDatesFacilityTimeUse[date][facility.FacilityName]
            # Find the faicitliy vist probability
            visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][facility.FaciltyIndex] * whole_simulation_period_bi.time_span_of_each_time_step_dic[time_step]
            # Compute hazard brought by the mega agent
            Facility_Hazard += sympt_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E
            Facility_Hazard += asym_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E * parameters.asym_hazard_multiplier
        return {facility.facility_name, Facility_Hazard}


    @staticmethod
    def compute_mega_agent_hazard(args: Dict[str, Any]) -> float:
        (
            mega_agent,
            visitors_on_the_date_and_time_step,
            date,
            time_step,
            facility_name,
            facility_index,
            time_spent,
            time_span_of_the_time_step,
            infection_rate,
            asym_hazard_multiplier
        ) = args.values()

        if mega_agent.MegaAgentPopulation == 0:
            return 0.0
        if len(mega_agent.MegaAgentState.Is) == 0 and len(mega_agent.MegaAgentState.Ia) == 0:
            return 0.0

        sympt_visitors_num = 0
        asym_visitors_num = 0

        if mega_agent.MegaAgentState.Is:
            sympt_intersection = mega_agent.MegaAgentState.Is & visitors_on_the_date_and_time_step
            sympt_visitors_num = len(sympt_intersection)
        if mega_agent.MegaAgentState.Ia:
            asym_intersection = mega_agent.MegaAgentState.Ia & visitors_on_the_date_and_time_step
            asym_visitors_num = len(asym_intersection)
        if sympt_visitors_num == 0 and asym_visitors_num == 0:
            return 0.0

        visit_prob = (
            mega_agent.FacilityStationaryDistribution[time_step][0][facility_index]
            * time_span_of_the_time_step
        )

        hazard = (
            sympt_visitors_num * time_spent * visit_prob * infection_rate
            + asym_visitors_num * time_spent * visit_prob * infection_rate * asym_hazard_multiplier
        )
        return hazard


    # not being used
    def computeFacilityHazardParallel(
        self,
        date: datetime,
        time_step: int,
        mega_agents: dict, 
        parameters,
        whole_simulation_period_bi,
        max_processor_num=4
    ) -> float:
        visitors_on_the_date_and_time_step = set(self.DateTimeStepAllVisitors[date][time_step])

        time_spent = whole_simulation_period_bi.AllDatesFacilityTimeUse[date][self.FacilityName]
        time_span_of_the_time_step = self.whole_simulation_period_bi.time_span_of_each_time_step_dic[time_step]
        infection_rate = parameters.infection_duration.rate_from_S_to_E
        asym_hazard_multiplier = parameters.asym_hazard_multiplier

        # Prepare arguments for each mega_agent
        args_list = []
        for mega_agent_name, mega_agent in mega_agents.items():
            args = {
                "mega_agent": mega_agent,
                "visitors_on_the_date_and_time_step": visitors_on_the_date_and_time_step,
                "date": date,
                "time_step": time_step,
                "facility_name": self.FacilityName,
                "facility_index": self.FaciltyIndex,
                "time_spent": time_spent,
                "time_span_of_the_time_step": time_span_of_the_time_step,
                "infection_rate": infection_rate,
                "asym_hazard_multiplier": asym_hazard_multiplier,
            }
            args_list.append(args)

        # Use ProcessPoolExecutor for parallel computation
        Facility_Hazard = 0.0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processor_num) as executor:
            results = executor.map(Facility.compute_mega_agent_hazard, args_list)
        Facility_Hazard = sum(results)
        return Facility_Hazard

