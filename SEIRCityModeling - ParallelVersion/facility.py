from datetime import datetime


class Facility:
    def __init__(self, 
                 facility_name: tuple, 
                 whole_simulation_period_bi):
        self.FacilityName = facility_name
        self.FacilityType = self.FacilityName[0]
        self.BlockID = self.FacilityName[1]

        self.FaciltyIndex = whole_simulation_period_bi.FacilityToIndex[self.FacilityName]
        whole_simulation_period_bi.IndexToFacilityObject[self.FaciltyIndex] = self
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
                                                        for traveler in whole_simulation_period_bi.urbano_agents_travelers_mapping[urbano_agent]
                                                        ]  
                                                for time_step, urano_agents in time_step_urbano_agents.items()
                                              } 
                                        for date, time_step_urbano_agents in whole_simulation_period_bi.facility_date_time_step_urbano_agent_dic[facility_name].items()
                                        }
        self.Date_TimeStep_Susceptibles = {
                                            date: {
                                                    time_step: set(traveler for traveler in travelers) 
                                                    for time_step, travelers in time_step_travelers.items()
                                                  } 
                                            for date, time_step_travelers in self.DateTimeStepAllVisitors.items()
                                          }
        self.FacilityHazard = 0

    def reset_facility(self, whole_simulation_period_bi):
        self.FacilityHazard = 0
        self.DateTimeStepAllVisitors = {
                                        date: {
                                                time_step:[
                                                        traveler 
                                                        for urbano_agent in urano_agents
                                                        for traveler in whole_simulation_period_bi.urbano_agents_travelers_mapping[urbano_agent]
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


    @staticmethod
    def computeFacilityHazard(FacilityName,
                              FaciltyIndex: int,
                              date: datetime,
                              time_step: int, 
                              mega_agents: dict, 
                              parameters, 
                              whole_simulation_period_bi,
                              DateTimeStepAllVisitors) -> float:
        """
        Calculate hazard of the facility. Should be used in time-step basis.
        """
        visitors_on_the_date_and_time_step = set(DateTimeStepAllVisitors[date][time_step])
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
            time_spent = whole_simulation_period_bi.AllDatesFacilityTimeUse[date][FacilityName]
            # Find the faicitliy vist probability
            # visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][self.FaciltyIndex] * (1/6) # consider 6 time steps
            visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][FaciltyIndex] * whole_simulation_period_bi.time_span_of_each_time_step_dic[time_step]
            # Compute hazard brought by the mega agent
            Facility_Hazard += sympt_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E
            Facility_Hazard += asym_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E * parameters.asym_hazard_multiplier
        return Facility_Hazard





    @staticmethod
    def create_facilityDateTimeStepAllVisitors(whole_simulation_period_bi):
        dic = {}
        for facility_name in whole_simulation_period_bi.AllFacilityNames:
            DateTimeStepAllVisitors = {
                                        date: {
                                                time_step:[
                                                        traveler 
                                                        for urbano_agent in urano_agents
                                                        for traveler in whole_simulation_period_bi.urbano_agents_travelers_mapping[urbano_agent]
                                                        ]  
                                                for time_step, urano_agents in time_step_urbano_agents.items()
                                                } 
                                        for date, time_step_urbano_agents in whole_simulation_period_bi.facility_date_time_step_urbano_agent_dic[facility_name].items()
                                        }
            dic[facility_name] = DateTimeStepAllVisitors
        return dic
    
    @staticmethod
    def create_facilityDate_TimeStep_Susceptibles(facilityDateTimeStepAllVisitors):
        dic = {}
        AllFacilityNames = list(facilityDateTimeStepAllVisitors.keys())
        for facility_name in AllFacilityNames:
            DateTimeStepAllVisitors = facilityDateTimeStepAllVisitors[facility_name]
            Date_TimeStep_Susceptibles = {
                                            date: {
                                                    time_step: set(traveler for traveler in travelers) 
                                                    for time_step, travelers in time_step_travelers.items()
                                                  } 
                                            for date, time_step_travelers in DateTimeStepAllVisitors.items()
                                          }
            dic[facility_name] = Date_TimeStep_Susceptibles
        return dic

    @staticmethod
    def initialize_facility_hazards(whole_simulation_period_bi):
        facility_hazards = {}
        for facility_name in whole_simulation_period_bi.AllFacilityNames: 
            facility_hazards[facility_name] = 0
        return facility_hazards

    @staticmethod
    def reset_facilities(whole_simulation_period_bi):
        facilityDateTimeStepAllVisitors = Facility.create_facilityDateTimeStepAllVisitors(whole_simulation_period_bi)
        facilityDate_TimeStep_Susceptibles = Facility.create_facilityDate_TimeStep_Susceptibles(facilityDateTimeStepAllVisitors)
        facility_hazards = Facility.initialize_facility_hazards(whole_simulation_period_bi)
        return (facilityDateTimeStepAllVisitors, 
                facilityDate_TimeStep_Susceptibles, 
                facility_hazards)

    @staticmethod
    def computeFacilityHazard(facility_name: tuple,
                              date: datetime,
                              time_step: int, 
                              parameters,
                              facilityDateTimeStepAllVisitors: dict,
                              whole_simulation_period_bi):
        """
        Calculate hazard of the facility. Should be used in time-step basis.
        """
        visitors_on_the_date_and_time_step = set(facilityDateTimeStepAllVisitors[facility_name][date][time_step])
        FaciltyIndex = whole_simulation_period_bi.FacilityToIndex[facility_name]
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
            time_spent = whole_simulation_period_bi.AllDatesFacilityTimeUse[date][facility_name]
            # Find the faicitliy vist probability
            # visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][self.FaciltyIndex] * (1/6) # consider 6 time steps
            visit_prob = mega_agent.FacilityStationaryDistribution[time_step][0][FaciltyIndex] * whole_simulation_period_bi.time_span_of_each_time_step_dic[time_step]
            # Compute hazard brought by the mega agent
            Facility_Hazard += sympt_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E
            Facility_Hazard += asym_visitors_num * time_spent * visit_prob * parameters.infection_duration.rate_from_S_to_E * parameters.asym_hazard_multiplier