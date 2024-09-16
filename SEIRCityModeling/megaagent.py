from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from typing import List



class State:
    def __init__(self):
        self.S_travelers = set()
        self.E_travelers_list = []

        self.Ia_travelers_list = []
        self.Is_travelers_list = []
        self.Ia = set()
        self.Is = set()

        self.R_travelers = set()
        
        self.V_travelers = {}############################

        self.Q_travelers = {}
        self.Qe_travelers = {}
        self.Qs_travelers = {}
        self.Qa_travelers = {}




class MegaAgent:
    """
    A group of travelers of the same type at a certain block as one "Mega Agent." 
    One MegaAgent is a unqie pair of home_block and traveler_type, i.e. (home_block, traveler_type).
    """
    def __init__(self,
                 MegaAgentName: tuple,
                 residents: set,
                 whole_simulation_period_bi):
        self.MegaAgentName = MegaAgentName
        self.BlockID = self.MegaAgentName[0]
        self.MegaAgentType = self.MegaAgentName[1]

        self.Residents = set(resident for resident in residents)
        self.UrbanoAgents = whole_simulation_period_bi.mega_agent_Urbano_mapping[(self.BlockID, self.MegaAgentType)]

        self.MegaAgentPopulation = len(self.Residents)
        self.SimulationPeriodBasicInfo = whole_simulation_period_bi
        self.FacilityStationaryDistribution = self.SimulationPeriodBasicInfo.StationaryDistributions[self.MegaAgentName]

        # One MegaAgent contains state info for the sake of transimission dynamics
        self.MegaAgentState = State()
        self.dynamic_time_spent_dic = dict()
        # initialize S
        self.MegaAgentState.S_travelers = set([traveler for traveler in self.Residents])
        self.Risk = 0
        self.if_initialized = False
        self.contact_trace_roster = defaultdict(set)
    

    def reset_MegaAgent(self) -> None:
        self.MegaAgentState = State()
        self.MegaAgentState.S_travelers = set([traveler for traveler in self.Residents])
        self.dynamic_time_spent_dic = dict()
        self.Risk = 0
        self.if_initialized = False  
        self.contact_trace_roster = defaultdict(set)


    def get_dynamic_time_spent(self, date:datetime):  
        travelers_time_spent_at_the_facilityType_dict = {}
        for urbano_agent in self.UrbanoAgents:  
            visit_facilities = self.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][date]  
            time_spent_dict = dict()
            for facility_name, visit_time_step in visit_facilities:
                facility_type = facility_name[0]
                daily_time_spent_in_facilityType = self.SimulationPeriodBasicInfo.AllDatesFacilityTimeUse[date][facility_name]
                time_spent_dict[facility_type] = dict() 
                for time_step, ft_visit_count in self.SimulationPeriodBasicInfo.date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type].items():
                    time_spent_dict[facility_type][time_step] = self.SimulationPeriodBasicInfo.time_span_of_each_time_step_dic[time_step] * ft_visit_count / self.SimulationPeriodBasicInfo.date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent][time_step]

                # unitize time_spent_dict[facility_type]
                time_spent_ratio_sum = sum(time_spent_dict[facility_type].values())
                time_spent_dict[facility_type] = {time_step: daily_time_spent_in_facilityType * time_spent_ratio/time_spent_ratio_sum for time_step, time_spent_ratio in time_spent_dict[facility_type].items()}

            for traveler in self.SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]:
                travelers_time_spent_at_the_facilityType_dict[traveler] = time_spent_dict
        return travelers_time_spent_at_the_facilityType_dict


    # orginal version, not being used
    def calculate_MegaAgent_risk0(self, 
                                date:datetime,
                                current_time_step: int,
                                all_facilities_objects: dict):
        risk = 0
        s_traveler_risk_dict = dict()
        
        # for facility_name, facility in all_facilities_objects.items():
        for facility in all_facilities_objects.values():
            facility_hazard = facility.FacilityHazard
            # Find time spent
            time_spent = self.SimulationPeriodBasicInfo.AllDatesFacilityTimeUse[date][facility.FacilityName]
            # Find the faicitliy vist probability
            visit_prob = self.FacilityStationaryDistribution[current_time_step][0][facility.FaciltyIndex]
            risk += facility_hazard * time_spent * visit_prob        

        # get visited facilities by the susceptible travelers of the maga_agent at the current time step
        s_traveler_visit_facilities = dict()
        for s_traveler in self.MegaAgentState.S_travelers:
            urbano_agent = self.SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[s_traveler]
            visit_facilities = self.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][date]
            for facility_name, visit_time_step in visit_facilities:
                if visit_time_step != current_time_step:
                    continue

                facility_object = all_facilities_objects[facility_name]
                if s_traveler in facility_object.Date_TimeStep_Susceptibles[date][current_time_step]:
                    if s_traveler not in s_traveler_visit_facilities:
                        s_traveler_visit_facilities[s_traveler] = set()
                    s_traveler_visit_facilities[s_traveler].add(facility_object)

        for s_traveler, visit_facilities in s_traveler_visit_facilities.items():
            s_traveler_risk = 0

            for v_f in visit_facilities:
                facility_hazard = v_f.FacilityHazard
                facility_type = v_f.FacilityType
                # Find the faicitliy vist probability
                visit_prob = self.FacilityStationaryDistribution[current_time_step][0][v_f.FaciltyIndex]
                
                # get time spent at the facility type at the target time step
                time_spent = self.dynamic_time_spent_dic[s_traveler][facility_type][current_time_step]
                s_traveler_risk += facility_hazard * time_spent * visit_prob

            s_traveler_risk_dict[s_traveler] = s_traveler_risk
        return risk, s_traveler_risk_dict


    def calculate_MegaAgent_risk(self, 
                                date:datetime,
                                current_time_step: int,
                                all_facilities_objects: dict):
        risk = 0
        
        for facility in all_facilities_objects.values():
            facility_hazard = facility.FacilityHazard
            # Find daily time spent
            daily_time_spent = self.SimulationPeriodBasicInfo.AllDatesFacilityTimeUse[date][facility.FacilityName]
            # Find the faicitliy vist probability
            visit_prob = self.FacilityStationaryDistribution[current_time_step][0][facility.FaciltyIndex]
            risk += facility_hazard * daily_time_spent * visit_prob
        return risk


    def calculate_susceptible_travelers_risks(self, 
                                             date:datetime,
                                             current_time_step: int,
                                             all_facilities_objects: dict) -> dict:
        s_traveler_risk_dict = dict()
        s_traveler_visit_facilities = dict()
        for urbano_agent in self.UrbanoAgents:
            s_travelers = self.MegaAgentState.S_travelers & self.SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]
            if len(s_travelers)==0:
                continue

            visit_facilities = self.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][date]
            for facility_name, visit_time_step in visit_facilities:
                if visit_time_step != current_time_step:
                    continue
            
                facility_object = all_facilities_objects[facility_name]
                for s_traveler in s_travelers:
                    if s_traveler in facility_object.Date_TimeStep_Susceptibles[date][current_time_step]:
                        if s_traveler not in s_traveler_visit_facilities:
                            s_traveler_visit_facilities[s_traveler] = set()
                    s_traveler_visit_facilities[s_traveler].add(facility_object)

        for s_traveler, visit_facilities in s_traveler_visit_facilities.items():
            s_traveler_risk = 0

            for v_f in visit_facilities:
                facility_hazard = v_f.FacilityHazard
                facility_type = v_f.FacilityType
                # Find the faicitliy vist probability
                visit_prob = self.FacilityStationaryDistribution[current_time_step][0][v_f.FaciltyIndex]
                
                # get time spent at the facility type at the target time step
                time_spent = self.dynamic_time_spent_dic[s_traveler][facility_type][current_time_step]
                s_traveler_risk += facility_hazard * time_spent * visit_prob

            s_traveler_risk_dict[s_traveler] = s_traveler_risk
        return s_traveler_risk_dict


    def selectSusceptibleBasedOnTravelerRisk(self,
                                             delta_S: int, 
                                             s_traveler_risk_dict: dict) -> set:
        sampled_travelers = set()
        if len(self.MegaAgentState.S_travelers) == 0:
            return sampled_travelers
        
        # Check if all risks are zero
        s_travelers = list(self.MegaAgentState.S_travelers)
        if all(risk == 0 for risk in s_traveler_risk_dict.values()):
            sampled_travelers = set(np.random.choice(s_travelers, size=delta_S, replace=False))
        else:
            s_risk_weights = list(s_traveler_risk_dict.values())
            list_without_zeros = [r for r in s_risk_weights if r != 0]
            min_risk_weight = min(list_without_zeros)
            # if the risk_weight is 0, add a little value to it to prevent 0 value
            # it will be easier when using np.random.choice
            s_risk_weights = [min_risk_weight/10000 if risk == 0 else risk for risk in s_risk_weights]  
            # unitized weights
            risk_sum = sum(s_risk_weights)
            s_risk_weights = [risk/risk_sum for risk in s_risk_weights]

            if len(s_risk_weights) != len(s_travelers):
                raise RuntimeError("Two lists must be equal length.")

            if delta_S >= len(s_travelers):
                sampled_travelers = set(s_traveler for s_traveler in s_travelers)
            else:
                # Sample travelers based on weights
                sampled_travelers = np.random.choice(s_travelers, size=delta_S, replace=False, p=s_risk_weights)
        return sampled_travelers


    def labelTravelsAsExposed(self, 
                              date:datetime,
                              current_time_step: int,
                              all_facilities_objects: dict,
                              initial_infected:List[int]=None) -> None:
        """
        Decide how many travelers should be labeled as exposed within the MegaAgent.
        """
        if len(self.MegaAgentState.S_travelers) == 0:
            return None

        to_infect = set()
        if initial_infected is not None:
            to_infect.update(initial_infected)
        else:
            # used to decide how many travelers are exposed and removed from S.
            # self.Risk, s_traveler_risk_dict = self.calculate_MegaAgent_risk(date, 
            #                                                                 current_time_step, 
            #                                                                 all_facilities_objects)
            self.Risk = self.calculate_MegaAgent_risk(date, current_time_step, all_facilities_objects)
            s_traveler_risk_dict = self.calculate_susceptible_travelers_risks(date, current_time_step, all_facilities_objects)
        
            delta_S = len(self.MegaAgentState.S_travelers) * self.Risk
            delta_S = round(delta_S)

            if delta_S == 0:
                return None

            if delta_S >= len(self.MegaAgentState.S_travelers):
                to_infect.update(self.MegaAgentState.S_travelers)
            else:
                # to_infect = set(np.random.choice(list(self.MegaAgentState.S_travelers), size=delta_S, replace=False))
                to_infect = self.selectSusceptibleBasedOnTravelerRisk(delta_S, s_traveler_risk_dict)

        random_nums = set(np.random.uniform(0, 1, size=len(to_infect)))
        for traveler, if_infectious in zip(to_infect, random_nums):
            # # if a traveler is not vacinated or is vacinated but useless,
            # # then label this as one exposed instance
            # if traveler not in self.MegaAgentState.V_travelers or if_infectious > parameters.vaccine_benefit_self:
            #     self.MegaAgentState.S_travelers.remove(traveler)

            if traveler not in self.MegaAgentState.V_travelers:
                self.MegaAgentState.S_travelers.remove(traveler)
                
                # remove the infected traveler from the facility Susceptible record.
                # from the infected date on, this infected traveler will be removed from
                # Susceptible record of every visited facility.
                current_date_index = self.SimulationPeriodBasicInfo.Dates.index(date)
                urbano_agent = self.SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[traveler]
                for d in self.SimulationPeriodBasicInfo.Dates[current_date_index:]:
                    for (f_name, visit_time_step) in self.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][d]:
                        if d is date and visit_time_step < current_time_step:
                            continue
                        else:
                            all_facilities_objects[f_name].Date_TimeStep_Susceptibles[d][visit_time_step].remove(traveler)

                self.MegaAgentState.E_travelers_list.append(traveler)


    def transferTravelersFromEtoI(self, 
                                  parameters) -> None:
        E_len = len(self.MegaAgentState.E_travelers_list)
        # num_of_travelers_from_E_to_I = math.ceil(E_len * parameters.infection_duration.rate_from_E_to_I)
        num_of_travelers_from_E_to_I = round(E_len * parameters.infection_duration.rate_from_E_to_I)
        if E_len <= 0:
            return None

        travelers_from_E_to_I = []
        if E_len >= num_of_travelers_from_E_to_I:
            travelers_from_E_to_I = self.MegaAgentState.E_travelers_list[:num_of_travelers_from_E_to_I]
            del self.MegaAgentState.E_travelers_list[:num_of_travelers_from_E_to_I] # remove travelers who are transferred to I
        else:
            travelers_from_E_to_I = [traveler for traveler in self.MegaAgentState.E_travelers_list]
            self.MegaAgentState.E_travelers_list.clear() # all travelers are removed to I

        num_of_travelers_to_sym_I = round(num_of_travelers_from_E_to_I * (1 - parameters.infection_duration.asym_fraction))
        travelers_to_sym_I = list(np.random.choice(travelers_from_E_to_I, size=num_of_travelers_to_sym_I, replace=False))
        self.MegaAgentState.Is_travelers_list.extend(travelers_to_sym_I)
        self.MegaAgentState.Is = set(self.MegaAgentState.Is_travelers_list)

        travelers_to_asym_I = list(set(travelers_from_E_to_I) - set(travelers_to_sym_I))
        self.MegaAgentState.Ia_travelers_list.extend(travelers_to_asym_I)
        self.MegaAgentState.Ia = set(self.MegaAgentState.Ia_travelers_list)
        

    def transferTravelersFromItoR(self, 
                                  date: datetime,
                                  parameters) -> None:
        Is_len = len(self.MegaAgentState.Is_travelers_list)
        num_of_travelers_from_Is_to_R = round(Is_len * parameters.infection_duration.rate_from_symI_to_R)

        Ia_len = len(self.MegaAgentState.Ia_travelers_list)
        num_of_travelers_from_Ia_to_R = round(Ia_len * parameters.infection_duration.rate_from_asymI_to_R)

        if Is_len <= 0 and Ia_len <= 0:
            return None

        if Is_len != 0:
            travelers_from_Is_to_R = set()
            if Is_len >= num_of_travelers_from_Is_to_R:
                travelers_from_Is_to_R = set(self.MegaAgentState.Is_travelers_list[:num_of_travelers_from_Is_to_R])
                del self.MegaAgentState.Is_travelers_list[:num_of_travelers_from_Is_to_R] # remove travelers who are transferred to R
                self.MegaAgentState.Is = set(self.MegaAgentState.Is_travelers_list)
            else:
                travelers_from_Is_to_R = set(traveler for traveler in self.MegaAgentState.Is_travelers_list)
                self.MegaAgentState.Is_travelers_list.clear() # all travelers are removed to R
                self.MegaAgentState.Is = set()
            
            if len(travelers_from_Is_to_R) != 0:
                self.MegaAgentState.R_travelers.update(travelers_from_Is_to_R)
                self.contact_trace_roster[date].update(travelers_from_Is_to_R) # report to the contact trace roster
        
        if Ia_len != 0:
            travelers_from_Ia_to_R = set()
            if Ia_len >= num_of_travelers_from_Ia_to_R:
                travelers_from_Ia_to_R = set(self.MegaAgentState.Ia_travelers_list[:num_of_travelers_from_Ia_to_R])
                del self.MegaAgentState.Ia_travelers_list[:num_of_travelers_from_Ia_to_R] # remove travelers who are transferred to R
                self.MegaAgentState.Ia = set(self.MegaAgentState.Ia_travelers_list)
            else:
                travelers_from_Ia_to_R = set(traveler for traveler in self.MegaAgentState.Ia_travelers_list)
                self.MegaAgentState.Ia_travelers_list.clear() # all travelers are removed to R
                self.MegaAgentState.Ia = set()

            if len(travelers_from_Ia_to_R) != 0:
                self.MegaAgentState.R_travelers.update(travelers_from_Ia_to_R)
                # self.contact_trace_roster[date].update(travelers_from_Ia_to_R) # report to the contact trace roster######################################
        

    def conduct_testing(self, 
                        date: datetime, 
                        probs_to_be_tested: dict,
                        test_accuracy_rate: float, 
                        prob_of_self_report: float) -> dict:
        """
        Put certain amount of travelers into contact tracing roster. For each traveler,
        if he is willing to be tested, and if the test result is positive (no matter whether 
        accurate or not), and if he is willing to self report the result, then he will be in contact
        tracing roster. Every "if" is controlled by a probability parameter.
        """
        contact_tracing_roster = {date: set()}
        # use binomial distribution sampling to get the testing group
        testing_group_E_Is_Ia = set()
        testing_group_S_R = set()
        if len(self.MegaAgentState.S_travelers) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["S"], size=len(self.MegaAgentState.S_travelers))
            testing_group_S_R.update(traveler for if_test, traveler in zip(sampling_result, self.MegaAgentState.S_travelers) if if_test==1)
        if len(self.MegaAgentState.E_travelers_list) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["E"], size=len(self.MegaAgentState.E_travelers_list))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, self.MegaAgentState.E_travelers_list) if if_test==1)
        if len(self.MegaAgentState.Is_travelers_list) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["Is"], size=len(self.MegaAgentState.Is_travelers_list))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, self.MegaAgentState.Is_travelers_list) if if_test==1)
        if len(self.MegaAgentState.Ia_travelers_list) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["Ia"], size=len(self.MegaAgentState.Ia_travelers_list))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, self.MegaAgentState.Ia_travelers_list) if if_test==1)
        if len(self.MegaAgentState.R_travelers) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["R"], size=len(self.MegaAgentState.R_travelers))
            testing_group_S_R.update(traveler for if_test, traveler in zip(sampling_result, self.MegaAgentState.R_travelers) if if_test==1)

        self_report_group = set()
        if len(testing_group_E_Is_Ia) != 0:
            # in testing group of E/Ia/Is, get the accurate testing group
            sampling_result = np.random.binomial(n=1, p=test_accuracy_rate, size=len(testing_group_E_Is_Ia))
            accu_testing_group_E_Is_Ia = set(traveler for if_test, traveler in zip(sampling_result, testing_group_E_Is_Ia) if if_test==1)
            if len(accu_testing_group_E_Is_Ia) != 0:
                # get the self report group
                sampling_result = np.random.binomial(n=1, p=prob_of_self_report, size=len(accu_testing_group_E_Is_Ia))
                self_report_group.update(traveler for if_test, traveler in zip(sampling_result, accu_testing_group_E_Is_Ia) if if_test==1)

        if len(testing_group_S_R) != 0:
            # in testing group of S/R, get the false testing group
            sampling_result = np.random.binomial(n=1, p=1-test_accuracy_rate, size=len(testing_group_S_R))
            accu_testing_group_S_R = set(traveler for if_test, traveler in zip(sampling_result, testing_group_S_R) if if_test==1)
            if len(accu_testing_group_S_R) != 0:
                # get the self report group
                sampling_result = np.random.binomial(n=1, p=prob_of_self_report, size=len(accu_testing_group_S_R))
                self_report_group.update(traveler for if_test, traveler in zip(sampling_result, accu_testing_group_S_R) if if_test==1)

        # add the traveler to the contact_tracing_roster
        for traveler in self_report_group:
            contact_tracing_roster[date].add(traveler)
        return contact_tracing_roster


    def initiate_quarantine(self, 
                            traveler: int,
                            all_facilities_objects:dict,
                            quaranting_start_date: datetime, 
                            quarantine_length: int) -> None:
        """
        Transfer traveler from S, E, I to Q or Qe or Qs or Qa.
        """
        quarantine_end_date = quaranting_start_date + timedelta(days = quarantine_length)

        if traveler in self.MegaAgentState.Is:
            self.MegaAgentState.Qs_travelers[traveler] = quarantine_end_date
            self.MegaAgentState.Is_travelers_list.remove(traveler)
            self.MegaAgentState.Is.remove(traveler)
        elif traveler in self.MegaAgentState.Ia:
            self.MegaAgentState.Qa_travelers[traveler] = quarantine_end_date
            self.MegaAgentState.Ia_travelers_list.remove(traveler)
            self.MegaAgentState.Ia.remove(traveler)
        elif traveler in self.MegaAgentState.E_travelers_list:
            self.MegaAgentState.Qe_travelers[traveler] = quarantine_end_date
            self.MegaAgentState.E_travelers_list.remove(traveler)
        elif traveler in self.MegaAgentState.S_travelers:
            self.MegaAgentState.S_travelers.remove(traveler)
            self.MegaAgentState.Q_travelers[traveler] = quarantine_end_date

            quarantine_dates = [quaranting_start_date + timedelta(days = d + 1) for d in range(quarantine_length)]
            for q_date in quarantine_dates:
                if q_date > self.SimulationPeriodBasicInfo.Dates[-1]:
                    continue
                # remove the traveler from facility record list, since the traveler is quarantined
                urbano_agent = self.SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[traveler]
                for (f_name, visit_time_step) in self.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][q_date]:
                    all_facilities_objects[f_name].Date_TimeStep_Susceptibles[q_date][visit_time_step].remove(traveler)
                    all_facilities_objects[f_name].DateTimeStepAllVisitors[q_date][visit_time_step].remove(traveler)

    
    def pool_recover(self,
                     compartment: dict, 
                     date: datetime) -> None:
        """
        Transfer people in quarantine compartments to recover.
        """
        transit_to_recovery = set()
        for traveler, quarantine_end_date in compartment.items():
            if quarantine_end_date == date:
                transit_to_recovery.add(traveler)
        for to_recover_traveler in transit_to_recovery:
            del compartment[to_recover_traveler]
            self.MegaAgentState.R_travelers.add(to_recover_traveler)


    def bring_quarantined_S_to_S(self,
                                 date: datetime) -> None:
        """
        This covers the special case of people who were in S and not infected but ended up in Q.
        """
        Q_transitions = set()
        for traveler, quarantine_end_date in self.MegaAgentState.Q_travelers.items():
            if quarantine_end_date == date:
                Q_transitions.add(traveler)
        for traveler in Q_transitions:
            del self.MegaAgentState.Q_travelers[traveler]
            self.MegaAgentState.S_travelers.add(traveler)


    def MegaAgent_daily_update(self, 
                               date: datetime,
                               parameters) -> None:
        """
        Daily update compartment transfer from E to I to R.
        """
        self.transferTravelersFromEtoI(parameters)
        self.transferTravelersFromItoR(date, parameters)

        self.pool_recover(self.MegaAgentState.Qe_travelers, date)
        self.pool_recover(self.MegaAgentState.Qa_travelers, date)
        self.pool_recover(self.MegaAgentState.Qs_travelers, date)
        self.bring_quarantined_S_to_S(date)


    def record_daily_stat(self,
                          date: datetime,
                          stat) -> None:
        # to record the epi-progression
        stat.MegaAgents_Sn[self.MegaAgentName][date].append(len(self.MegaAgentState.S_travelers) + len(self.MegaAgentState.Q_travelers))
        stat.MegaAgents_En[self.MegaAgentName][date].append(len(self.MegaAgentState.E_travelers_list))
        stat.MegaAgents_In[self.MegaAgentName][date].append(len(self.MegaAgentState.Ia_travelers_list) + len(self.MegaAgentState.Is_travelers_list))
        stat.MegaAgents_Rn[self.MegaAgentName][date].append(len(self.MegaAgentState.R_travelers))
        stat.MegaAgents_Qn[self.MegaAgentName][date].append(len(self.MegaAgentState.Q_travelers) + len(self.MegaAgentState.Qe_travelers) + len(self.MegaAgentState.Qs_travelers) + len(self.MegaAgentState.Qa_travelers))
