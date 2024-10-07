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
        self.FacilityStationaryDistribution = whole_simulation_period_bi.StationaryDistributions[self.MegaAgentName]

        # One MegaAgent contains state info for the sake of transimission dynamics
        self.MegaAgentState = State()
        self.dynamic_time_spent_dic = dict()
        # initialize S
        self.MegaAgentState.S_travelers = set([traveler for traveler in self.Residents])
        self.Risk = 0
        self.if_initialized = False
        self.contact_trace_roster = defaultdict(set)
        # used to calculate R_eff
        self.new_Ia_count = 0
        self.new_Is_count = 0

    def reset_MegaAgent(self) -> None:
        self.MegaAgentState = State()
        self.MegaAgentState.S_travelers = set([traveler for traveler in self.Residents])
        self.dynamic_time_spent_dic = dict()
        self.Risk = 0
        self.if_initialized = False  
        self.contact_trace_roster = defaultdict(set)
        self.new_Ia_count = 0
        self.new_Is_count = 0

    @staticmethod
    def get_dynamic_time_spent(date:datetime, 
                               UrbanoAgents,
                               SimulationPeriodBasicInfo):  
        travelers_time_spent_at_the_facilityType_dict = {}
        for urbano_agent in UrbanoAgents:  
            visit_facilities = SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][date]  
            time_spent_dict = dict()
            for facility_name, visit_time_step in visit_facilities:
                facility_type = facility_name[0]
                daily_time_spent_in_facilityType = SimulationPeriodBasicInfo.AllDatesFacilityTimeUse[date][facility_name]
                time_spent_dict[facility_type] = dict() 
                for time_step, ft_visit_count in SimulationPeriodBasicInfo.date_urbano_agent_timeStep_FacilityTypeVisitCount_dic[date][urbano_agent][facility_type].items():
                    time_spent_dict[facility_type][time_step] = SimulationPeriodBasicInfo.time_span_of_each_time_step_dic[time_step] * ft_visit_count / SimulationPeriodBasicInfo.date_urbano_agent_timeStep_AllFacilityVisitCount_dic[date][urbano_agent][time_step]

                # unitize time_spent_dict[facility_type]
                time_spent_ratio_sum = sum(time_spent_dict[facility_type].values())
                time_spent_dict[facility_type] = {time_step: daily_time_spent_in_facilityType * time_spent_ratio/time_spent_ratio_sum for time_step, time_spent_ratio in time_spent_dict[facility_type].items()}

            for traveler in SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]:
                travelers_time_spent_at_the_facilityType_dict[traveler] = time_spent_dict
        return travelers_time_spent_at_the_facilityType_dict


    @staticmethod
    def calculate_MegaAgent_risk(MegaAgentName, 
                                date:datetime,
                                current_time_step: int,
                                all_facilities_objects: dict,
                                SimulationPeriodBasicInfo):
        risk = 0
        FacilityStationaryDistribution = SimulationPeriodBasicInfo.StationaryDistributions[MegaAgentName]
        for facility in all_facilities_objects.values():
            facility_hazard = facility.FacilityHazard
            # Find daily time spent
            daily_time_spent = SimulationPeriodBasicInfo.AllDatesFacilityTimeUse[date][facility.FacilityName]
            # Find the faicitliy vist probability
            visit_prob = FacilityStationaryDistribution[current_time_step][0][facility.FaciltyIndex]
            risk += facility_hazard * daily_time_spent * visit_prob
        return risk
    

    @staticmethod
    def calculate_susceptible_travelers_risks(MegaAgentName,
                                             date:datetime,
                                             current_time_step: int,
                                             all_facilities_objects: dict,
                                             UrbanoAgents,
                                             SimulationPeriodBasicInfo,
                                             dynamic_time_spent_dic,
                                             S_travelers) -> dict:
        s_traveler_risk_dict = dict()
        s_traveler_visit_facilities = dict()
        for urbano_agent in UrbanoAgents:
            s_travelers = S_travelers & SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]
            if len(s_travelers)==0:
                continue

            visit_facilities = SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][date]
            for facility_name, visit_time_step in visit_facilities:
                if visit_time_step != current_time_step:
                    continue
            
                facility_object = all_facilities_objects[facility_name]
                for s_traveler in s_travelers:
                    if s_traveler in facility_object.Date_TimeStep_Susceptibles[date][current_time_step]:
                        if s_traveler not in s_traveler_visit_facilities:
                            s_traveler_visit_facilities[s_traveler] = set()
                    s_traveler_visit_facilities[s_traveler].add(facility_object)

        FacilityStationaryDistribution = SimulationPeriodBasicInfo.StationaryDistributions[MegaAgentName]
        for s_traveler, visit_facilities in s_traveler_visit_facilities.items():
            s_traveler_risk = 0

            for v_f in visit_facilities:
                facility_hazard = v_f.FacilityHazard
                facility_type = v_f.FacilityType
                # Find the faicitliy vist probability
                visit_prob = FacilityStationaryDistribution[current_time_step][0][v_f.FaciltyIndex]
                
                # get time spent at the facility type at the target time step
                time_spent = dynamic_time_spent_dic[s_traveler][facility_type][current_time_step]
                s_traveler_risk += facility_hazard * time_spent * visit_prob

            s_traveler_risk_dict[s_traveler] = s_traveler_risk
        return s_traveler_risk_dict


    @staticmethod
    def selectSusceptibleBasedOnTravelerRisk(S_travelers,
                                             delta_S: int, 
                                             s_traveler_risk_dict: dict) -> set:
        sampled_travelers = set()
        if len(S_travelers) == 0:
            return sampled_travelers
        
        # Check if all risks are zero
        s_travelers = list(S_travelers)
        if all(risk == 0 for risk in s_traveler_risk_dict.values()):
            sampled_travelers = set(np.random.choice(s_travelers, size=delta_S, replace=False))
        else:
            s_risk_weights = list(s_traveler_risk_dict.values())
            list_without_zeros = [r for r in s_risk_weights if r != 0]
            min_risk_weight = min(list_without_zeros)
            # if the risk_weight is 0, add a little value to it to prevent 0 value
            # it will be easier when using np.random.choice
            s_risk_weights = [min_risk_weight/1000 if risk == 0 else risk for risk in s_risk_weights]  
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


    @staticmethod
    def labelTravelsAsExposed(MegaAgentName,
                              date:datetime,
                              current_time_step: int,
                              all_facilities_objects: dict,
                              UrbanoAgents,
                              whole_simulation_period_bi,
                              dynamic_time_spent_dic,
                              S_travelers,
                              E_travelers_list,
                              V_travelers,
                              initial_infected:List[int]=None) -> None:
        """
        Decide how many travelers should be labeled as exposed within the MegaAgent.
        """
        if len(S_travelers) == 0:
            return None

        to_infect = set()
        if initial_infected is not None:
            to_infect.update(initial_infected)
        else:
            # used to decide how many travelers are exposed and removed from S.
            Risk = MegaAgent.calculate_MegaAgent_risk(MegaAgentName, date, current_time_step, all_facilities_objects, whole_simulation_period_bi)
            s_traveler_risk_dict = MegaAgent.calculate_susceptible_travelers_risks(MegaAgentName, date, current_time_step, all_facilities_objects, UrbanoAgents,
                                                                                   whole_simulation_period_bi, dynamic_time_spent_dic, S_travelers)
        
            delta_S = len(S_travelers) * Risk
            delta_S = round(delta_S)

            if delta_S == 0:
                return None

            if delta_S >= len(S_travelers):
                to_infect.update(S_travelers)
            else:
                to_infect = MegaAgent.selectSusceptibleBasedOnTravelerRisk(S_travelers, delta_S, s_traveler_risk_dict)

        random_nums = set(np.random.uniform(0, 1, size=len(to_infect)))
        for traveler, if_infectious in zip(to_infect, random_nums):
            # # if a traveler is not vacinated or is vacinated but useless,
            # # then label this as one exposed instance
            # if traveler not in self.MegaAgentState.V_travelers or if_infectious > parameters.vaccine_benefit_self:
            #     self.MegaAgentState.S_travelers.remove(traveler)

            if traveler not in V_travelers:
                S_travelers.remove(traveler)
                
                # remove the infected traveler from the facility Susceptible record.
                # from the infected date on, this infected traveler will be removed from
                # Susceptible record of every visited facility.
                current_date_index = whole_simulation_period_bi.Dates.index(date)
                urbano_agent = whole_simulation_period_bi.traveler_urbano_agent_mapping[traveler]
                for d in whole_simulation_period_bi.Dates[current_date_index:]:
                    for (f_name, visit_time_step) in whole_simulation_period_bi.urbano_agent_date_time_step_facility_dic[urbano_agent][d]:
                        if d is date and visit_time_step < current_time_step:
                            continue
                        else:
                            all_facilities_objects[f_name].Date_TimeStep_Susceptibles[d][visit_time_step].remove(traveler)

                E_travelers_list.append(traveler)

   
    @staticmethod
    def transferTravelersFromEtoI(parameters,
                                  E_travelers_list,
                                  Is_travelers_list,
                                  Ia_travelers_list,
                                  Is_set,
                                  Ia_set):
        E_len = len(E_travelers_list)
        # num_of_travelers_from_E_to_I = math.ceil(E_len * parameters.infection_duration.rate_from_E_to_I)
        num_of_travelers_from_E_to_I = round(E_len * parameters.infection_duration.rate_from_E_to_I)
        if E_len <= 0:
            return (0, 0)

        travelers_from_E_to_I = []
        if E_len >= num_of_travelers_from_E_to_I:
            travelers_from_E_to_I = E_travelers_list[:num_of_travelers_from_E_to_I]
            del E_travelers_list[:num_of_travelers_from_E_to_I] # remove travelers who are transferred to I
        else:
            travelers_from_E_to_I = [traveler for traveler in E_travelers_list]
            E_travelers_list.clear() # all travelers are removed to I

        num_of_travelers_to_sym_I = round(num_of_travelers_from_E_to_I * (1 - parameters.infection_duration.asym_fraction))
        travelers_to_sym_I = list(np.random.choice(travelers_from_E_to_I, size=num_of_travelers_to_sym_I, replace=False))
        Is_travelers_list.extend(travelers_to_sym_I)
        Is_set.update(set(Is_travelers_list))
        new_Is_count = len(travelers_to_sym_I)

        travelers_to_asym_I = list(set(travelers_from_E_to_I) - set(travelers_to_sym_I))
        Ia_travelers_list.extend(travelers_to_asym_I)
        Ia_set.update(set(Ia_travelers_list))
        new_Ia_count = len(travelers_to_asym_I)
        return (new_Is_count, new_Ia_count)
        

    @staticmethod
    def transferTravelersFromItoR(parameters,
                                  Is_travelers_list,
                                  Ia_travelers_list,
                                  Is_set,
                                  Ia_set,
                                  R_travelers) -> None:
        Is_len = len(Is_travelers_list)
        num_of_travelers_from_Is_to_R = round(Is_len * parameters.infection_duration.rate_from_symI_to_R)

        Ia_len = len(Ia_travelers_list)
        num_of_travelers_from_Ia_to_R = round(Ia_len * parameters.infection_duration.rate_from_asymI_to_R)

        if Is_len <= 0 and Ia_len <= 0:
            return None

        if Is_len != 0:
            travelers_from_Is_to_R = set()
            if Is_len >= num_of_travelers_from_Is_to_R:
                travelers_from_Is_to_R = set(Is_travelers_list[:num_of_travelers_from_Is_to_R])
                del Is_travelers_list[:num_of_travelers_from_Is_to_R] # remove travelers who are transferred to R
                Is_set.clear()
                Is_set.update(set(Is_travelers_list))
            else:
                travelers_from_Is_to_R = set(traveler for traveler in Is_travelers_list)
                Is_travelers_list.clear() # all travelers are removed to R
                Is_set.clear()
            
            if len(travelers_from_Is_to_R) != 0:
                R_travelers.update(travelers_from_Is_to_R)
                # self.contact_trace_roster[date].update(travelers_from_Is_to_R) # report to the contact trace roster
        
        if Ia_len != 0:
            travelers_from_Ia_to_R = set()
            if Ia_len >= num_of_travelers_from_Ia_to_R:
                travelers_from_Ia_to_R = set(Ia_travelers_list[:num_of_travelers_from_Ia_to_R])
                del Ia_travelers_list[:num_of_travelers_from_Ia_to_R] # remove travelers who are transferred to R
                Ia_set.clear()
                Ia_set.update(set(Ia_travelers_list))
            else:
                travelers_from_Ia_to_R = set(traveler for traveler in Ia_travelers_list)
                Ia_travelers_list.clear() # all travelers are removed to R
                Ia_set.clear()

            if len(travelers_from_Ia_to_R) != 0:
                R_travelers.update(travelers_from_Ia_to_R)
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
                            quarantine_length: int,
                            SimulationPeriodBasicInfo) -> None:
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
                if q_date > SimulationPeriodBasicInfo.Dates[-1]:
                    continue
                # remove the traveler from facility record list, since the traveler is quarantined
                urbano_agent = SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[traveler]
                for (f_name, visit_time_step) in SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][q_date]:
                    all_facilities_objects[f_name].Date_TimeStep_Susceptibles[q_date][visit_time_step].remove(traveler)
                    all_facilities_objects[f_name].DateTimeStepAllVisitors[q_date][visit_time_step].remove(traveler)

    
    @staticmethod
    def pool_recover(compartment: dict, 
                     R_travelers,
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
            R_travelers.add(to_recover_traveler)


    @staticmethod
    def bring_quarantined_S_to_S(date: datetime,
                                 S_travelers,
                                 Q_travelers) -> None:
        """
        This covers the special case of people who were in S and not infected but ended up in Q.
        """
        Q_transitions = set()
        for traveler, quarantine_end_date in Q_travelers.items():
            if quarantine_end_date == date:
                Q_transitions.add(traveler)
        for traveler in Q_transitions:
            del Q_travelers[traveler]
            S_travelers.add(traveler)


    @staticmethod
    def MegaAgent_daily_update(date: datetime,
                               parameters,
                               S_travelers,
                               E_travelers_list,
                               Is_travelers_list,
                               Ia_travelers_list,
                               Is_set,
                               Ia_set,
                               R_travelers,
                               Q_travelers,
                               Qe_travelers,
                               Qa_travelers,
                               Qs_travelers):
        """
        Daily update compartment transfer from E to I to R.
        """
        (new_Is_count, new_Ia_count) = MegaAgent.transferTravelersFromEtoI(parameters,
                                                                            E_travelers_list,
                                                                            Is_travelers_list,
                                                                            Ia_travelers_list,
                                                                            Is_set,
                                                                            Ia_set)
        MegaAgent.transferTravelersFromItoR(parameters,
                                            Is_travelers_list,
                                            Ia_travelers_list,
                                            Is_set,
                                            Ia_set,
                                            R_travelers)

        MegaAgent.pool_recover(Qe_travelers, R_travelers, date)
        MegaAgent.pool_recover(Qa_travelers, R_travelers, date)
        MegaAgent.pool_recover(Qs_travelers, R_travelers, date)
        MegaAgent.bring_quarantined_S_to_S(date, S_travelers, Q_travelers)
        return new_Is_count, new_Ia_count

    @staticmethod
    def initialize_MegaAgent(date: datetime, 
                             parameters, 
                             S_travelers,
                             Is_travelers_list,
                             Ia_travelers_list,
                             Is_set,
                             Ia_set,
                             new_Is_count,
                             new_Ia_count,
                             R_travelers,
                             Q_travelers,
                             Qe_travelers,
                             Qa_travelers,
                             Qs_travelers,
                             initial_infected:List[int]=None) -> None:
        num_of_initial_infected = len(initial_infected)
        num_of_travelers_to_sym_I = round(num_of_initial_infected * (1 - parameters.infection_duration.asym_fraction))
        travelers_to_sym_I = list(np.random.choice(initial_infected, size=num_of_travelers_to_sym_I, replace=False))
        Is_travelers_list.extend(travelers_to_sym_I)
        Is_set.clear()
        Is_set.update(set(Is_travelers_list))
        new_Is_count = len(travelers_to_sym_I)

        travelers_to_asym_I = list(set(initial_infected) - set(travelers_to_sym_I))
        Ia_travelers_list.extend(travelers_to_asym_I)
        Ia_set.clear()
        Ia_set.update(set(Ia_travelers_list))
        new_Ia_count = len(travelers_to_asym_I)

        MegaAgent.transferTravelersFromItoR(parameters,
                                            Is_travelers_list,
                                            Ia_travelers_list,
                                            Is_set,
                                            Ia_set,
                                            R_travelers)
        MegaAgent.pool_recover(Qe_travelers, R_travelers, date)
        MegaAgent.pool_recover(Qa_travelers, R_travelers, date)
        MegaAgent.pool_recover(Qs_travelers, R_travelers, date)
        MegaAgent.bring_quarantined_S_to_S(date, S_travelers, Q_travelers)


    def record_daily_stat(self,
                          date: datetime,
                          stat) -> None:
        # to record the epi-progression
        stat.MegaAgents_Sn[self.MegaAgentName][date].append(len(self.MegaAgentState.S_travelers) + len(self.MegaAgentState.Q_travelers))
        stat.MegaAgents_En[self.MegaAgentName][date].append(len(self.MegaAgentState.E_travelers_list))
        stat.MegaAgents_In[self.MegaAgentName][date].append(len(self.MegaAgentState.Ia_travelers_list) + len(self.MegaAgentState.Is_travelers_list))
        stat.MegaAgents_Rn[self.MegaAgentName][date].append(len(self.MegaAgentState.R_travelers))
        stat.MegaAgents_Qn[self.MegaAgentName][date].append(len(self.MegaAgentState.Q_travelers) + len(self.MegaAgentState.Qe_travelers) + len(self.MegaAgentState.Qs_travelers) + len(self.MegaAgentState.Qa_travelers))
        
        # to record data for later calculation of effective reproduction number
        R_eff_dict = dict()
        R_eff_dict["new infectious count"] = dict()
        R_eff_dict["new infectious count"]["Ia count"] = self.new_Ia_count
        R_eff_dict["new infectious count"]["Is count"] = self.new_Is_count

        R_eff_dict["infectious count"] = {}
        R_eff_dict["infectious count"]["Ia count"] = len(self.MegaAgentState.Ia)
        R_eff_dict["infectious count"]["Is count"] = len(self.MegaAgentState.Is)
        stat.R_effs[self.MegaAgentName][date].append(R_eff_dict)
                
        # reset the value at the end of each day
        self.new_Ia_count = 0
        self.new_Is_count = 0