from collections import defaultdict
from datetime import datetime, timedelta
import math
import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
from scipy.stats import lognorm
from typing import List

class Weibull:
    def __init__(self, mean_day, std_day):
        self.mean_day = mean_day # Mean time (days)
        self.std_day = std_day # Standard deviation (days)
        # Calculate the coefficient of variation (CV)
        self.cv = self.std_day/self.mean_day
        # Solve for k numerically
        k_initial_guess = 1.0 # Initial guess for k
        # weibull_k is the shape parameter of the weibull distribution
        self.weibull_k, = fsolve(self.cv_difference, k_initial_guess)
        # Once k is found, compute lambda, lambda is the scale parameter of weibull distribution
        self.weibull_lambda = self.mean_day / gamma(1 + 1 / self.weibull_k)

    def cv_difference(self, k):
        """compute the difference between theoretical CV and given CV."""
        if k <= 0:
            return np.inf
        cv_theoretical = np.sqrt(gamma(1 + 2 / k) / (gamma(1 + 1 / k))**2 - 1)
        return cv_theoretical - self.cv
    
    def weibull_cdf(self, day_len):
        """Calculate epi state transition probability for each person using Weibull CDF"""
        # p = 1 - np.exp(- (day_len / self.weibull_lambda)**self.weibull_k)
        # if p <= 1e-6:
        #     p = 1e-6
        return 1 - np.exp(- (day_len / self.weibull_lambda)**self.weibull_k)

    def get_normalized_weibull_probs(self, day_lens: list):
        # probabilities = [self.weibull_cdf(d_l) for d_l in day_lens]
        probabilities = self.weibull_cdf(day_lens)
        probabilities = [ 1e-6 if p <= 1e-6 else p for p in probabilities]
        # normalize the probabilities to sum to 1 (sampling weights)
        weights = probabilities / np.sum(probabilities)
        return weights

class LogNormal:
    def __init__(self, mean_day, std_day):
        self.mean_day = np.log(mean_day) # Mean time (days)
        self.std_day = std_day # Standard deviation (days)

    def lognormal_cdf(self, day_len):
        """Compute probability."""
        # Ensure day_len > 0
        day_len = np.maximum(day_len, 1e-6)
        # p = lognorm.cdf(day_len, s=self.std_day, scale=np.exp(self.mean_day))
        # if p <= 1e-6:
        #     p = 1e-6
        return lognorm.cdf(day_len, s=self.std_day, scale=np.exp(self.mean_day))
    
    def get_normalized_lognormal_probs(self, day_lens: list):
        probabilities = self.lognormal_cdf(day_lens)
        probabilities = [1e-6 if p <= 1e-6 else p for p in probabilities]
        # Normalize probabilities to use as weights
        weights = probabilities / np.sum(probabilities)
        return weights
        

class State:
    def __init__(self):
        self.S_set = set()
        self.E_dict = dict()

        self.Ia_dict = dict()
        self.Is_dict = dict()
        self.Ia_set = set()
        self.Is_set = set()

        self.R_dict = dict()

        self.V_dict = dict()

        self.Q_dict = dict()
        self.Qe_dict = dict()
        self.Qs_dict = dict()
        self.Qa_dict = dict()


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
        self.MegaAgentPopulation = len(self.Residents)
        self.FacilityStationaryDistribution = whole_simulation_period_bi.StationaryDistributions[self.MegaAgentName]

        # One MegaAgent contains state info for the sake of transimission dynamics
        self.MegaAgentState = State()
        self.dynamic_time_spent_dic = dict()
        # initialize S
        self.MegaAgentState.S_set = set([traveler for traveler in self.Residents])
        self.Risk = 0
        self.if_initialized = False
        # used to calculate R_eff
        self.new_Ia_count = 0
        self.new_Is_count = 0


    def reset_MegaAgent(self) -> None:
        self.MegaAgentState = State()
        self.MegaAgentState.S_set = set([traveler for traveler in self.Residents])
        self.dynamic_time_spent_dic = dict()
        self.if_initialized = False  
        self.new_Ia_count = 0
        self.new_Is_count = 0


    def record_daily_stat(self,
                          date: datetime,
                          stat) -> None:
        # to record the epi-progression
        stat.MegaAgents_Sn[self.MegaAgentName][date].append(len(self.MegaAgentState.S_set) + len(self.MegaAgentState.Q_dict))
        stat.MegaAgents_En[self.MegaAgentName][date].append(len(self.MegaAgentState.E_dict))
        stat.MegaAgents_In[self.MegaAgentName][date].append(len(self.MegaAgentState.Ia_dict) + len(self.MegaAgentState.Is_dict))
        stat.MegaAgents_Rn[self.MegaAgentName][date].append(len(self.MegaAgentState.R_dict))
        stat.MegaAgents_Qn[self.MegaAgentName][date].append(len(self.MegaAgentState.Q_dict) + len(self.MegaAgentState.Qe_dict) + len(self.MegaAgentState.Qs_dict) + len(self.MegaAgentState.Qa_dict))
        
        # to record data for later calculation of effective reproduction number
        R_eff_dict = dict()
        R_eff_dict["new infectious count"] = dict()
        R_eff_dict["new infectious count"]["Ia count"] = self.new_Ia_count
        R_eff_dict["new infectious count"]["Is count"] = self.new_Is_count

        R_eff_dict["infectious count"] = {}
        R_eff_dict["infectious count"]["Ia count"] = len(self.MegaAgentState.Ia_dict)
        R_eff_dict["infectious count"]["Is count"] = len(self.MegaAgentState.Is_dict)
        stat.R_effs[self.MegaAgentName][date].append(R_eff_dict)
                
        # reset the value at the end of each day
        self.new_Ia_count = 0
        self.new_Is_count = 0


    @staticmethod
    def get_dynamic_time_spent(date: datetime, 
                               UrbanoAgents: set,
                               SimulationPeriodBasicInfo):  
        #travelers_time_spent_at_the_facilityType_dict = {}
        urbano_agent_time_spent_at_the_facilityType_dict = {}
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

        #     for traveler in SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]:
        #         travelers_time_spent_at_the_facilityType_dict[traveler] = time_spent_dict
        # return travelers_time_spent_at_the_facilityType_dict

            urbano_agent_time_spent_at_the_facilityType_dict[urbano_agent] = time_spent_dict
        return urbano_agent_time_spent_at_the_facilityType_dict


    @staticmethod
    def calculate_MegaAgent_risk(MegaAgentName: tuple, 
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
    def calculate_susceptible_travelers_risks(MegaAgentName: tuple,
                                              date:datetime,
                                              current_time_step: int,
                                              all_facilities_objects: dict,
                                              SimulationPeriodBasicInfo,
                                              dynamic_time_spent_dic,
                                              S_set: set) -> dict:
        s_traveler_risk_dict = dict()
        s_traveler_visit_facilities = dict()
        UrbanoAgents = SimulationPeriodBasicInfo.mega_agent_Urbano_mapping[(MegaAgentName[0], MegaAgentName[1])]
        for urbano_agent in UrbanoAgents:
            s_travelers = S_set & SimulationPeriodBasicInfo.urbano_agents_travelers_mapping[urbano_agent]
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
                
                # get corresponding urbano agent
                urbano_agent = SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[s_traveler]
                # get time spent at the facility type at the target time step
                time_spent = dynamic_time_spent_dic[urbano_agent][facility_type][current_time_step]
                s_traveler_risk += facility_hazard * time_spent * visit_prob

            s_traveler_risk_dict[s_traveler] = s_traveler_risk
        return s_traveler_risk_dict


    @staticmethod
    def selectSusceptibleBasedOnTravelerRisk(S_set: set,
                                             delta_S: int, 
                                             s_traveler_risk_dict: dict) -> set:
        sampled_travelers = set()
        if len(S_set) == 0:
            return sampled_travelers
        
        # Check if all risks are zero
        s_travelers_list = list(S_set)
        if all(risk == 0 for risk in s_traveler_risk_dict.values()):
            sampled_travelers = set(np.random.choice(s_travelers_list, size=delta_S, replace=False))
        else:
            s_risk_weights = list(s_traveler_risk_dict.values())
            list_without_zeros = [r for r in s_risk_weights if r != 0]
            min_risk_weight = min(list_without_zeros)
            # if the risk_weight is 0, add a little value to it to prevent 0 value
            # it will be easier when using np.random.choice
            s_risk_weights = [min_risk_weight/1000 if risk == 0 else risk for risk in s_risk_weights]  
            # unitized weights
            # risk_sum = sum(s_risk_weights)
            # s_risk_weights = [risk/risk_sum for risk in s_risk_weights]
            s_risk_weights = s_risk_weights/sum(s_risk_weights)

            if len(s_risk_weights) != len(s_travelers_list):
                raise RuntimeError("Two lists must be equal length.")

            if delta_S >= len(s_travelers_list):
                sampled_travelers = set(s_traveler for s_traveler in s_travelers_list)
            else:
                # Sample travelers based on weights
                sampled_travelers = np.random.choice(s_travelers_list, size=delta_S, replace=False, p=s_risk_weights)
        return sampled_travelers


    @staticmethod
    def selectTravelersBasedOnDayLength(date: datetime, 
                                        transition_rate: float,
                                        days_std: float,
                                        num_of_travelers_to_transfer: int,
                                        traveler_list: List[tuple],
                                        distribution_type="weibull"):
        day_diffs = list()
        traveler_ids_list = list()
        traveler_date_dict = dict()
        for (traveler, first_transferred_date) in traveler_list:
            day_diff = (date-first_transferred_date).days
            day_diffs.append(day_diff)
            traveler_ids_list.append(traveler)
            traveler_date_dict[traveler]=first_transferred_date

        mean_day_until_transition = 1.0/transition_rate
        std_day_until_transition = days_std
        
        # sampling_weights = list()
        # if distribution_type=="weibull":
        #     wbd = Weibull(mean_day_until_transition, std_day_until_transition)
        #     sampling_weights = wbd.get_normalized_weibull_probs(day_diffs)
        # elif distribution_type=="lognormal":
        #     lgn = LogNormal(mean_day_until_transition, std_day_until_transition)
        #     sampling_weights = lgn.get_normalized_lognormal_probs(day_diffs)
        # else: 
        #     raise RuntimeError("distribution type must be either weibull or lognormal.")

        sampling_weights = [1/len(traveler_ids_list) for _ in range(len(traveler_ids_list))]
        sampled_traveler_ids = set(np.random.choice(traveler_ids_list, size=num_of_travelers_to_transfer, replace=False, p=sampling_weights))
        sampled_travelers = [(traveler_id, date) for traveler_id in sampled_traveler_ids]

        for traveler_id in sampled_traveler_ids:
            del traveler_date_dict[traveler_id]
        traveler_list = [(traveler_id, date) for traveler_id, date in traveler_date_dict.items()]
        return traveler_list, sampled_travelers


    @staticmethod
    def labelTravelsAsExposed(MegaAgentName: tuple,
                              date:datetime,
                              current_time_step: int,
                              parameters,
                              all_facilities_objects: dict,
                              whole_simulation_period_bi,
                              dynamic_time_spent_dic: dict,
                              S_set: set,
                              E_dict: dict,
                              V_dict: dict,
                              initial_infected:List[int]=None) -> None:
        """
        Decide how many travelers should be labeled as exposed within the MegaAgent.
        """
        if len(S_set) == 0:
            return None

        to_infect = set()
        if initial_infected is not None:
            to_infect.update(initial_infected)
        else:
            # used to decide how many travelers are exposed and removed from S.
            Risk = MegaAgent.calculate_MegaAgent_risk(MegaAgentName, date, current_time_step, all_facilities_objects, whole_simulation_period_bi)
            s_traveler_risk_dict = MegaAgent.calculate_susceptible_travelers_risks(MegaAgentName, date, current_time_step, all_facilities_objects,
                                                                                   whole_simulation_period_bi, dynamic_time_spent_dic, S_set)
            Risk = 1 - math.exp(-Risk)
            delta_S = len(S_set) * Risk
            delta_S = round(delta_S)

            if delta_S == 0:
                return None

            if delta_S >= len(S_set):
                to_infect.update(S_set)
            else:
                to_infect = MegaAgent.selectSusceptibleBasedOnTravelerRisk(S_set, delta_S, s_traveler_risk_dict)
        
        random_nums = set(np.random.uniform(0, 1, size=len(to_infect)))
        for traveler, if_infectious in zip(to_infect, random_nums):
            # # if a traveler is not vacinated or is vacinated but useless,
            # # then label this as one exposed instance
            # if traveler not in self.MegaAgentState.V_travelers or if_infectious > parameters.vaccine_benefit_self:
            #     self.MegaAgentState.S_set.remove(traveler)

            if traveler not in V_dict:
                S_set.remove(traveler)
                
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
                E_dict[traveler] = date
        
        # This covers spontaneous infections.
        spontaneous = np.random.poisson(parameters.infection_duration.daily_spontaneous_prob * len(S_set))
        if spontaneous:
            to_infect = set(np.random.choice(list(S_set), replace = True, size = spontaneous))
            random_nums = set(np.random.uniform(0, 1, size=len(to_infect)))
            for traveler, if_infectious in zip(to_infect, random_nums):
                if traveler not in V_dict:
                    S_set.remove(traveler)

                    current_date_index = whole_simulation_period_bi.Dates.index(date)
                    urbano_agent = whole_simulation_period_bi.traveler_urbano_agent_mapping[traveler]
                    for d in whole_simulation_period_bi.Dates[current_date_index:]:
                        for (f_name, visit_time_step) in whole_simulation_period_bi.urbano_agent_date_time_step_facility_dic[urbano_agent][d]:
                            if d is date and visit_time_step < current_time_step:
                                continue
                            else:
                                all_facilities_objects[f_name].Date_TimeStep_Susceptibles[d][visit_time_step].remove(traveler)
                    E_dict[traveler] = date


    @staticmethod
    def transferTravelersFromEtoI(date: datetime,
                                  parameters,
                                  E_dict: dict,
                                  Is_dict: dict,
                                  Ia_dict: dict, 
                                  Is_set: set, 
                                  Ia_set: set):
        E_len = len(E_dict)
        num_of_travelers_from_E_to_I = round(E_len * parameters.infection_duration.rate_from_E_to_I)
        if E_len <= 0:
            return (0, 0)

        travelers_from_E_to_I = []
        if E_len >= num_of_travelers_from_E_to_I:
            day_diffs = [(date-first_transferred_date).days for traveler, first_transferred_date in E_dict.items()]
            wbd = Weibull(parameters.infection_duration.rate_from_E_to_I, parameters.infection_duration.std_from_E_to_I)
            sampling_weights = wbd.get_normalized_weibull_probs(day_diffs)
            travelers_from_E_to_I = list(np.random.choice(list(E_dict.keys()), size=num_of_travelers_from_E_to_I, replace=False, p=sampling_weights))
            #travelers_from_E_to_I = list(E_dict.keys())[:num_of_travelers_from_E_to_I]

            for traveler in travelers_from_E_to_I: 
                del E_dict[traveler] # remove travelers who are transferred to I
        else:
            travelers_from_E_to_I = list(E_dict.keys())
            E_dict.clear() # all travelers are removed to I

        num_of_travelers_to_sym_I = round(num_of_travelers_from_E_to_I * (1 - parameters.infection_duration.asym_fraction))
        travelers_to_sym_I = list(np.random.choice(travelers_from_E_to_I, size=num_of_travelers_to_sym_I, replace=False))
        travelers_to_asym_I = list(set(travelers_from_E_to_I) - set(travelers_to_sym_I))

        for traveler in travelers_to_sym_I:
            Is_dict[traveler] = date
            Is_set.add(traveler)
        new_Is_count = len(travelers_to_sym_I)

        for traveler in travelers_to_asym_I:
            Ia_dict[traveler] = date
            Ia_set.add(traveler)
        new_Ia_count = len(travelers_to_asym_I)
        return (new_Is_count, new_Ia_count)
        

    @staticmethod
    def transferTravelersFromItoR(date: datetime,
                                  parameters,
                                  Is_dict: dict,
                                  Ia_dict: dict,
                                  Is_set: set,
                                  Ia_set: set,
                                  R_dict: dict) -> None:
        Is_len = len(Is_dict)
        num_of_travelers_from_Is_to_R = round(Is_len * parameters.infection_duration.rate_from_symI_to_R)

        Ia_len = len(Ia_dict)
        num_of_travelers_from_Ia_to_R = round(Ia_len * parameters.infection_duration.rate_from_asymI_to_R)

        if Is_len <= 0 and Ia_len <= 0:
            return None

        if Is_len != 0:
            travelers_from_Is_to_R = set()
            if Is_len >= num_of_travelers_from_Is_to_R:
                day_diffs = [(date-first_transferred_date).days for traveler, first_transferred_date in Is_dict.items()]
                lgn = LogNormal(parameters.infection_duration.rate_from_symI_to_R, parameters.infection_duration.std_from_Is_to_R)
                sampling_weights = lgn.get_normalized_lognormal_probs(day_diffs)
                travelers_from_Is_to_R = list(np.random.choice(list(Is_dict.keys()), size=num_of_travelers_from_Is_to_R, replace=False, p=sampling_weights))

                for traveler in travelers_from_Is_to_R:
                    del Is_dict[traveler] # remove travelers who are transferred to R
                    Is_set.remove(traveler)
            else:
                travelers_from_Is_to_R = set(Is_dict.keys())
                Is_dict.clear() # all travelers are removed to R
                Is_set.clear()
            
            for traveler in travelers_from_Is_to_R:
                R_dict[traveler] = date
        
        if Ia_len != 0:
            travelers_from_Ia_to_R = set()
            if Ia_len >= num_of_travelers_from_Ia_to_R:
                day_diffs = [(date-first_transferred_date).days for traveler, first_transferred_date in Ia_dict.items()]
                lgn = LogNormal(parameters.infection_duration.rate_from_asymI_to_R, parameters.infection_duration.std_from_Ia_to_R)
                sampling_weights = lgn.get_normalized_lognormal_probs(day_diffs)
                travelers_from_Ia_to_R = list(np.random.choice(list(Ia_dict.keys()), size=num_of_travelers_from_Ia_to_R, replace=False, p=sampling_weights))

                for traveler in travelers_from_Ia_to_R:
                    del Ia_dict[traveler] # remove travelers who are transferred to R
                    Ia_set.remove(traveler)
            else:
                travelers_from_Ia_to_R = set(Ia_dict.keys())
                Ia_dict.clear() # all travelers are removed to R
                Ia_set.clear()

            for traveler in travelers_from_Ia_to_R:
                R_dict[traveler] = date


    @staticmethod
    def conduct_testing(date: datetime, 
                        probs_to_be_tested: dict,
                        test_accuracy_rate: float, 
                        prob_of_self_quarantined: float,
                        S_set: set,
                        E_dict: dict,
                        Is_dict: dict,
                        Ia_dict: dict,
                        R_dict: dict) -> dict:
        """
        For each traveler, if he is willing to be tested, and if the test result is positive (no matter whether 
        accurate or not), then he will choose if self quarantine. Every "if" is controlled by a probability parameter.
        """
        self_quarantined_dict = {date: set()}
        # use binomial distribution sampling to get the testing group
        testing_group_E_Is_Ia = set()
        testing_group_S_R = set()
        if len(S_set) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["S"], size=len(S_set))
            testing_group_S_R.update(traveler for if_test, traveler in zip(sampling_result, S_set) if if_test==1)
        if len(E_dict) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["E"], size=len(E_dict))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, E_dict) if if_test==1)
        if len(Is_dict) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["Is"], size=len(Is_dict))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, Is_dict) if if_test==1)
        if len(Ia_dict) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["Ia"], size=len(Ia_dict))
            testing_group_E_Is_Ia.update(traveler for if_test, traveler in zip(sampling_result, Ia_dict) if if_test==1)
        if len(R_dict) != 0:
            sampling_result = np.random.binomial(n=1, p=probs_to_be_tested["R"], size=len(R_dict))
            testing_group_S_R.update(traveler for if_test, traveler in zip(sampling_result, R_dict) if if_test==1)

        self_quarantined_group = set()
        if len(testing_group_E_Is_Ia) != 0:
            # in testing group of E/Ia/Is, get the accurate testing group
            accu_result = np.random.binomial(n=1, p=test_accuracy_rate, size=len(testing_group_E_Is_Ia))
            accu_testing_group_E_Is_Ia = set(traveler for if_accu, traveler in zip(accu_result, testing_group_E_Is_Ia) if if_accu==1)
            if len(accu_testing_group_E_Is_Ia) != 0:
                # get the self quarantined group
                quarantine_result = np.random.binomial(n=1, p=prob_of_self_quarantined, size=len(accu_testing_group_E_Is_Ia))
                self_quarantined_group.update(traveler for if_quarantine, traveler in zip(quarantine_result, accu_testing_group_E_Is_Ia) if if_quarantine==1)

        if len(testing_group_S_R) != 0:
            # in testing group of S/R, get the false testing group
            accu_result = np.random.binomial(n=1, p=1-test_accuracy_rate, size=len(testing_group_S_R))
            accu_testing_group_S_R = set(traveler for if_accu, traveler in zip(accu_result, testing_group_S_R) if if_accu==1)
            if len(accu_testing_group_S_R) != 0:
                # get the self quarantined group
                sampling_result = np.random.binomial(n=1, p=prob_of_self_quarantined, size=len(accu_testing_group_S_R))
                self_quarantined_group.update(traveler for if_test, traveler in zip(sampling_result, accu_testing_group_S_R) if if_test==1)

        # add the traveler to the self_quarantined_group
        for traveler in self_quarantined_group:
            self_quarantined_dict[date].add(traveler)
        return self_quarantined_dict


    @staticmethod
    def initiate_quarantine(traveler: int,
                            all_facilities_objects:dict,
                            SimulationPeriodBasicInfo,
                            quaranting_start_date: datetime, 
                            quarantine_length: int,
                            S_set: set,
                            E_dict: dict,
                            Is_dict: dict,
                            Ia_dict: dict,
                            Is_set: set,
                            Ia_set: set,
                            Q_dict: dict,
                            Qe_dict: dict,
                            Qs_dict: dict,
                            Qa_dict: dict) -> None:
        """
        Transfer traveler from S, E, I to Q or Qe or Qs or Qa.
        """
        quarantine_end_date = quaranting_start_date + timedelta(days = quarantine_length)
        #print(traveler)
        if traveler in Is_dict:
            Qs_dict[traveler] = quarantine_end_date
            del Is_dict[traveler]
            Is_set.remove(traveler)
        elif traveler in Ia_dict:
            Qa_dict[traveler] = quarantine_end_date
            del Ia_dict[traveler]
            Ia_set.remove(traveler)
        elif traveler in E_dict:
            Qe_dict[traveler] = quarantine_end_date
            del E_dict[traveler]
        elif traveler in S_set:
            S_set.remove(traveler)
            Q_dict[traveler] = quarantine_end_date

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
                     R_dict: dict,
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
            R_dict[to_recover_traveler] = date


    @staticmethod
    def bring_quarantined_S_to_S(date: datetime,
                                 S_set: set,
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
            S_set.add(traveler)


    @staticmethod
    def MegaAgent_daily_update(date: datetime,
                               parameters,
                               S_set: set,
                               E_dict: dict,
                               Is_dict: dict,
                               Ia_dict: dict,
                               Is_set: set,
                               Ia_set: set,
                               R_dict: dict,
                               Q_dict: dict,
                               Qe_dict: dict,
                               Qa_dict: dict,
                               Qs_dict: dict):
        """
        Daily update compartment transfer from E to I to R.
        """
        (new_Is_count, new_Ia_count) = MegaAgent.transferTravelersFromEtoI(date,
                                                                           parameters,
                                                                           E_dict,
                                                                           Is_dict,
                                                                           Ia_dict,
                                                                           Is_set, 
                                                                           Ia_set)
        MegaAgent.transferTravelersFromItoR(date,
                                            parameters,
                                            Is_dict,
                                            Ia_dict,
                                            Is_set,
                                            Ia_set,
                                            R_dict)

        MegaAgent.pool_recover(Qe_dict, R_dict, date)
        MegaAgent.pool_recover(Qa_dict, R_dict, date)
        MegaAgent.pool_recover(Qs_dict, R_dict, date)
        MegaAgent.bring_quarantined_S_to_S(date, S_set, Q_dict)
        return new_Is_count, new_Ia_count


    @staticmethod
    def initialize_MegaAgent(date: datetime, 
                             parameters, 
                             S_set: set,
                             Is_dict: dict,
                             Ia_dict: dict,
                             Is_set: set,
                             Ia_set: set,
                             new_Is_count,
                             new_Ia_count,
                             R_dict: dict,
                             Q_dict: dict,
                             Qe_dict: dict,
                             Qa_dict: dict,
                             Qs_dict: dict,
                             initial_infected:List[int]=None) -> None:
        num_of_initial_infected = len(initial_infected)
        num_of_travelers_to_sym_I = round(num_of_initial_infected * (1 - parameters.infection_duration.asym_fraction))
        travelers_to_sym_I = list(np.random.choice(initial_infected, size=num_of_travelers_to_sym_I, replace=False))
        for traveler in travelers_to_sym_I:
            Is_dict[traveler] = date
            Is_set.add(traveler)
        new_Is_count = len(travelers_to_sym_I)

        travelers_to_asym_I = list(set(initial_infected) - set(travelers_to_sym_I))
        for traveler in travelers_to_asym_I:
            Ia_dict[traveler] = date
            Ia_set.add(traveler)
        new_Ia_count = len(travelers_to_asym_I)

        MegaAgent.transferTravelersFromItoR(date,
                                            parameters,
                                            Is_dict,
                                            Ia_dict,
                                            Is_set,
                                            Ia_set,
                                            R_dict)

        MegaAgent.pool_recover(Qe_dict, R_dict, date)
        MegaAgent.pool_recover(Qa_dict, R_dict, date)
        MegaAgent.pool_recover(Qs_dict, R_dict, date)
        MegaAgent.bring_quarantined_S_to_S(date, S_set, Q_dict)
