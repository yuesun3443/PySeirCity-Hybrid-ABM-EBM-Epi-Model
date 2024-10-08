from datetime import datetime, timedelta
import numpy as np
from megaagent import MegaAgent

class DefaultTesting:
    def __init__(self, 
                probs_to_be_tested: float,
                test_accuracy_rate: float, 
                quarantine_prob: float,
                quarantine_length: int):
        pass
    def __str__(self):
        return 'DefaultTesting(no testing)'
    def test(self, 
             mega_agent,
             date: datetime, 
             all_facilities_objects:dict, 
             SimulationPeriodBasicInfo):
        pass

class GeneralTesting:
    """
    For each traveler, if he is willing to be tested, and if the test result is positive, then he will choose if to be
    quarantined. Every "if" is controlled by a probability parameter.
    """
    def __init__(self, 
                probs_to_be_tested: dict,
                test_accuracy_rate: float, 
                quarantine_prob: float,
                quarantine_length: int):
        self.probs_to_be_tested = probs_to_be_tested
        self.test_accuracy_rate = test_accuracy_rate
        self.quarantine_prob = quarantine_prob
        self.quarantine_length = quarantine_length
    
    def __str__(self):
        note = "Tesing: Probaility that a traveler chooses to test: " + str(self.probs_to_be_tested) + \
                " Test Accuracy: " + str(self.test_accuracy_rate) +\
                " Probability that the traveler self quarantined if test positive: " + str(self.quarantine_prob)  
        return note

    def test(self, 
             mega_agent,
             date: datetime, 
             all_facilities_objects:dict, 
             SimulationPeriodBasicInfo) -> None:
        # get the roster of self quarantined
        self_quarantined_dict = MegaAgent.conduct_testing(date, 
                                                        self.probs_to_be_tested, 
                                                        self.test_accuracy_rate, 
                                                        self.quarantine_prob,
                                                        mega_agent.MegaAgentState.S_set,
                                                        mega_agent.MegaAgentState.E_dict,
                                                        mega_agent.MegaAgentState.Is_dict,
                                                        mega_agent.MegaAgentState.Ia_dict,
                                                        mega_agent.MegaAgentState.R_dict)
        quaranting_start_date = date
        # quaranting travelers in the list
        for date, travelers_set in self_quarantined_dict.items():
            for traveler in travelers_set:
                MegaAgent.initiate_quarantine(traveler, 
                                            all_facilities_objects, 
                                            SimulationPeriodBasicInfo,
                                            quaranting_start_date, 
                                            self.quarantine_length,
                                            mega_agent.MegaAgentState.S_set,
                                            mega_agent.MegaAgentState.E_dict,
                                            mega_agent.MegaAgentState.Is_dict,
                                            mega_agent.MegaAgentState.Ia_dict,
                                            mega_agent.MegaAgentState.Is_set,
                                            mega_agent.MegaAgentState.Ia_set,
                                            mega_agent.MegaAgentState.Q_dict,
                                            mega_agent.MegaAgentState.Qe_dict,
                                            mega_agent.MegaAgentState.Qa_dict,
                                            mega_agent.MegaAgentState.Qs_dict)




class DefaultContactTracing:
    def __init__(self,
                 contact_tracing_date_length: int, 
                 quarantine_date_length: int,
                 quarantine_prob: float):
        pass
    def trace(self,
              mega_agent,
              date: datetime, 
              contact_trace_list: dict,
              all_facilities_objects: dict) -> None:
        pass
    def __str__(self):
        return "Default Contact Tracing: no action"


class ContactTracing:
    """
    To find the people who met with the travelers in contact tracing 
    roster within a specific period of time. And to initiate quarantine for all these people.
    """
    def __init__(self, 
                 contact_tracing_date_length: int, 
                 quarantine_date_length: int, 
                 quarantine_prob: float):
        self.contact_tracing_date_length = contact_tracing_date_length
        self.quarantine_date_length = quarantine_date_length
        self.quarantine_prob = quarantine_prob
    
    def __str__(self):
        note = "Contact Tracing: Trace date length: " + str(self.contact_tracing_date_length) + \
                " Quarantine length: " + str(self.quarantine_date_length) + \
                " Quarantine Probability: " + str(self.quarantine_prob)
        return note
    
    def trace(self, 
              mega_agent,
              cur_date: datetime, 
              contact_trace_list: dict,
              all_facilities_objects: dict) -> None:
        ''' 
        For each traveler to trace, find all of their peers from the past
        trace_length days (3 days by default). Then, quarantine all those 
        peers for some days.
        '''
        first_date_of_simulation = mega_agent.SimulationPeriodBasicInfo.Dates[0]
        quaranting_start_date = cur_date

        trace_dates = [cur_date - timedelta(days = d + 1) for d in range(self.contact_tracing_date_length)]
        # if d is earlier than the first_date_of_simulation, then do not add to the list
        trace_dates = [d for d in trace_dates if d >= first_date_of_simulation] 

        to_be_quarantined_list = set()
        # for traveler in mega_agent.contact_trace_roster[date]:
        for traveler in contact_trace_list[cur_date]:
            for d in trace_dates:
                # get all facilities visited by this traveler on date d
                urbano_agent = mega_agent.SimulationPeriodBasicInfo.traveler_urbano_agent_mapping[traveler]
                visit_facilities_on_d = set(mega_agent.SimulationPeriodBasicInfo.urbano_agent_date_time_step_facility_dic[urbano_agent][d])

                for (f_name, visit_time_step) in visit_facilities_on_d:
                    # get the peers that are around the traveler at the time_step on the date d
                    if traveler in all_facilities_objects[f_name].DateTimeStepAllVisitors[d][visit_time_step]:
                        to_be_quarantined_list.update(all_facilities_objects[f_name].DateTimeStepAllVisitors[d][visit_time_step])

        if len(to_be_quarantined_list) != 0:
            sampling_result = np.random.binomial(n=1, p=self.quarantine_prob, size=len(to_be_quarantined_list))
            quarantine_list = set(traveler for traveler, if_quarantine in zip(to_be_quarantined_list, sampling_result) if if_quarantine==1)

            # quaranting travelers in the list
            for to_be_quarantined_traveler in quarantine_list:
                mega_agent.initiate_quarantine(to_be_quarantined_traveler, all_facilities_objects, quaranting_start_date, self.quarantine_date_length)