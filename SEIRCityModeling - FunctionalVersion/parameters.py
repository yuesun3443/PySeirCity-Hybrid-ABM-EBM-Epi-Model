from interventions import DefaultTesting, DefaultContactTracing

class Parameters:
    ''' 
    This object holds default parameter settings.
    Modify each item that you want non-default values for. 
    '''
    def __init__(self, 
                 iteration: int, 
                 initial_exposed_count: int=10, 
                 asym_hazard_multiplier: float=0.8):
        self.Iteration = iteration
        self.initial_exposed_count = initial_exposed_count
        self.infection_duration = self.DefaultInfectionDuration()
        self.asym_hazard_multiplier = asym_hazard_multiplier

        self.quarantine_length = 14
        self.vaccine_benefit_self = 0.9 # if a traveler is vaccinated, prob of this traveler not get infected
        self.vaccine_benefit_others  = 0.5 # if a traveler is vaccinated, prob of this traveler not infect other people
        
        self.probs_to_be_tested = 0.75
        self.test_accuracy_rate = 0.9
        self.prob_of_self_report = 0.6
        self.quarantine_prob = 0.5
        self.conduct_testing = DefaultTesting(self.probs_to_be_tested, 
                                              self.test_accuracy_rate, 
                                              self.prob_of_self_report)

        self.contact_trace_date_length = 3
        self.contact_tracing = DefaultContactTracing(self.contact_trace_date_length, 
                                                     self.quarantine_length,
                                                     self.quarantine_prob)


    def info(self):
        for x in self.__dir__():
            if x[0] != '_' and x != 'info':
                if type(self.__getattribute__(x)) == float:
                    print(x, '=', round(self.__getattribute__(x), 5))
                else:
                    print(x, '=', self.__getattribute__(x))
                    
    def DefaultInfectionDuration(self):
        return VariedInfectionDurationResponse(1 / 4, 1 / 3.5, 1 / 4.5, 1 / 2, 0.75)


class VariedInfectionDurationResponse:
    ''' 
    rate_from_S_to_E - time-step rate at which susceptible person become exposed.
    rate_from_E_to_I - daily rate at which exposed person beomces infectious.
    rate_from_asymI_to_R - daily rate at which infectious, asymptomatic person recovers.
    rate_from_symI_to_R - daily rate at which infectious person develops symptoms.
    asym_fraction - fraction of people who will be asymptomatic.
    asym_hazard_multiplier - discount of hazard brought by an asymptomatic individual
    '''
    def __init__(self, 
                 rate_from_S_to_E: float=1/5/6,
                 rate_from_E_to_I: float=1/3.5, 
                 rate_from_asymI_to_R: float=1/4.5, 
                 rate_from_symI_to_R: float=1/2, 
                 asym_fraction: float=0.75, 
                 asym_hazard_multiplier: float=0.5,
                 std_from_E_to_I=0.72,
                 std_from_Is_to_R=2.0,
                 std_from_Ia_to_R=2.0):
        self.rate_from_S_to_E = rate_from_S_to_E
        self.rate_from_E_to_I = rate_from_E_to_I
        self.rate_from_asymI_to_R = rate_from_asymI_to_R
        self.rate_from_symI_to_R = rate_from_symI_to_R
        self.asym_fraction = asym_fraction
        self.asym_hazard_multiplier = asym_hazard_multiplier

        self.std_from_E_to_I = std_from_E_to_I
        self.std_from_Is_to_R = std_from_Is_to_R
        self.std_from_Ia_to_R = std_from_Ia_to_R

    
    def __str__(self):
        def nice_look(num):
            num = str(num)
            if len(num) > 6:
                return num[:6]
            else:
                return num
        return ''.join(['VariedResponse(rate from S to E: ', nice_look(self.rate_from_S_to_E),
                                     ', rate from E to I: ', nice_look(self.rate_from_E_to_I),
                                     ', rate from asymtomatic I to R: ', nice_look(self.rate_from_asymI_to_R),
                                       ', rate from symtomatic I to R: ', nice_look(self.rate_from_symI_to_R),
                                     ', asymptomatic fraction: ', nice_look(self.asym_fraction), 
                                     ', asymptomatic hazard multiplier: ', nice_look(self.asym_hazard_multiplier), ')'])
