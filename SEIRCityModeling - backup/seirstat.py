
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import numpy as np

class Statistics:
    def __init__(self, all_mega_agents:dict, dates: List[datetime]):
        self.Tn = dates
        self.all_mega_agents = all_mega_agents
        self.dates = dates

        # to record city level progression
        self.S_agg = {}
        self.E_agg = {}
        self.I_agg = {}
        self.R_agg = {}
        self.Q_agg = {}

        # to record mege_agent level progression
        self.MegaAgents_Sn = {}
        self.MegaAgents_En = {}
        self.MegaAgents_In = {}
        self.MegaAgents_Rn = {}
        self.MegaAgents_Qn = {}
        self.R_effs = {}
        self.initialize_stats()


    def initialize_stats(self):
        for mega_agent_name, mega_agent in self.all_mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue  # Skip this iteration if population is zero
            self.MegaAgents_Sn[mega_agent_name] = {}
            self.MegaAgents_En[mega_agent_name] = {}
            self.MegaAgents_In[mega_agent_name] = {}
            self.MegaAgents_Rn[mega_agent_name] = {}
            self.MegaAgents_Qn[mega_agent_name] = {}
            self.R_effs[mega_agent_name] = {}

            for date in self.dates:
                self.MegaAgents_Sn[mega_agent_name][date] = []
                self.MegaAgents_En[mega_agent_name][date] = []
                self.MegaAgents_In[mega_agent_name][date] = []
                self.MegaAgents_Rn[mega_agent_name][date] = []
                self.MegaAgents_Qn[mega_agent_name][date] = []
                self.R_effs[mega_agent_name][date] = []
            
    def get_aggregated_progression(self, 
                                   progressions: dict, 
                                   iteration: int) -> dict:
        """
        progressions should be one of self.MegaAgents_Sn, self.MegaAgents_En, 
        self.MegaAgents_In, and self.MegaAgents_Rn.
        """
        aggregated = {}
        for d in self.Tn:
            day_sum_res_array = np.zeros(iteration)
            for mega_agent_name, mega_agent in self.all_mega_agents.items():
                if mega_agent.MegaAgentPopulation == 0:
                    continue  # Skip this iteration if population is zero
                day_sum_res_array += np.array(progressions[mega_agent_name][d])
            day_sum_res_list = day_sum_res_array.tolist()
            aggregated[d] = day_sum_res_list
        return aggregated

    def get_aggregated_R_effs(self, 
                              parameters):
        """
        Effective reproduction number is to divide the number of 
        new infections on day t by the number of actively infectious 
        people on day t, multiplied by the average duration of infectiousness.
        """
        aggregated = {}
        total_new_Ia_count_dict = {}
        total_new_Is_count_dict = {}
        for d in self.Tn:
            total_new_Ia_count_dict[d] = list()
            total_new_Is_count_dict[d] = []
            day_R_effs = []
            for i in range(parameters.Iteration):
                total_new_Ia_count = 0
                total_new_Is_count = 0
                total_Ia_count = 0
                total_Is_count = 0
                for mega_agent_name, mega_agent in self.all_mega_agents.items():
                    if mega_agent.MegaAgentPopulation == 0:
                        continue  # Skip this iteration if population is zero

                    total_new_Ia_count += self.R_effs[mega_agent_name][d][i]["new infectious count"]["Ia count"]
                    total_new_Is_count += self.R_effs[mega_agent_name][d][i]["new infectious count"]["Is count"]
                    total_Ia_count += self.R_effs[mega_agent_name][d][i]["infectious count"]["Ia count"]
                    total_Is_count += self.R_effs[mega_agent_name][d][i]["infectious count"]["Is count"]
                total_new_Ia_count_dict[d].append(total_new_Ia_count)
                total_new_Is_count_dict[d].append(total_new_Is_count)

                if total_Ia_count == 0 and total_Is_count == 0:
                    day_R_eff = 0
                    day_R_effs.append(day_R_eff)
                else:
                    ################################################################################
                    prob_asymI_to_R = parameters.infection_duration.rate_from_asymI_to_R
                    Ia_denominator = 1e-9
                    for day, total_new_Ia_count_list in total_new_Ia_count_dict.items():
                        day_diff = (d-day).days
                        if day_diff == 0:
                            day_diff = 1
                        geometric_prob = prob_asymI_to_R * ((1.0-prob_asymI_to_R)**(day_diff-1.0))
                        new_Ia_on_day = total_new_Ia_count_list[i]
                        Ia_denominator += geometric_prob * new_Ia_on_day
                    R_eff_Ia = total_new_Ia_count/Ia_denominator                
                    
                    prob_symI_to_R = parameters.infection_duration.rate_from_symI_to_R
                    Is_denominator = 1e-9
                    for day, total_new_Is_count_list in total_new_Is_count_dict.items():
                        day_diff = (d-day).days
                        if day_diff == 0:
                            day_diff = 1
                        geometric_prob = prob_symI_to_R * ((1.0-prob_symI_to_R)**(day_diff-1.0))
                        new_Is_on_day = total_new_Is_count_list[i]
                        Is_denominator += geometric_prob * new_Is_on_day
                    R_eff_Is = total_new_Is_count/Is_denominator

                    day_R_eff = parameters.infection_duration.asym_fraction * R_eff_Ia +\
                                (1-parameters.infection_duration.asym_fraction) * R_eff_Is

                    ################################################################################
                    # days_Ia_infectious = 1.0/parameters.infection_duration.rate_from_asymI_to_R
                    # Ia_denominator = total_Ia_count * days_Ia_infectious
                    # R_eff_Ia = total_new_Ia_count/Ia_denominator  

                    # days_Is_infectious = 1.0/parameters.infection_duration.rate_from_symI_to_R
                    # Is_denominator = total_Is_count * days_Is_infectious
                    # R_eff_Is = total_new_Ia_count/Is_denominator    

                    # day_R_eff = parameters.asym_fraction * R_eff_Ia +\
                    #             (1-parameters.asym_fraction) * R_eff_Is
                    ################################################################################

                    # day_R_eff = (total_new_Ia_count+total_new_Is_count)/(total_Ia_count+total_Is_count)

                    day_R_effs.append(day_R_eff)
            aggregated[d] = day_R_effs
        return aggregated

    def plot_confidence_interval(self, compartment_progression, date_list, ax):
        # plot 95% confidence interval for total exposed
        means = [np.mean(compartment_progression[t]) for t in date_list]
        c_intervals_es = [1.96 * np.std(compartment_progression[t])/np.sqrt(len(compartment_progression[t])) for t in date_list]
        lower_bound_es = [ii-jj for ii,jj in zip(means,c_intervals_es)]
        upper_bound_es = [ii+jj for ii,jj in zip(means,c_intervals_es)]
        ax.plot(date_list, means)
        ax.fill_between(date_list, lower_bound_es, upper_bound_es, color='b',alpha=0.4)

    def plot_aggregated_progression(self, 
                                    parameters,
                                    if_plot_seperate=False):
        locator = mdates.AutoDateLocator(interval_multiples = False)
        formatter = mdates.AutoDateFormatter(locator)

        self.S_agg = self.get_aggregated_progression(self.MegaAgents_Sn, parameters.Iteration)
        self.E_agg = self.get_aggregated_progression(self.MegaAgents_En, parameters.Iteration)
        self.I_agg = self.get_aggregated_progression(self.MegaAgents_In, parameters.Iteration)
        self.R_agg = self.get_aggregated_progression(self.MegaAgents_Rn, parameters.Iteration)
        self.Q_agg = self.get_aggregated_progression(self.MegaAgents_Qn, parameters.Iteration)
        self.Reff_agg = self.get_aggregated_R_effs(parameters)

        print('Average final exposure count is', np.mean(self.R_agg[max(self.R_agg)]))
        if (len(self.R_agg[max(self.R_agg)]) > 1):
            print('Sample Standard deviation is:', np.std(self.R_agg[max(self.R_agg)], ddof = 1))

         # Plot percentiles
        ps = [1, 5, 25, 50, 75, 95, 100]
        ps.sort(reverse = True)

        if if_plot_seperate == True:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]           
            
            Sp = {p : [np.percentile(self.S_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax1.plot(self.Tn, Sp[p])

            Ep = {p : [np.percentile(self.E_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax2.plot(self.Tn, Ep[p])

            Ip = {p : [np.percentile(self.I_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax3.plot(self.Tn, Ip[p])

            Rp = {p : [np.percentile(self.R_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax4.plot(self.Tn, Rp[p])

            if parameters.Iteration >= 30:
                self.plot_confidence_interval(self.S_agg, self.Tn, ax1)
                self.plot_confidence_interval(self.E_agg, self.Tn, ax2)
                self.plot_confidence_interval(self.I_agg, self.Tn, ax3)
                self.plot_confidence_interval(self.R_agg, self.Tn, ax4)

            ax1.set_title('Aggregated Progression: Percentiles of Susceptible People')
            ax2.set_title('Aggregated Progression: Percentiles of Exposed People')
            ax3.set_title('Aggregated Progression: Percentiles of Infected People')
            ax4.set_title('Aggregated Progression: Percentiles of Recovered People')
            for ax in (ax1, ax2, ax3, ax4):
                ax.legend(list(map(lambda x : str(x) + '%', ps)))
                ax.set_ylabel('Number of People')
                ax.set_xlabel('Day of Year')
                ax.grid(True)
                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels 45 degrees
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 5))
            ax1.plot(self.Tn, [np.mean(self.S_agg[d]) for d in self.Tn])
            ax1.plot(self.Tn, [np.mean(self.E_agg[d]) for d in self.Tn])
            ax1.plot(self.Tn, [np.mean(self.I_agg[d]) for d in self.Tn])
            ax1.plot(self.Tn, [np.mean(self.R_agg[d]) for d in self.Tn])
            ax1.plot(self.Tn, [np.mean(self.Q_agg[d]) for d in self.Tn])
            ax1.legend(['S', 'E', 'I', 'R', 'Q'])
            ax1.set_title('Aggregated Epidemic Progression')
            ax1.set_xlabel('Day of Year')
            ax1.set_ylabel('Number of People')
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)
            ax1.grid(True)
            ax1.tick_params(axis='x', rotation=45)

            # Plot percentiles for total exposed.
            Ep = {p : [np.percentile(self.E_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax2.plot(self.Tn, Ep[p])
            ax2.legend(list(map(lambda x : str(x) + '%', ps)))
            ax2.set_title('Aggregated Progression: Percentiles of Exposed People')
            ax2.set_ylabel('Number of People')
            ax2.set_xlabel('Day of Year')
            ax2.grid(True)
            ax2.tick_params(axis='x', rotation=45)

            # Plot percentiles for number infectious.
            Ip = {p : [np.percentile(self.I_agg[d], p) for d in self.Tn] for p in ps}
            for p in ps:
                ax3.plot(self.Tn, Ip[p])
            ax3.legend(list(map(lambda x : str(x) + '%', ps)))
            ax3.set_title('Aggregated Progression: Percentiles of Infected People')
            ax3.set_ylabel('Number of People')
            ax3.set_xlabel('Day of Year')
            ax3.grid(True)
            ax3.tick_params(axis='x', rotation=45)

            if parameters.Iteration >= 30:
                self.plot_confidence_interval(self.E_agg, self.Tn, ax2)
                self.plot_confidence_interval(self.I_agg, self.Tn, ax3)
            
            # plot effective reproduction value
            plt.figure(figsize=(8,4.5))
            for i in range(parameters.Iteration):
                plt.plot(self.Tn, [self.Reff_agg[d][i] for d in self.Tn])
            plt.title('Effective Reproduction Number')
            plt.xlabel('Day of Year')
            plt.ylabel('Value')
            # Show the grid, legend, and the plot
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)  # Rotate date labels if they overlap
            plt.tight_layout()


    def get_MegaAgents_progressions(self):
        S_dict = {}
        E_dict = {}
        I_dict = {}
        R_dict = {}
        for mega_agent_name, mega_agent in self.all_mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue  # Skip this iteration if population is zero
            Ss = [np.mean(self.MegaAgents_Sn[mega_agent_name][d]) for d in self.Tn]
            S_dict[mega_agent_name] = Ss
            Es = [np.mean(self.MegaAgents_En[mega_agent_name][d]) for d in self.Tn]
            E_dict[mega_agent_name] = Es
            Is = [np.mean(self.MegaAgents_In[mega_agent_name][d]) for d in self.Tn]
            I_dict[mega_agent_name] = Is
            Rs = [np.mean(self.MegaAgents_Rn[mega_agent_name][d]) for d in self.Tn]
            R_dict[mega_agent_name] = Rs
        return S_dict, E_dict, I_dict, R_dict

    def plot_MegaAgents_progressions(self):
        locator = mdates.AutoDateLocator(interval_multiples = False)
        formatter = mdates.AutoDateFormatter(locator)
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        for mega_agent_name, mega_agent in self.all_mega_agents.items():
            if mega_agent.MegaAgentPopulation == 0:
                continue  # Skip this iteration if population is zero

            Ss = [np.mean(self.MegaAgents_Sn[mega_agent_name][d]) for d in self.Tn]
            ax1.plot(self.Tn, Ss)
            Es = [np.mean(self.MegaAgents_En[mega_agent_name][d]) for d in self.Tn]
            ax2.plot(self.Tn, Es)            
            Is = [np.mean(self.MegaAgents_In[mega_agent_name][d]) for d in self.Tn]
            ax3.plot(self.Tn, Is)            
            Rs = [np.mean(self.MegaAgents_Rn[mega_agent_name][d]) for d in self.Tn]
            ax4.plot(self.Tn, Rs)    

        ax1.set_title('Epidemic Progression of Individual MegaAgents: S')
        ax2.set_title('Epidemic Progression of Individual MegaAgents: E')
        ax3.set_title('Epidemic Progression of Individual MegaAgents: I')
        ax4.set_title('Epidemic Progression of Individual MegaAgents: R')

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_ylabel('Number of People')
            ax.set_xlabel('Day of Year')
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels 45 degrees
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)


