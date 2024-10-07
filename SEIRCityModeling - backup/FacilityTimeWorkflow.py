import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

# Adjust the activity classification & retrieve baseline time use
# The classification of activity in US TIME USE SURVEY is not the same as that of urbano. 
# We need to re-classify the activities in the same way as thoese in urbano.


Urbano_FacilityTypes = ['home', 'meal', 
                        'other', 'school', 
                        'shoperr', 'socialrec', 'work']
date_format = '%Y-%m-%d'


def load_data_into_dataframe(facility_time_use_file):
    # Load the data into a pandas DataFrame
    original_time_use_df = pd.read_excel(facility_time_use_file, sheet_name='tu')
    original_time_use_df = original_time_use_df.set_index('Activity')
    return original_time_use_df


def re_structure_time_use_df(original_time_use_df):
    Time_Use_Columns_list = ["Avg hours/day, civilian pop, Weekdays",
                             "Avg hours/day, civilian pop, Weekends and holidays"]
    Activities_from_file = ["Personal care activities",
                            "Eating and drinking",
                            "Household activities",
                            "Purchasing goods and services",
                            "Caring for and helping household members",
                            "Caring for and helping nonhousehold members",
                            "Working and work-related activities",
                            "Educational activities",
                            "Organizational, civic, and religious activities",
                            "Leisure and sports",
                            "Telephone calls, mail, and e-mail",
                            "Other activities, not elsewhere classified"]

    # Filter the DataFrame to only include rows where the 'Activity' column matches one of the activities in Activities_from_file list
    filtered_time_use_df = original_time_use_df[original_time_use_df.index.isin(Activities_from_file)]
    filtered_time_use_df = filtered_time_use_df[Time_Use_Columns_list]
    return filtered_time_use_df


def get_baseline_time_use(time_use_df, total_hour_per_day = 24.0):
    Urbano_FacilityTypes = ['home', 'meal', 'other', 
                            'school', 'shoperr', 'socialrec', 'work']
    activity_time_use_baselines = []
    
    home_activities = ['Personal care activities', 
                   'Household activities', 
                   'Caring for and helping household members']
    home_df = time_use_df.loc[home_activities]
    home_hour_array = home_df.values.astype(float)
    home_hour_array = home_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(home_hour_array)

    meal_activities = ['Eating and drinking']
    meal_df = time_use_df.loc[meal_activities]
    meal_hour_array = meal_df.values.astype(float)
    meal_hour_array = meal_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(meal_hour_array)

    other_activities = ['Other activities, not elsewhere classified']
    other_df = time_use_df.loc[other_activities]
    other_hour_array = other_df.values.astype(float)
    other_hour_array = other_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(other_hour_array)

    school_activities = ['Educational activities']
    school_df = time_use_df.loc[school_activities]
    school_hour_array = school_df.values.astype(float)
    school_hour_array = school_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(school_hour_array)

    shoperr_activities = ['Purchasing goods and services']
    shoperr_df = time_use_df.loc[shoperr_activities]
    shoperr_hour_array = shoperr_df.values.astype(float)
    shoperr_hour_array = shoperr_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(shoperr_hour_array)

    socialrec_activities = ['Caring for and helping nonhousehold members', 
                        'Organizational, civic, and religious activities', 
                        'Leisure and sports']
    socialrec_df = time_use_df.loc[socialrec_activities]
    socialrec_hour_array = socialrec_df.values.astype(float)
    socialrec_hour_array = socialrec_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(socialrec_hour_array)

    work_activities = ['Telephone calls, mail, and e-mail',
                   'Working and work-related activities']
    work_df = time_use_df.loc[work_activities]
    work_hour_array = work_df.values.astype(float)
    work_hour_array = work_hour_array.sum(axis=0)/total_hour_per_day
    activity_time_use_baselines.append(work_hour_array)
    activity_time_use_baselines = np.vstack(activity_time_use_baselines)
    
    New_Time_Use_Columns_list = ['Avg prcntg/day, civilian pop, Weekdays',
                             'Avg prcntg/day, civilian pop, Weekends and holidays']
    baseline_time_use_df = pd.DataFrame(activity_time_use_baselines, 
                                        index=Urbano_FacilityTypes, columns=New_Time_Use_Columns_list)
    return baseline_time_use_df, activity_time_use_baselines



# Retrive Mobility Changes data#####################################################
def get_county_mobility_changes_df(mobility_change_file, county):
    all_mobility_changes_df = pd.read_csv(mobility_change_file)
    activity_time_use_change = ["date",
                                "retail_and_recreation_percent_change_from_baseline",
                                "grocery_and_pharmacy_percent_change_from_baseline",
                                "parks_percent_change_from_baseline",
                                "transit_stations_percent_change_from_baseline",
                                "workplaces_percent_change_from_baseline",
                                "residential_percent_change_from_baseline"]
    geo_info = ["sub_region_1",
                "sub_region_2", 
                "census_fips_code",
                "place_id"]
    mobility_column_list = []
    mobility_column_list.extend(activity_time_use_change)
    mobility_column_list.extend(geo_info)

    county_mobility_changes_df = all_mobility_changes_df[all_mobility_changes_df["sub_region_2"]==county]
    county_mobility_changes_df = county_mobility_changes_df[mobility_column_list].set_index("date")
    return county_mobility_changes_df, activity_time_use_change, geo_info 


def If_weekday(date: datetime)-> bool:
    # Check if it's a weekday or weekend
    # Weekdays are 0 (Monday) through 4 (Friday)
    if date.weekday() < 5:
        return True
    else:
        return False


# def get_real_time_use(time_baseline_use_df,
#                       county_mobility_changes_df, 
#                       date_time: datetime, 
#                       activity: str):
#     assert type(activity)==str, "activity must be a string"
#     assert activity in Urbano_FacilityTypes, "please select activity from ['home', 'meal, 'school', 'shoperr', 'socialrec', 'work']"
    
#     if_weekday = If_weekday(date_time)
#     date = date_time.strftime(date_format)
#     activity_time_use_df = None
#     if if_weekday:
#         activity_time_use_df = time_baseline_use_df[["Avg prcntg/day, civilian pop, Weekdays"]]
#     else:
#         activity_time_use_df = time_baseline_use_df[["Avg prcntg/day, civilian pop, Weekends and holidays"]]

#     base_line_time = None
#     time_change_percentage = None
#     if activity == 'home':
#         base_line_time = activity_time_use_df.at["home", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "residential_percent_change_from_baseline"]
#     elif activity == 'meal':
#         base_line_time = activity_time_use_df.at["meal", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "retail_and_recreation_percent_change_from_baseline"]
#     elif activity == 'other':
#         base_line_time = activity_time_use_df.at["other", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "transit_stations_percent_change_from_baseline"]
#     elif activity == 'school':
#         base_line_time = activity_time_use_df.at["school", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "workplaces_percent_change_from_baseline"]
#     elif activity == 'shoperr':
#         base_line_time = activity_time_use_df.at["shoperr", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "grocery_and_pharmacy_percent_change_from_baseline"]
#     elif activity == 'socialrec':
#         base_line_time = activity_time_use_df.at["socialrec", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "parks_percent_change_from_baseline"]
#     else:
#         base_line_time = activity_time_use_df.at["work", activity_time_use_df.columns[0]]
#         time_change_percentage = county_mobility_changes_df.at[date, "workplaces_percent_change_from_baseline"]
    
#     real_time_use = base_line_time * (1 + time_change_percentage/100)
#     return real_time_use



def get_baseline_time_and_mobility_change(facility_time_use_file: str, 
                                          mobility_change_file: str,
                                          county: str="New York County"):
    """
    Comprehensive method to output time_use_baseline_df and county_mobility_changes_df
    """
    # Load the data into a pandas DataFrame
    original_time_use_df = load_data_into_dataframe(facility_time_use_file)
    # Filter the DataFrame to only include rows where the 'Activity' column 
    # matches one of the activities in Activities_from_file list
    filtered_time_use_df = re_structure_time_use_df(original_time_use_df)
    time_use_baseline_df, activity_time_use_baselines = get_baseline_time_use(filtered_time_use_df)
    county_mobility_changes_df, activity_time_use_change, geo_info = get_county_mobility_changes_df(mobility_change_file, county)
    return time_use_baseline_df, county_mobility_changes_df


def convert_dfs_to_dict(time_baseline_use_df, county_mobility_changes_df):
    weekday_activity_time_use_df = time_baseline_use_df[["Avg prcntg/day, civilian pop, Weekdays"]]
    weekend_activity_time_use_df = time_baseline_use_df[["Avg prcntg/day, civilian pop, Weekends and holidays"]]

    time_use_baseline_dict = {}
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"] = {}
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"] = {}

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['home'] = weekday_activity_time_use_df.at["home", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['home'] = weekend_activity_time_use_df.at["home", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['meal'] = weekday_activity_time_use_df.at["meal", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['meal'] = weekend_activity_time_use_df.at["meal", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['other'] = weekday_activity_time_use_df.at["other", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['other'] = weekend_activity_time_use_df.at["other", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['school'] = weekday_activity_time_use_df.at["school", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['school'] = weekend_activity_time_use_df.at["school", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['shoperr'] = weekday_activity_time_use_df.at["shoperr", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['shoperr'] = weekend_activity_time_use_df.at["shoperr", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['socialrec'] = weekday_activity_time_use_df.at["socialrec", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['socialrec'] = weekend_activity_time_use_df.at["socialrec", weekend_activity_time_use_df.columns[0]]

    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]['work'] = weekday_activity_time_use_df.at["work", weekday_activity_time_use_df.columns[0]]
    time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]['work'] = weekend_activity_time_use_df.at["work", weekend_activity_time_use_df.columns[0]]

    county_mobility_changes_dict = {}
    for date, _ in county_mobility_changes_df.iterrows():
        date_obj = datetime.strptime(date, date_format)
        county_mobility_changes_dict[date_obj] = {}
        county_mobility_changes_dict[date_obj]["home"] = county_mobility_changes_df.at[date, "residential_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["meal"] = county_mobility_changes_df.at[date, "retail_and_recreation_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["other"] = county_mobility_changes_df.at[date, "transit_stations_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["school"] = county_mobility_changes_df.at[date, "workplaces_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["shoperr"] = county_mobility_changes_df.at[date, "grocery_and_pharmacy_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["socialrec"] = county_mobility_changes_df.at[date, "parks_percent_change_from_baseline"]
        county_mobility_changes_dict[date_obj]["work"] = county_mobility_changes_df.at[date, "workplaces_percent_change_from_baseline"]
    return time_use_baseline_dict, county_mobility_changes_dict


def get_real_time_use(time_use_baseline_dict: dict,
                      county_mobility_changes_dict: dict, 
                      date_time: datetime, 
                      activity: str):
    assert type(activity)==str, "activity must be a string"
    assert activity in Urbano_FacilityTypes, "please select activity from ['home', 'meal, 'school', 'shoperr', 'socialrec', 'work']"
    
    if_weekday = If_weekday(date_time)
    date = date_time.strftime(date_format)
    activity_time_use_dict = None
    if if_weekday:
        activity_time_use_dict = time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekdays"]
    else:
        activity_time_use_dict = time_use_baseline_dict["Avg prcntg/day, civilian pop, Weekends and holidays"]

    # base_line_time = None
    # time_change_percentage = None
    # if activity == 'home':
    #     base_line_time = activity_time_use_dict["home"]
    #     time_change_percentage = county_mobility_changes_dict[date]["home"]
    # elif activity == 'meal':
    #     base_line_time = activity_time_use_dict["meal"]
    #     time_change_percentage = county_mobility_changes_dict[date]["meal"]
    # elif activity == 'other':
    #     base_line_time = activity_time_use_dict["other"]
    #     time_change_percentage = county_mobility_changes_dict[date]["other"]
    # elif activity == 'school':
    #     base_line_time = activity_time_use_dict["school"]
    #     time_change_percentage = county_mobility_changes_dict[date]["school"]
    # elif activity == 'shoperr':
    #     base_line_time = activity_time_use_dict["shoperr"]
    #     time_change_percentage = county_mobility_changes_dict[date]["shoperr"]
    # elif activity == 'socialrec':
    #     base_line_time = activity_time_use_dict["socialrec"]
    #     time_change_percentage = county_mobility_changes_dict[date]["socialrec"]
    # else:
    #     base_line_time = activity_time_use_dict["work"]
    #     time_change_percentage = county_mobility_changes_dict[date]["work"]

    base_line_time = activity_time_use_dict[activity]
    time_change_percentage = county_mobility_changes_dict[date][activity]
    
    real_time_use = base_line_time * (1 + time_change_percentage/100)
    return real_time_use