import pandas as pd
from itertools import product
import torch
import numpy

# This workflow is to construct stationary distributions for all 
# the traveler types at all the blocks. Each value in a distribution
# means p_type_f in the equation.


def create_transition_tensor(size: int):
    """
    Function to create a zero tensor for the transition matrix.
    """
    # Ensure the tensor is created directly on the GPU if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Add a small constant to handle rows that sum to zero if there are any
    fill_value = 1e-9
    return torch.full((size, size), fill_value, dtype=torch.float32, device=device)

def compute_stationary_distribution(transition_matrix, iteration=50):
    # Assuming transition_matrix is a PyTorch tensor and already on the GPU
    n = transition_matrix.shape[1]
    # Create an initial distribution, uniform distribution
    v = torch.full((1,n), 1/n, device=transition_matrix.device)
    # Perform normalization if not already done
    row_sums = transition_matrix.sum(dim=1, keepdim=True)
    transition_matrix_norm = transition_matrix / row_sums
    # Iteratively apply the transition matrix to the distribution
    for _ in range(iteration):
        v = torch.matmul(v, transition_matrix_norm)
    return v

def create_tm_for_non_exist_homeblock_travelertype_pair(size: int):
    # Create a transition tensor initialized with zeros
    transition_tensor = create_transition_tensor(size)
    # # Convert the transition_tensor to cuda version
    # transition_tensor = transition_tensor.cuda()
    # Add a small constant to handle rows that sum to zero if there are any
    transition_tensor += torch.where(transition_tensor == 0, 1e-9, 0)
    # Normalize the transition matrix to make the rows sum to 1
    transition_tensor = transition_tensor / transition_tensor.sum(dim=1, keepdim=True)
    return transition_tensor

def create_sd_for_non_exist_homeblock_travelertype_pair(size: int):
    transition_tensor = create_tm_for_non_exist_homeblock_travelertype_pair(size)
    stationary_distribution = compute_stationary_distribution(transition_tensor)
    #stationary_distribution = stationary_distribution.cpu()
    return stationary_distribution


def create_tm_sd_for_exist_homeblock_travelertype_pair(urbano_trip_volumn_file: str,
                                                       TotalBlocks_count: int):
    """
    Unique HomeBlock-TravelerType pair, i.e. (home_block, traveler_type).
    """
    # Load the data into a pandas DataFrame
    urbano_trip_df = pd.read_excel(urbano_trip_volumn_file, sheet_name='TripInformation')
    # Ensure AsPopulation is of integer type
    urbano_trip_df['AsPopulation'] = urbano_trip_df['AsPopulation'].astype(int)
    # Duplicate each row by AsPopulation value
    urbano_trip_df = urbano_trip_df.loc[urbano_trip_df.index.repeat(urbano_trip_df['AsPopulation'])].reset_index(drop=True).drop(columns=['AsPopulation'])
    urbano_trip_df = urbano_trip_df.groupby(['TravelerHomeBlock','TravelerType', 'TimeOfDay', 'FromActivity', 'OriginBlockID', 'ToActivity', 'DestBlockID']).size().reset_index(name='Count')
    # Get all distinct TravelerHomeBlock and TravelerType pairs
    distinct_pairs = urbano_trip_df[['TravelerHomeBlock', 'TravelerType']].drop_duplicates()


    # Basic Parameters of the city and facilities
    Urbano_FacilityTypes = ['home', 'meal', 'other', 'school', 'shoperr', 'socialrec', 'work']
    Facilities = list(product(Urbano_FacilityTypes, range(TotalBlocks_count)))
    size = len(Facilities)
    # Mapping facilities to indices
    facility_to_index = {facility: i for i, facility in enumerate(Facilities)} 
    time_steps = [i for i in range(1,7)]


    # Initialize a dictionary to store the Stationary distribution for each pair
    stationary_distributions = {}

    # Iterate over each distinct pair to create a transition matrix
    # Then from the matrix to create a stationary distribution
    for _, row in distinct_pairs.iterrows():
        home_block, traveler_type = row['TravelerHomeBlock'], row['TravelerType']
        stationary_distributions[(home_block, traveler_type)] = {}

        for time_step in time_steps:
            # Filter dataframe for the current pair
            filtered_df = urbano_trip_df[(urbano_trip_df['TravelerHomeBlock'] == home_block) & 
                                        (urbano_trip_df['TravelerType'] == traveler_type) &
                                        (urbano_trip_df['TimeOfDay'] == time_step)]
            
            # Create a transition tensor initialized with zeros
            transition_tensor = create_transition_tensor(size)
            
            # Update the tensor with counts from filtered_df
            for _, trip in filtered_df.iterrows():
                from_index = facility_to_index[(trip['FromActivity'], trip['OriginBlockID'])]
                to_index = facility_to_index[(trip['ToActivity'], trip['DestBlockID'])]
                count = trip['Count']
                
                # Increment the appropriate cell in the tensor
                transition_tensor[from_index, to_index] += count
            
            # Convert the transition_tensor to cuda version
            transition_tensor = transition_tensor.cuda()
            # Normalize the transition matrix to make the rows sum to 1
            transition_tensor = transition_tensor / transition_tensor.sum(dim=1, keepdim=True)

            # Compute stationary distribution
            stationary_distribution = compute_stationary_distribution(transition_tensor, iteration=50)
            stationary_distribution = stationary_distribution.cpu().detach()
            # Convert to NumPy
            stationary_distribution = stationary_distribution.numpy()
            
            # Store the tensor in the dictionary
            stationary_distributions[(home_block, traveler_type)][time_step] = stationary_distribution
    return stationary_distributions, facility_to_index 


def create_tm_sd_for_all_homeblock_travelertype_pair(urbano_trip_volumn_file: str,
                                                     TotalBlocks_count: int,
                                                     TotalTravelerTypeCount :int=8):
    """
    Comprehensive method to create transition_matrices and stationary_distributions.
    transition_matrices is in the format of: {(home_block, traveler_type): transition_matrix}.
    stationary_distributions is in the format of: {(home_block, traveler_type): stationary_distribution}.
    """                                                 
    stationary_distributions, facility_to_index = create_tm_sd_for_exist_homeblock_travelertype_pair(urbano_trip_volumn_file,
                                                                                                     TotalBlocks_count)
    exist_homeblock_travelertype_pairs = list(stationary_distributions.keys())
    # Get all TravelerHomeBlock and TravelerType pairs
    all_HomeBlock_TravelerType_pairs = list(product(range(TotalBlocks_count), range(1, TotalTravelerTypeCount+1)))
    # Get non existing TravelerHomeBlock and TravelerType pairs
    non_exist_homeblock_travelertype_pairs = list(set(all_HomeBlock_TravelerType_pairs)-set(exist_homeblock_travelertype_pairs))
    time_steps = [time_step for time_step in range(1,7)]

    # Create stationary distribution for non existing TravelerHomeBlock and TravelerType pairs
    sd_for_non_exist_pair = create_sd_for_non_exist_homeblock_travelertype_pair(len(facility_to_index))
    # Move to CPU and detach
    sd_for_non_exist_pair = sd_for_non_exist_pair.cpu().detach()
    # Convert to NumPy
    sd_for_non_exist_pair = sd_for_non_exist_pair.numpy()
    sds_for_non_exist_pair = {time_step: sd_for_non_exist_pair for time_step in time_steps}
    # Store the sd and tm of non_exist_homeblock_travelertype_pairs in the dictionary
    for homeblock_travelertype_pair in non_exist_homeblock_travelertype_pairs:
        stationary_distributions[homeblock_travelertype_pair] = sds_for_non_exist_pair
    return stationary_distributions, facility_to_index