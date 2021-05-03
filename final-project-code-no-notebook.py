######### FINAL PROJECT CODE ##########
"""
This code was originally done in a Jupyter Notebook 
but has been cleaned up to be easier to read and 
remove unimportant comments, markdown, and debugging.
"""
#######################################

from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.utilities import *
import networkx as nx
import numpy as np
import pandas as pd

#######################################
"""
This cell creates the transmission networks.  If you run it in a notebook
I'd suggest running it in its own cell because it's a little slow.
"""
#######################################

population_size = 10000

# supplying an array of distancing scales will allow the network generator
# to return multiple versions of the graph with different levels of social
# distancing and will also return the base graph as well.
distancing = [12, 6, 1]
demographic_graphs, individual_age_groups, households = generate_demographic_contact_network(
    N=population_size,
    demographic_data=household_country_data('US'),
    distancing_scales=distancing
)

# WARNING: These take a little while to generate
G_base = demographic_graphs['baseline']
G_distancing = demographic_graphs['distancingScale12']
G_strong_distancing = demographic_graphs['distancingScale6']
G_quarantine = demographic_graphs['distancingScale1']

#######################################
"""
This code plots the degree distribution of the four different
graphs.  Since the graphs are random it will not be
quite the same as the graphs in the report
"""
#######################################

network_info(G_base, 'Baseline', plot=True)
network_info(G_distancing, 'Distancing', plot=True)
network_info(G_strong_distancing, 'Strong Distancing', plot=True)
network_info(G_quarantine, 'Quarantine', plot=True)

#######################################
"""
This code sets up the model parameters and creates several dictionaries
of checkpoints.  Checkpoints are used to change out model parameters at
certain timesteps (in the 't' list).  For a full explanation see the
SEIRS plus docs.  They are mostly very good.

I copied most of the parameters from the SEIRS plus documentation since
they are supposed to be reasonable approximations for the real COVID
parameters.  I don't really know why they did 1/(1/GAMMA), or why I
copied it...
"""
#######################################

SIGMA = 1/5.2 # Sigma is the inverse of the latent period (average time till infectiousness)
GAMMA = 1/10 # Gamma recovery rate
MU_I = 0.002 # Death rate for the hospitalized

R0 = 2.5 # Average number infected by first infectious person
BETA = 1/(1/GAMMA) * R0 # infectiousness
BETA_Q = 0.5 * BETA # infectiousness while in quarantine

P = 0.2 # probability of global interaction
Q = 0.05 # p but for quarantined individuals
INITI = 50 # initial number of infected individuals

# Checkpoints for single day violation
checkpoints_one_day_baseline = {
    't': [5, 25, 50, 180, 181, 185],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, P, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}

checkpoints_one_day_high_p = {
    't': [5, 25, 50, 180, 181, 185],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, 1, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}

checkpoints_one_day_low_p = {
    't': [5, 25, 50, 180, 181, 185],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, 0.5 * P, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}


### Checkpoints for week long violation
checkpoints_one_week_baseline = {
    't': [5, 25, 50, 180, 187, 192],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, P, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}

checkpoints_one_week_high_p = {
    't': [5, 25, 50, 180, 187, 192],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, 1, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}

checkpoints_one_week_low_p = {
    't': [5, 25, 50, 180, 187, 192],
    'G': [G_strong_distancing, G_quarantine, 
        G_distancing, G_base, G_distancing, G_distancing],
    'p': [0.4 * P, 0.2 * P, 0.6 * P, 0.5 * P, 0.6 * P, 0.6 * P],
    'q': [Q, 0, Q, Q, Q, Q],
    'theta_E': [0.0025, 0.004, 0.0066, 0.0066, 0.01, 0.0066],
    'theta_I': [0.005, 0.008, 0.013, 0.013, 0.033, 0.013],
    'phi_E': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'phi_I': [0.5, 0.5, 0.75, 0.75, 0.75, 0.75]    
}

# This was mentioned in the Karaivanov paper, I think.
single_IFR = 0.0037

###################################
"""
A utility to reset the epidemic model.  Otherwise it
continues where it left off.  There might be an easier
way to do this.
"""
###################################

def reset_model():
    return SEIRSNetworkModel(
    G = G_base,
    beta = BETA,
    sigma = SIGMA,
    gamma = GAMMA,
    mu_I = MU_I, # hospitalization fatality rate rate
    f=single_IFR, # fatality rate
    p = P,
    G_Q = G_quarantine,
    beta_Q = BETA_Q,
    sigma_Q = SIGMA,
    gamma_Q = GAMMA,
    mu_Q = MU_I,
    q = Q,
    initI = INITI
)

###################################
"""
Run the model for all 6 different cases and graph the results
"""
###################################

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_day_baseline)
model.figure_infections(
    vlines=checkpoints_one_day_baseline['t'], 
    ylim=0.1
)

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_day_high_p)
model.figure_infections(
    vlines=checkpoints_one_day_high_p['t'], 
    ylim=0.1
)

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_day_low_p)
model.figure_infections(
    vlines=checkpoints_one_day_low_p['t'], 
    ylim=0.1
)

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_week_baseline)
model.figure_infections(
    vlines=checkpoints_one_week_baseline['t'], 
    ylim=0.1
)

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_week_high_p)
model.figure_infections(
    vlines=checkpoints_one_week_high_p['t'],
    ylim=0.1
)

model = reset_model()
model.run(T=300, checkpoints=checkpoints_one_week_low_p)
model.figure_infections(
    vlines=checkpoints_one_week_low_p['t'],
    ylim=0.1
)


# I didn't include this graph in my project but this is what happens  #
# with no intervention.  The ylim should probably be set a lot higher #
model = reset_model()
model.run(T=300)
model.figure_infections(ylim=0.1)

####################################
"""
This section deals with flight data from the TSA.  It won't run
because it reads a file you almost certainly don't have.  There
is an HTML table in the source linked in the project bibliography.

I had trouble reading directly from the url, so I use the browser tools
to view the page source, pasted it into a .txt file and read it that way.

Then I do some boring preprocessing and plot it

The dates won't be correct since time has passed since I accessed the
site.  The code could be modified to filter out dates after 4/28/2021
though.
"""
#####################################

# extracting TSA data
tables = pd.read_html('tsa-html-data.txt')
# This goes up to 4/28/2021
# weird format because it counts down (starting at 4/28/2021 ...)
TSA_TRAVEL_DATA = tables[0]
TSA_TRAVEL_DATA['Date'] = pd.to_datetime(
    TSA_TRAVEL_DATA['Date'],
    infer_datetime_format=True
)


this_year = TSA_TRAVEL_DATA[
    TSA_TRAVEL_DATA['Date'].dt.year == 2021
].sort_values('Date')

last_year = TSA_TRAVEL_DATA[
    TSA_TRAVEL_DATA['Date'].dt.year == 2020
].sort_values('Date')
both_years = TSA_TRAVEL_DATA.sort_values('Date')

last_year.plot(x='Date', y=[
    '2021 Traveler Throughput', 
    '2020 Traveler Throughput', 
    '2019 Traveler Throughput'
])
this_year.plot(x='Date', y=[
    '2021 Traveler Throughput',
    '2020 Traveler Throughput', 
    '2019 Traveler Throughput'
])
both_years.plot(x='Date', y=[
    '2021 Traveler Throughput',
    '2020 Traveler Throughput',
    '2019 Traveler Throughput'
])

#################################
"""
This code applies 7-day averaging to the flight dates to make them
better match with the case and test data.
""""
#################################

# Note this will remove the first 6 dates since they can't be smoothed (we don't have enough prior data)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
def smooth_travel(travel_array):
    return np.array([
        np.mean(travel_array[i:i+7]) for i in range(len(travel_array) - 6)]
    )

smoothed2019 = smooth_travel(last_year['2019 Traveler Throughput'].to_numpy())
smoothed2020 = smooth_travel(last_year['2020 Traveler Throughput'].to_numpy())
smoothedDates = last_year['Date'].to_numpy()[6:]
print(len(smoothed2019), len(smoothed2020), len(smoothedDates))
smoothdf = pd.DataFrame(data={
    'date':smoothedDates, 
    '2019':smoothed2019, 
    '2020':smoothed2020, 
    'percent 2020 of 2019':smoothed2020/smoothed2019 * 100
})
smoothdf.plot(x='date', y=['2019', '2020'])
smoothdf.plot(x='date', y=['percent 2020 of 2019'])

##################################
"""
This section loads the data from Our World in Data.  The link for this
.csv file can be found from the page linked in the bibliography.

Again, accessing at a different date might complicate things.  The code
or data may need to be modified slightly to account for this.
"""
###################################

# read covid data
huge_dataset = pd.read_csv('owid-covid-data.csv')
us_data = huge_dataset[huge_dataset['iso_code'] == 'USA']
us_data['date'] = pd.to_datetime(us_data['date'], infer_datetime_format=True)
us_data = us_data[us_data['date'] >= smoothedDates[0]]
us_data = us_data[us_data['date'] <= smoothedDates[-1]]

###################################
"""
This code graphs and explores the COVID and travel data.
"""
###################################
cases, tests, pr = (
    us_data['new_cases_smoothed'].to_numpy(), 
    us_data['new_tests_smoothed'].to_numpy(), 
    us_data['positive_rate'].to_numpy()
)
combined = pd.DataFrame({
    'date':smoothedDates, 
    'travel':smoothed2020, 
    'new cases':cases, 
    'new tests': tests, 
    'positive rate': pr
})
combined.plot(x='date', y=['travel', 'new cases', 'new tests'])
combined.plot(
    x='date', 
    y='travel', 
    title='7-day Average of US Air Travel (millions)'
)
combined.plot(
    x='date', 
    y='new cases', 
    title='7-day Average new case count'
)
combined.plot(
    x='date', 
    y='new tests', 
    title='7-day Average new COVID tests (millions)'
)
combined.plot(
    x='date', 
    y='positive rate', 
    title='7-day Average rate of positive COVID tests'
)
print(f'correlation new tests and new cases: {np.corrcoef(tests, cases)}')
print(f'correlation travel and new cases: {np.corrcoef(smoothed2020, cases)}')
# I feel like lag should be a factor here, but I'm not really sure the best way to go about checking
# the checks I've tried all seem to reduce the correlation with travel, this might be right for all I know.
print('correlation 5 day lagged and starts near holidays travel and new cases:' + 
f'{np.corrcoef(smoothed2020[154:], cases[150:-4])}')

################### END ##############################