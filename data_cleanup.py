import pandas as pd
import numpy as np

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

# clean up data
portfolio = portfolio.rename(columns={'id': 'offer_id', 
                                      'duration': 'offer_duration',
                                      'reward': 'offer_reward'})
portfolio['offer_duration_hours'] = portfolio['offer_duration']*24

# drop users with missing gender and income
profile_cleaned = profile.dropna(subset=['gender', 'income']).rename(
    columns={'id': 'user_id'})

# clarify the gender variable
gender_sub = {'F': 'Female', 'M': 'Male', 'O': 'Other'}
profile_cleaned['gender'] = profile_cleaned['gender'].apply(
    lambda x: ' '.join([gender_sub.get(i, i) for i in x.split()]))

# expand the column specifying offer id or transaction amount
transcript = pd.concat([transcript, 
                        pd.DataFrame.from_records(transcript.value.tolist())],
                        axis=1)
transcript['offer_id'].fillna(transcript['offer id'], inplace=True)
transcript = transcript.drop(columns=['offer id', 'value'])

# create user-offer data set
# drop records for users without demographic data 
# and records for transactions (not related to offers directly)
transcript_u = transcript[
    transcript.person.isin(profile_cleaned.user_id)
    ].sort_values(by=['person', 'time']).rename(
        columns={'person': 'user_id'}).dropna(subset=['offer_id'])

def get_offer_data(user_data, offer, time_received, current_offer_duration):
    
    '''
    Extract data on a given offer received by a particular user:
    whether and when the offer was viewed and/or completed
    
    Args:
        user_data: a subset of transactions for the specific user
        offer: specific offer id
        time_received: time at which the offer was sent
        current_offer_duration: the duration of the specific offer in hours
        
    Returns: data frame with information on the reception, viewing,
    and completion of a particular offer by a particular user
    '''
    
    # limit transactions to the period in which the offer was active
    offer_data = user_data[
        (user_data.time >= time_received) &
        (user_data.time <= time_received + current_offer_duration) &
        (user_data.offer_id == offer)]
    
    # check whether the offer was completed/redeemed in that period
    offer_completed = int('offer completed' in offer_data.event.tolist())
    
    # check whether the offer was viewed in that period
    offer_viewed = int('offer viewed' in offer_data.event.tolist())
    
    # capture when the offer was completed/viewed
    if offer_completed == 1:
        time_completed = offer_data[
            offer_data.event == 'offer completed'].time.iloc[0]
    else:
        time_completed = np.nan
        
    if offer_viewed == 1:
        time_viewed = offer_data[
            offer_data.event == 'offer viewed'].time.iloc[0]
    else:
        time_viewed = np.nan
        
    # check whether the offer was completed before viewing it
    if offer_completed == 1 & offer_viewed == 1:
        if time_completed < time_viewed:
            offer_viewed_before = 0
        else:
            offer_viewed_before = 1
    elif offer_completed == 1 & offer_viewed == 0:
        offer_viewed_before = 0
    # set to nan if offer not completed
    else:
        offer_viewed_before = np.nan
    
    # combine everything into a data frame            
    user_offer_data = pd.DataFrame(
        {'user_id': offer_data.user_id.unique().tolist(),
         'offer_id': [offer],
         'completed': [offer_completed],
         'viewed': [offer_viewed],
         'viewed_before': [offer_viewed_before],
         'time_received': [time_received],
         'time_viewed': [time_viewed],
         'time_completed': [time_completed],
         'offer_duration': [current_offer_duration]})
    
    return user_offer_data


def check_user_offers(user_data, offer):
    
    '''
    Check how many times a given offer was received by a particular user
    and extract data on offer viewing/completion
    
    Args:
        user_data: a subset of transactions for the specific user
        offer: specific offer id
        
    Returns: data frame with information on the reception, viewing,
    and completion of all offers with a particular id by a particular user
    '''

    # extract the duration of the offer in hours from offer metadata
    current_offer_duration = portfolio[
        portfolio.offer_id == offer].offer_duration_hours.item()
    
    # check how many times the user received the same offer
    received_times = user_data[(user_data.event == 'offer received') &
                               (user_data.offer_id == offer)].time.tolist()
    
    # extract offer data
    user_offer_data = [get_offer_data(user_data, offer,
                                      time_point,
                                      current_offer_duration) for
                       time_point in received_times]
    user_offer_data = pd.concat(user_offer_data)
    
    # how many times the same offer was received 
    user_offer_data['offer_count'] = len(received_times)
    user_offer_data['time_points'] = '.'.join(str(x) for x in received_times)
        
    return user_offer_data
        

def clean_offers_by_user(user):
    
    '''
    Get data on all offers received by a particular user
    
    Args:
        user: user id
        
    Returns: data frame with information on the reception, viewing,
    and completion of all offers received by a particular user
    '''

    # a data set of all records on offers received, viewed, or completed
    # by a particular user
    user_transactions = transcript_u[transcript_u.user_id == user]
    
    # the ids of all offers received by the user
    user_offers = user_transactions.offer_id.unique().tolist()
    
    # extract data on all offers received by this user
    user_offer_data = [check_user_offers(user_transactions, offer) for 
                       offer in user_offers]
    user_offer_data = pd.concat(user_offer_data)
    
    return user_offer_data

# create the user-offer data frame
offers_by_user = [clean_offers_by_user(user) for user in 
                  transcript_u.user_id.unique().tolist()]
offers_by_user = pd.concat(offers_by_user)

# check how quickly offers were viewed/completed        
offers_by_user['time_to_completion'] = offers_by_user[
    'time_completed'] - offers_by_user['time_received']
offers_by_user['time_to_viewing'] = offers_by_user[
    'time_viewed'] - offers_by_user['time_received']
offers_by_user[['time_to_completion', 'time_to_viewing']].hist(bins=100)
offers_by_user = offers_by_user.drop(columns=['time_to_completion',
                                              'time_to_viewing'])

# calculate the number of offers completed by the same user
# before receiving the current offer
offers_by_user = offers_by_user.sort_values(
    by=['user_id', 'offer_id', 'time_received']).reset_index(drop=True)
offers_by_user['same_offer_completed_before'] = offers_by_user.groupby(
    by=['user_id', 'offer_id'])['completed'].cumsum()
offers_by_user['same_offer_completed_before'] = offers_by_user[
    'same_offer_completed_before'] - offers_by_user['completed']
offers_by_user = offers_by_user.sort_values(
    by=['user_id', 'time_received']).reset_index(drop=True)
offers_by_user['any_offer_completed_before'] = offers_by_user.groupby(
    by=['user_id'])['completed'].cumsum()
offers_by_user['any_offer_completed_before'] = offers_by_user[
    'any_offer_completed_before'] - offers_by_user['completed']

# merge with user data
offers_by_user = pd.merge(offers_by_user, profile_cleaned)

# calculate amount of time in months between the start of one's membership
# and receiving the current offer
# assume test started on 8/1/2018
offers_by_user['test_date'] = '20180801' 
offers_by_user['test_date'] = pd.to_datetime(offers_by_user['test_date'], 
                                             format="%Y%m%d")

# calculate dates when offers were received
offers_by_user['offer_date'] = offers_by_user[
    'test_date'] + pd.to_timedelta(offers_by_user.time_received/24, unit='d')
offers_by_user = offers_by_user.drop(columns='test_date')

# calculate the difference between offer date and start of membership
offers_by_user['member_date'] = pd.to_datetime(
    offers_by_user['became_member_on'], format="%Y%m%d") 
offers_by_user['member_months'] = 12 * (
    offers_by_user.offer_date.dt.year - 
    offers_by_user.member_date.dt.year) + (
        offers_by_user.offer_date.dt.month - 
         offers_by_user.member_date.dt.month)
        
# join with metadata on offers
offers_by_user = pd.merge(offers_by_user, 
                          portfolio.drop(columns=['offer_duration',
                                                  'offer_duration_hours']), 
                          how='left')

# save data
offers_by_user.to_csv('data/cleaned_offer_data.csv', index=False)
