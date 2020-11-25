# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime

def clean(df):
    
    def timer(start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
    beginningtime = timer()
    # Check if the the processed data frame is the training set
    if 'booking_bool' in df.columns:
        print("Okay, we're cleaning the training set boys.\n")
        #fill NaNs in 'gross_bookings_usd with 0.
        df.loc[df['gross_bookings_usd'].isnull(), 'gross_bookings_usd'] = 0
        # Add an additional column called 'score' to the training set, hereafter used as the target value
        df['score'] = 4 * df.booking_bool + df.click_bool
    else:
        print("Okay, we're cleaning the test set boys.\n")
        
    
    # Convert datetime to year and month features 
    print("Convert date time column")
    df['date_time'] = df['date_time'].apply(lambda x: pd.Timestamp(x))
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    del df['date_time']
    
    #Replace NaN with zeros > sparse matrix will leave these out
           #SHOULD THESE BE ZERO OR MEAN: #Prop_review_score, prop_location_score2
           #srch_query_affinity_score NA's to -350 as it is log of probability
    print("Fill NaN's with zeros or averages")
    start_time = timer()
    
    df.fillna({'visitor_hist_starrating': 0,'visitor_hist_adr_usd': 0, 'prop_review_score': 0, 'prop_location_score2': 0,
              'srch_query_affinity_score': -350, 'orig_destination_distance': df.orig_destination_distance.mean()}, inplace=True)
  
    #Simultaneously competetitor columns to fill with zeros
    print("Fill NaN's for competitors with zeros")
    
    if 'booking_bool' in df.columns:
        comp_descr = {i: 0 for i in df.columns[27:51]}
        df.fillna(comp_descr, inplace=True)
    else:
        comp_descr = {i: 0 for i in df.columns[26:50]}
        df.fillna(comp_descr, inplace=True)
    timer(start_time)
    
    print("Start feature extraction")
    
    #Create log price feature 
    df['price_usd'].replace({0:1}, inplace=True)
    df['log_price_usd'] = np.log(df.price_usd)

    #Differenced historical data
    df['starrating_diff'] = np.abs(df['visitor_hist_starrating'] - df['prop_starrating'])
    df['usd_diff'] = np.abs(df['visitor_hist_adr_usd'] - df['price_usd'])
    del df['price_usd']
    
    # Normalize features
    def normalize_wrt_column(df,column):
        norm_columns1 = ['srch_id','site_id','month','srch_length_of_stay']
        norm_columns2 = ['prop_country_id','prop_id','srch_destination_id']
        set1 = ['prop_starrating','prop_review_score', 'prop_location_score1', 'prop_location_score2',
                'prop_log_historical_price', 'log_price_usd', 'orig_destination_distance']
        set12 = set1+['srch_length_of_stay', 'srch_booking_window','srch_adults_count', 'srch_children_count',
                'srch_room_count','srch_query_affinity_score']
        standardscaler = StandardScaler()
        ids = list(set(df.loc[:,column]))
        if column in norm_columns1:
            set1.insert(0,str(column))
            df = df.loc[:,set1]
            for i in ids:
                df.loc[df.loc[:,column]==i,set1] = standardscaler.fit_transform(df.loc[df.loc[:,column]==i,set1])
        if column in norm_columns2:
            set12.insert(0,str(column))
            df = df.loc[:,set12]
            for i in ids:
                df.loc[df.loc[:,column]==i,set12] = standardscaler.fit_transform(df.loc[df.loc[:,column]==i,set12])
        return df
    
    print('Normalize wrt to site_id')
    df_site = normalize_wrt_column(df,'site_id')
    df_site = df_site.add_suffix('_wrt_site_id')
                                 
    print('Normalize wrt to prop_country_id')
    df_propc = normalize_wrt_column(df,'prop_country_id')
    df_propc = df_propc.add_suffix('_wrt_prop_country_id')
                                 
    print('Normalize wrt to srch_destination_id')
    df_dest = normalize_wrt_column(df,'srch_destination_id')
    df_dest = df_dest.add_suffix('_wrt_srch_destination_id')
    
    print('Normalize wrt to month')
    df_month = normalize_wrt_column(df,'month')
    df_month = df_month.add_suffix('_wrt_month')
    
    print('Normalize wrt to srch_length_of_stay')
    df_length = normalize_wrt_column(df,'srch_length_of_stay')
    df_length = df_length.add_suffix('_wrt_srch_length_of_stay')
                        
    # Create new features for the properties based on the numeric values
    def numeric_transform(df):
        for var in ['prop_starrating', 'prop_review_score', 'log_price_usd',
                    'prop_log_historical_price', 'prop_location_score1']:
            print('Adding mean, median and std of: ',var)  # For progression purposes
            selector = pd.DataFrame(df.groupby('srch_id')[str(var)].mean())
            selector.columns = ['mean_' + str(var)]
            df = pd.merge(df, selector, how='left', on='srch_id')

            selector = pd.DataFrame(df.groupby('srch_id')[str(var)].median())
            selector.columns = ['median_' + str(var)]
            df = pd.merge(df, selector, how='left', on='srch_id')

            selector = pd.DataFrame(df.groupby('srch_id')[str(var)].std())
            selector.columns = ['std_' + str(var)]
            df = pd.merge(df, selector, how='left', on='srch_id')

        return df
    start_time = timer()
    df = numeric_transform(df)
    timer(start_time)
                                 
    #Concatenate normalization frames
    print('Concatenating site normalizations')
    df_norm = pd.concat([df,df_site.iloc[:,1:len(df_site.columns)]],axis=1)
    
    print('Concatenating prop_country normalizations')
    df_norm = pd.concat([df_norm,df_propc.iloc[:,1:len(df_propc.columns)]],axis=1)
    
    print('Concatenating srch_destination normalizations')
    df_norm = pd.concat([df_norm,df_dest.iloc[:,1:len(df_dest.columns)]],axis=1)
    
    print('Concatenating month normalizations')
    df_norm = pd.concat([df_norm,df_month.iloc[:,1:len(df_month.columns)]],axis=1)
    
    print('Concatenating srch_length normalizations')
    df_norm = pd.concat([df_norm,df_length.iloc[:,1:len(df_length.columns)]],axis=1)
    
    timer(beginningtime)
    return df_norm

#Define mooi timertj amattie
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2))) 
        
    #Define click_book_rate constructor a matti
def click_book_rates(df):
    #Create dataframe of click and booking rates per property
    if 'booking_bool' in df.columns:
        print('Correct dataset, start computing click and book rates')

        counts = pd.DataFrame(df['prop_id'].value_counts(sort=False).reset_index())
        counts.columns = ['prop_id', 'counts']

        clicks = pd.DataFrame(df.groupby('prop_id')['click_bool'].sum().reset_index())
        clicks.columns = ['prop_id', 'clicks']

        bookings = pd.DataFrame(df.groupby('prop_id')['booking_bool'].sum().reset_index())
        bookings.columns = ['prop_id', 'bookings']

        temp = pd.merge(counts, clicks, left_on='prop_id', right_on='prop_id')
        df_rates = pd.merge(temp, bookings, left_on='prop_id', right_on='prop_id')

        df_rates['click_rate'] = df_rates['clicks'] / df_rates['counts']
        df_rates['booking_rate'] = df_rates['bookings'] / df_rates['counts']
        df_rates.drop(columns=['counts', 'clicks', 'bookings'], axis=1, inplace=True)

        return df_rates

    else:
        return print('Incorrect dataset, provide training set')
    