#!/usr/bin/env python
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from collections import Counter
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

MAX_VALID_LAT, MIN_VALID_LAT =  90, -90
MAX_VALID_LON, MIN_VALID_LON =  180, -180

# Relevant paper: https://arxiv.org/pdf/1509.05257.pdf

# calculate distance among two gps points
def haversine(coord1,coord2):
    sin = math.sin
    cos = math.cos
    atan2 = math.atan2
    sqrt = math.sqrt
    lon1,lat1=coord1
    lon2,lat2=coord2
    R=6371000 #metres
    phi1=lat1 * (3.1415 / 180)
    phi2=lat2 * (3.1415 / 180)
    Dphi= phi2 - phi1
    Dlambda = (lon2 -lon1) *  (3.1415 / 180)
    a = sin(Dphi / 2) ** 2 + cos(phi1)*cos(phi2) *sin(Dlambda/2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a))
    d = R*c
    return d

def heatmap(trips, nrbins = 200):
    hist = np.zeros((nrbins, nrbins))
    for trip in trips:
        # Compute the histogram with the longitude and latitude data as a source
        hist_new, _, _  = np.histogram2d(x = np.array([x.lat for x in trip]),
                                         y = np.array([x.lon for x in trip]),
                                         bins = nrbins,
                                         )

        # Add the new counts to the previous counts
        hist = hist + hist_new
    # We consider the counts on a logarithmic scale
    img = np.log(hist[::-1,:] + 1)
    # Plot the counts
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Trip density')
    plt.savefig('trips_density.png')

def data_preprocessing(dataset_path):
    # load data
    table = pq.read_table(dataset_path)
    df = table.to_pandas()

    # remove duplicates according to a subset of the columns
    df = df.drop_duplicates(['deviceId', 'eventTime',  'eventType'])

    # remove rows with invalid lat values
    df = df.loc[(df['lat'] > MIN_VALID_LAT) & (df['lat'] < MAX_VALID_LAT)]

    # remove rows with invalid lon values
    df = df.loc[(df['lon'] > MIN_VALID_LON) & (df['lat'] < MAX_VALID_LON)]

    # sort element by evenetTime
    df = df.sort_values(by='eventTime', ascending=True)


    # common analysis
    distinct_events = df.eventType.unique()
    min_lat = df.lat.min()
    max_lat = df.lat.max()
    mid_lat = df.lat.median()
    mid_lon = df.lon.median()
    min_lon = df.lon.min()
    max_lon = df.lon.max()

    print 'Distrinct events types available: ', distinct_events
    print 'Min, Max, Mid lon: ', min_lon, max_lon, mid_lon
    print 'Min, Max, Mid lat: ', min_lat, max_lat, mid_lat

    grouped_device_id = df.groupby(['deviceId'])
    print 'Nbr of initial trips: ', len(grouped_device_id)

    # Parse devices trips and discard invalid ones
    trips = {}
    for device_name, device_rows in grouped_device_id:
        # group device rows by time
        rows_by_time = {}
        for row in device_rows.iterrows():
            if row[1].eventTime in rows_by_time:
                rows_by_time[row[1].eventTime].append(row[1])
            else:
                rows_by_time[row[1].eventTime] = [row[1]]

        # discard trips with less than 2 distinct timestamps
        if len(rows_by_time.keys()) <=1:
            continue

        sorted_rows = sorted(rows_by_time.iteritems(), key=lambda x: x[0])

        # iterate on events and check trip validity
        new_trip = []
        trip_start_event = None
        trip_end_event = None
        event_types = []
        for event_time, rows in sorted_rows:
            row = None
            if len(rows) > 1:
                rows = sorted(rows, key=lambda x:x.eventTime)
            for row in rows:
                if row.eventType == 'trip-end':
                    if not trip_end_event is None:
                        if row.eventTime >= trip_end_event.eventTime:
                            trip_end_event = row
                    else:
                        trip_end_event = row
                elif row.eventType == 'trip-start':
                    if not trip_start_event is None:
                        if row.eventTime < trip_start_event.eventTime:
                            trip_start_event = row
                    else:
                        trip_start_event = row
                else:
                    new_trip.append(row)
                    event_types.append(row.eventType)

        # skip trips without trip-end or trip-start
        if trip_end_event is None or trip_start_event is None:
            continue
        if not new_trip and trip_start_event.eventTime == trip_end_event.eventTime:
            continue

        new_trip = [trip_start_event] + new_trip + [trip_end_event]
        # sort by time
        new_trip = sorted(new_trip, key=lambda x: x.eventTime)
        trips[device_name] = {'trip': new_trip,
                              'trip_nbr_updates': len(new_trip),
                              'dest_travel_delay': (new_trip[-1].eventTime - new_trip[0].eventTime).seconds,
                              'dest_haversine_distance': haversine((new_trip[0].lon, new_trip[0].lat), (new_trip[-1].lon, new_trip[-1].lat)),
                              'dest_euclidean_distance': distance.euclidean((new_trip[0].lon, new_trip[0].lat), (new_trip[-1].lon, new_trip[-1].lat)),
                              # Shape complexity: the ratio between the (Euclidean) traveled distance and
                              # the Haversine distance between the first and the last GPS location.
                              'dest_shape_complexity': distance.euclidean((new_trip[0].lon, new_trip[0].lat), (new_trip[-1].lon, new_trip[-1].lat))/haversine((new_trip[0].lon, new_trip[0].lat), (new_trip[-1].lon, new_trip[-1].lat)),
                              'trip_start_time': new_trip[0].eventTime,
                              'src_lon': new_trip[0].lon,
                              'src_lat': new_trip[0].lat,
                              'dest_lon': new_trip[-1].lon,
                              'dest_lat': new_trip[-1].lat,
                              }

    print 'Nbr of trips after discarding invalid ones: ', len(trips)

    # Nbr of events per trip
    trips_nbr_events = [len(x) for x in trips]
    nbr_events_dist = Counter(trips_nbr_events)
    print 'Trips nbr of events: ', nbr_events_dist.viewitems()

    # trips delays
    trips_delays = [(x['trip'][-1].eventTime - x['trip'][0].eventTime).seconds/60 for x in trips.values()]
    delays_dist = Counter(trips_delays)
    print 'Trips delays (min): ', delays_dist.viewitems()

    # trips distances
    trips_distances = [haversine((x['trip'][0].lon, x['trip'][0].lat), (x['trip'][-1].lon, x['trip'][-1].lat)) for x in trips.values()]
    distances_dist = Counter(trips_distances)
    print 'Trips distances: ', distances_dist.viewitems()

    # plot trips hitmap
    heatmap([x['trip'] for x in trips.values()])

    # Dataset & features generation
    columns = ['device_name', 'dest_lat', 'dest_lon', 'dest_shape_complexity', 'dest_euclidean_distance',
               'dest_haversine_distance', 'dest_travel_delay', 'trip_nbr_updates', 'src_lat', 'src_lon'
               ]
    rows = []
    for device_name, trip_details in trips.iteritems():
        dest_lat = trip_details['dest_lat']
        dest_lon = trip_details['dest_lon']
        src_lat = trip_details['src_lat']
        src_lon = trip_details['src_lon']
        dest_shape_complexity = trip_details['dest_shape_complexity']
        dest_euclidean_distance = trip_details['dest_euclidean_distance']
        dest_haversine_distance = trip_details['dest_haversine_distance']
        travel_delay = (trip_details['trip'][-1].eventTime - trip_details['trip_start_time']).seconds
        trip_nbr_updates = trip_details['trip_nbr_updates']
        rows.append([device_name, dest_lat, dest_lon, dest_shape_complexity, dest_euclidean_distance,
                     dest_haversine_distance, travel_delay, trip_nbr_updates, src_lat, src_lon
                     ]
                    )
    dataset = pd.DataFrame(data=rows, columns=columns)
    return dataset

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--data', default='./testDataset/testDataset/part-00000-711fabb0-5efc-4d83-afad-0e03a3156794.snappy.parquet', help='The src dataset. Default: %default.')

    options, args_left = parser.parse_args()

    # data pre-processing
    dataset = data_preprocessing(options.data)

    # dump dataset
    dataset.to_csv('dataset.csv')

    print 'Training ...'

    rng = np.random.RandomState(1)
    lat_ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                                    n_estimators=300,
                                    random_state=rng
                                    )
    lon_ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                                    n_estimators=300,
                                    random_state=rng
                                    )


    y_lat = dataset[['dest_lat']]
    y_lon = dataset[['dest_lon']]
    X = dataset[['dest_shape_complexity', 'dest_euclidean_distance',
                  'dest_haversine_distance', 'dest_travel_delay', 'trip_nbr_updates', 'src_lat', 'src_lon'

                ] ]

    X = X.as_matrix()
    X = preprocessing.scale(X)

    X_train, X_test, y_lat_train, y_lat_test = train_test_split(X, y_lat.as_matrix(), test_size=0.33, random_state=42)
    X_train, X_test, y_lon_train, y_lon_test = train_test_split(X, y_lon.as_matrix(), test_size=0.33, random_state=42)

    lat_ada_reg.fit(X_train, y_lat_train)
    lon_ada_reg.fit(X_train, y_lon_train)

    # Predict
    y_lat_pred = lat_ada_reg.predict(X_test)
    y_lon_pred = lon_ada_reg.predict(X_test)

    # MSE
    lat_error = mean_squared_error([x[0] for x in y_lat_test], y_lat_pred)
    lon_error = mean_squared_error([x[0] for x in y_lon_test], y_lon_pred)

    lat_scores = cross_val_score(lat_ada_reg, X, y_lat, cv=5, scoring='neg_mean_squared_error')
    lon_scores = cross_val_score(lon_ada_reg, X, y_lat, cv=5, scoring='neg_mean_squared_error')

    print 'lat mse: ', lat_error
    print 'lon mse: ', lon_error
    print 'Lat Cross validation mse scores: ', lat_scores
    print 'Average Lon mse: ', np.mean(lat_scores)
    print 'Lon Cross validation mse scores: ', lat_scores
    print 'Average Lon mse: ', np.mean(lat_scores)

    print 'Done'

if __name__ == '__main__':
    main()