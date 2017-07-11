# Automotive Machine Learning usecase 

The dataset aggregates data transmitted from a number of cars while they are driving around. 

## Goal

- Task 1: Predict where a person is heading to next (that is, where the next trip-end will be transmitted from) based on the time and the location of the trip-start.

- Task 2: For each received event, predict where a person is heading to next (that is, where the next trip-end will be transmitted from) based on the partial trajectories calculated for each event.
            
## Dataset description

The dataset contains the following variables:
- deviceId This is a unique identifier for the car transmitting the data.
- eventId This variable is not required for the analysis below.
- eventTime This is the the unix timestamp for when the event was transmitted.
- lat This is the latitude of the vehicle at the time of transmission.
- lon This is the longitude of the vehicle at the time of transmission.
- eventType This categorical variable can take the values ‘trip-start’, ‘trip-end’ as well as others. If the value is ‘trip-start’ the transmission occurred at the beginning of a trip, and similarly for ‘trip-end’. The other events occurred in between a trip-end and trip-start.


## Usage
```
Usage: process.py [options]

Options:
  -h, --help   show this help message and exit
  --data=DATA  The src dataset. Default:
               ./testDataset/testDataset/part-00000-711fabb0-5efc-4d83-afad-
               0e03a3156794.snappy.parquet.
```
