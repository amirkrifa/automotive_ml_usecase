# Automotive Machine Learning usecase 

The dataset aggregates data transmitted from a number of cars while they are driving around. 

## Dataset description

The dataset contains the following variables:
- deviceId This is a unique identifier for the car transmitting the data.
- eventId This variable is not required for the analysis below.
- eventTime This is the the unix timestamp for when the event was transmitted.
- lat This is the latitude of the vehicle at the time of transmission.
- lon This is the longitude of the vehicle at the time of transmission.
- eventType This categorical variable can take the values ‘trip-start’, ‘trip-end’ as well as others. If the value is ‘trip-start’ the transmission occurred at the beginning of a trip, and similarly for ‘trip-end’. The other events occurred in between a trip-end and trip-start.


## Goal

Predict the destination of cars trips based on partial trajectories.

## Usage
```
Usage: process.py [options]

Options:
  -h, --help   show this help message and exit
  --data=DATA  The src dataset. Default:
               ./testDataset/testDataset/part-00000-711fabb0-5efc-4d83-afad-
               0e03a3156794.snappy.parquet.
```
