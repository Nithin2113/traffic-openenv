\# Traffic Optimization OpenEnv



\## Overview

This project simulates a traffic signal control system where an AI agent optimizes traffic flow at a 4-way intersection.



\## Environment

\- 4 directions: north, south, east, west

\- Signal control: NS or EW

\- Objective: minimize congestion



\## API

\- POST /reset → initialize environment

\- POST /step → take action

\- GET /state → current state



\## Actions

\- 0 → keep signal

\- 1 → switch signal



\## Tasks

\- Easy → low traffic

\- Medium → moderate traffic

\- Hard → high congestion



\## Reward

\- Penalizes high congestion

\- Rewards improvement

\- Penalizes unnecessary switching



\## Setup



```bash

docker build -t traffic-env .

docker run -p 8000:8000 traffic-env

