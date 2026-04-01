---
title: Traffic Optimization OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# Traffic Optimization OpenEnv

## Overview
This project implements a reinforcement learning environment for optimizing traffic flow at a four-way intersection. It simulates vehicle queues and signal control, enabling agents to learn policies that reduce congestion and waiting time.

The environment follows the OpenEnv specification and is designed for experimentation with adaptive traffic control strategies.

---

## Problem Statement
Urban traffic congestion leads to increased travel time, fuel consumption, and environmental impact. Traditional traffic signals operate on fixed timing and fail to adapt to dynamic traffic conditions.

This project models a system where an agent can dynamically control signal switching to improve traffic efficiency.

---

## Environment Design

### State
- Vehicle count in each direction: north, south, east, west  
- Current signal state: NS (north-south) or EW (east-west)  

### Actions
- `0`: Maintain current signal  
- `1`: Switch signal  

### Dynamics
- Traffic flow updates based on the active signal  
- Vehicle queues evolve dynamically at each step  

---

## Tasks

The environment provides three difficulty levels:

- Easy: Low traffic density  
- Medium: Moderate traffic conditions  
- Hard: High congestion scenario  

Each task evaluates performance under different traffic loads.

---

## Reward Function
The reward function is designed to encourage efficient traffic management:

- Penalizes higher vehicle wait times  
- Encourages reduction in queue length  
- Promotes balanced signal switching  

Objective: minimize congestion over time.

---

## Evaluation Metrics

Performance is evaluated using:

- Total accumulated reward  
- Average queue length  
- Traffic throughput efficiency  

Example output:

```
easy score: -300
medium score: -450
hard score: -500
```


---

## Implementation Details

- OpenEnv compliant environment  
- Implements step(), reset(), and state handling  
- Structured models for observations and rewards  
- Dockerized for reproducibility  
- Deployed on Hugging Face Spaces  

---

## Project Structure

```
app/
  env.py
  models.py
  tasks.py
inference.py
openenv.yaml
Dockerfile
README.md
requirements.txt
```


---

## Key Differentiators

- Real-world system simulation instead of a toy problem  
- Multi-level task evaluation (easy to hard scenarios)  
- Fully containerized and reproducible environment  
- Designed for integration with reinforcement learning agents  
- Clean OpenEnv compliance with structured modeling  

---

## Future Work

- Extend to multi-intersection traffic networks  
- Integrate reinforcement learning agents for adaptive signal control  
- Add real-time traffic visualization  
- Introduce stochastic traffic patterns  

---

## Conclusion

This project demonstrates how reinforcement learning environments can be applied to real-world infrastructure problems such as traffic optimization. It provides a strong foundation for building intelligent and adaptive traffic systems.
