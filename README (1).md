# UAV Path Planning using a Genetic Algorithm

This is my AI course assignment (BSCS 5, Bahria University, Lahore Campus).

## What this project does
We have 3 drones (UAVs) and a disaster area on a map. Some parts of the map are more important than others, some parts are dangerous, and some parts are no-fly zones. The drones must fly over the important parts and find the hidden people, without using too much energy or time.

I used a Genetic Algorithm to find good flight paths. It starts with random paths and slowly makes them better, a bit like survival of the fittest.

## Files
- uav_ga.py : the main code
- sensitivity.py : tests what happens when I change population size, mutation rate and tournament size
- requirements.txt : the Python libraries you need (numpy and matplotlib)
- convergence.png : graph of how the fitness improves, compared with random search
- paths_evolution.png : the drone paths and coverage maps at different generations
- sensitivity.png : graph of the sensitivity test

## How to run
1. Install the libraries:
   pip install -r requirements.txt
2. Run the main code:
   python uav_ga.py
3. Run the sensitivity test:
   python sensitivity.py

## How it works
- Each drone has 6 waypoints. Each waypoint has an x position, a y position, a speed and a sensor power.
- The fitness score goes up when the drones cover more of the important area and find more targets. It goes down when they use more energy, take more time or fly through dangerous areas.
- Parents are picked with tournament selection.
- Children are made by swapping parts of the paths or mixing the two parents.
- Mutation moves waypoints a little, sometimes jumps them to a random place, and changes speed and sensor power.
- The best 2 paths are always kept for the next generation.
- If a waypoint is inside a no-fly zone, it is moved out. If a route is too long, it is shortened. A penalty is added if a problem is still left.
- The program stops after a fixed number of generations.
