# Rust 2.5D ThetaStar Bidirectional Search

A fast 2.5D ThetaStar implementation written in Rust with Python Bindings in PyO3. 2.5D search includes all directions available in the xy-direction plus two directions to level change z planes from the current position (+/- z). The level change directions are penalized using a `z_penalty`, since raising/lowering altitude can be demanding on energy use and time constraints. Furtermore, this version supports valid start/goal search if the start/goal inputted is initially invalid. The search algorithm will attempt to find a valid start/goal to check for a path. 

## Installation
First install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

In a python virtual environment execute the following:

```
pip install maturin numpy
```

Build the release binary and link to python environment with Maturin

```
maturin develop --release
```

## Python Interface with PyO3

Create Planning Class Object

```python
import theta_star_bi_rust as tsb

world_map = ... # 3D Grid List
dims = ... # dimensions of world map 3D Tuple
is_theta_star = ... # Use Theta* or A*
planner = tsb.ThetaStarBi(world_map, dims, is_theta_star)
```

Call planning search function with parameters
``` 
start: Start in NED (North-East-Down)
goal: Goal in NED
altitude_gap: altitude gap between each z plane in the world map
min_altitude: minimum altitude -D, so positive altitude is above the ground
z_penalty: penalty for altitude changes
is_min_support: at minimum obstacle buffer is buffering obstacles
centering_offset: centering offset for the map at (0, 0)
free_space_search_max_range: if start/goal is not in free space, search for free space within this range
```

Call function

```python
final_path, context = planner.search(start, goal, altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, free_space_search_max_range)
```

Output
- `final_path` is a list of 3D tuples in NED
- `context` is a concatentated String with messages delimited by `\n`. There are two types of messages:
    - Invalid Goal: `"IG: x y z"` Two cases: The first case is if the inputted initial goal is deemed invalid and `is_min_support` is set to `true`, the search will attempt to find a valid goal in range `free_space_search_max_range`. For each voxel cell it deems not in a freespace, it will create an invalid goal message and append to `context`. The second case occurs if the path planning was executed but a path could not be found. In this case an invalid goal message is created for the current goal set.
    - Goal Found: `"GF: x y z"` If a path is successfully found to a goal point this message will be appended to `context` indicating success. (Note: this message can be appended about multiple invalid goal messages if a valid goal is discovered)

## Testing

### Rust Unittests
```shell
cargo test
```

### Simple Test

```shell
python scripts/theta_star_bi_test.py
```

### Microsoft AirSim Neighborhood Environment Voxelized
Voxelized version of AirSimNH from Microsoft AirSim (see [releases](https://github.com/microsoft/AirSim/releases)). The map we use contains three z-planes at altitudes of `[-5, -10, -15]` meters each with a 2D map of size `(400, 400)`. Stacking these into one 2.5D map, the final dimensions are: `(400, 400, 3)`. This allows the planner to consider changing the altitude by 5 meters if needed for efficiency. 

Sample 1000 start and goal positions (potentially invalid i.e. inside of obstacles). Then find a path if one exists. If start or goal is invalid, then a valid start/goal is determined if possible. Sample output is provided below.

```shell
python scripts/nbd_paths_eval.py
```

Sample Output
```
...
Iter 998, Start: [-34.5  98.5  -5. ], Goal: [ 33.5 135.5  -5. ]
Time taken: 0.0008990764617919922
Iter 999, Start: [182.5 -12.5  -5. ], Goal: [153.5  11.5  -5. ]
Time taken: 0.00024080276489257812

Success rate: 0.998
Average time taken: 0.02304612946510315
Std dev time taken: 0.02241056682127938
Max time taken: 0.14485836029052734
Min time taken: 1.7881393432617188e-05
Total time taken: 23.04612946510315
Average Mean altitude: -6.712627274751523
Average Max altitude: -9.03807615230461
```


