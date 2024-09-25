import numpy as np
import theta_star_bi_rust as tsb

if __name__ == "__main__":
    world_map = np.zeros((3, 3, 3), dtype=np.uint8)
    dims = tuple(world_map.shape)
    planner = tsb.ThetaStarBi(world_map.tolist(), dims, True)
    
    start = (0, 0, 0) # Start in NED
    goal = (2, 2, -2) # Goal in NED
    altitude_gap = 1 # altitude gap between each z plane in the world map
    min_altitude = 0 # minimum altitude -D, so positive altitude is above the ground
    z_penalty = 1 # penalty for moving up or down in the z direction
    is_min_support = True # at minimum obstacle buffer
    centering_offset = (0, 0) # centering offset for the map at (0, 0)
    free_space_search_max_range = 5 # if start/goal is not in free space, search for free space within this range
    final_path, context = planner.search(start, goal, altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, free_space_search_max_range)
    
    assert final_path == [(0.0, 0.0, -0.0), (1.0, 1.0, -0.0), (1.0, 1.0, -2.0), (2.0, 2.0, -2.0)]
    assert context == "GF: 2 2 -2\n"