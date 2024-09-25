import os
import time

import numpy as np

from convert_binvox import ConvertBinvox
import theta_star_bi_rust as tsb

def create_3D_stack(altitudes, obs_map_reshaped):
    stacked_array = np.transpose(cb.get_slice(obs_map_reshaped, 
                                altitude=altitudes[0], 
                                tolerance=0), (1, 0))
    for altitude in altitudes[1:]:
        stacked_array = np.dstack((stacked_array, 
                                   np.transpose(cb.get_slice(obs_map_reshaped, 
                                   altitude=altitude, 
                                   tolerance=0), (1, 0))))
    return stacked_array
    

if __name__ == '__main__':
    np.random.seed(0)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cb = ConvertBinvox()
    obs_map, width, height, _ = cb.to_map(cb.load_binvox(os.path.join(parent_dir, 'scripts/map_data', 'voxelgrid_400_400_50.binvox')))
    obs_map_reshaped = obs_map.reshape((width, height))
    world_map_3D = create_3D_stack([5, 10, 15], obs_map_reshaped).astype(np.uint8)
    assert world_map_3D.shape == (400, 400, 3)
    planner = tsb.ThetaStarBi(world_map_3D.tolist(), tuple(world_map_3D.shape), True)
    
    # start = (0, 0, -5) # Start in NED
    # goal = (130.5, 130.5, -5) # Goal in NED
    altitude_gap = 5 # altitude gap between each z plane in the world map
    min_altitude = 5 # minimum altitude -D, so positive altitude is above the ground
    z_penalty = 1.5 # penalty for moving up or down in the z direction
    is_min_support = True # at minimum obstacle buffer
    centering_offset = (-199.5, -199.5) # centering offset for the map at (0, 0)
    free_space_search_max_range = 5 # if start/goal is not in free space, search for free space within this range
    
    num_points = 1000
    start_xs = np.random.randint(0, 400, num_points)
    start_ys = np.random.randint(0, 400, num_points)
    start_zs = np.full(num_points, -5)
    starts = np.dstack((start_xs, start_ys, start_zs))[0].astype(np.float64)
    starts[:, :2] += centering_offset
    
    goal_xs = np.random.randint(0, 400, num_points)
    goal_ys = np.random.randint(0, 400, num_points)
    goal_zs = np.full(num_points, -5)
    goals = np.dstack((goal_xs, goal_ys, goal_zs))[0].astype(np.float64)
    goals[:, :2] += centering_offset

    time_taken = []
    path_success = 0
    avg_altitudes = []
    max_altitudes = []
    for i, (start, goal) in enumerate(zip(starts, goals)):
        print(f"Iter {i}, Start: {start}, Goal: {goal}")
        start_time = time.time()
        final_path, context = planner.search(tuple(start), tuple(goal), altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, free_space_search_max_range)
        time_taken.append(time.time() - start_time)
        if final_path:
            path_success += 1
            final_path = np.array(final_path)
            avg_altitudes.append(np.mean(final_path[:, 2]))
            max_altitudes.append(np.min(final_path[:, 2]))
        print(f"Time taken: {time_taken[-1]}")
    print()
    print(f"Success rate: {path_success/num_points}")
    print(f"Average time taken: {np.mean(time_taken)}")
    print(f"Std dev time taken: {np.std(time_taken)}")
    print(f"Max time taken: {np.max(time_taken)}")
    print(f"Min time taken: {np.min(time_taken)}")
    print(f"Total time taken: {np.sum(time_taken)}")
    print(f"Average Mean altitude: {np.mean(avg_altitudes)}")
    print(f"Average Max altitude: {np.mean(max_altitudes)}")
    
    
    
    
    