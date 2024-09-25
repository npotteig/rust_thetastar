use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::f64::INFINITY;
use nalgebra::Vector3;
use pyo3::prelude::*;

struct Node{
    state: (usize, usize, usize),
    g: f64,
    h: f64,
    f: f64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.partial_cmp(&self.f).unwrap()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

#[pyclass]
pub struct ThetaStarBi {
    world_map: Vec<Vec<Vec<usize>>>,
    dims: (usize, usize, usize),
    is_theta_star: bool, // True if Theta* is used, False if A* is used
}

#[pymethods]
impl ThetaStarBi {
    #[new]
    pub fn new(world_map: Vec<Vec<Vec<usize>>>, dims: (usize, usize, usize), is_theta_star: bool) -> Self {
        if !ThetaStarBi::_validate_world_map(world_map.clone(), dims) {
            panic!("Invalid world_map dimensions");
        }
        ThetaStarBi { world_map, dims, is_theta_star }
    }

    #[setter]
    pub fn set_world_map(&mut self, world_map: Vec<Vec<Vec<usize>>>) {
        if !ThetaStarBi::_validate_world_map(world_map.clone(), self.dims) {
            panic!("Invalid world_map dimensions");
        }
        self.world_map = world_map;
    }


    pub fn search(&self, 
                start: (f64, f64, f64), // NED coordinates
                goal: (f64, f64, f64), // NED coordinates
                altitude_gap: usize, // altitude gap between each z plane in the world map
                min_altitude: usize, // minimum altitude -D, so positive altitude is above the ground
                z_penalty: f64, // Penalty for altitude raising
                is_min_support: bool, // at minimum obstacle buffer
                centering_offset: (f64, f64), // centering offset, assuming the world map is centered at (0, 0)
                freespace_search_max_range: usize // if start/goal is not in free space, search for free space within this range
            ) -> (Vec<(f64, f64, f64)>, String) {
        
        let mut transformed_start = ((start.0 - centering_offset.0).round() as usize, (start.1 - centering_offset.1).round() as usize, (((start.2 * -1.0) / altitude_gap as f64).round() * (altitude_gap as f64)) as usize);
        let mut transformed_goal = ((goal.0 - centering_offset.0).round() as usize, (goal.1 - centering_offset.1).round() as usize, (((goal.2 * -1.0) / altitude_gap as f64).round() * (altitude_gap as f64)) as usize);
        let mut new_start: bool = false;
        let mut context = String::new();

        if !self._freespace(&transformed_start, altitude_gap, min_altitude){
            match self._get_freespace(transformed_start, freespace_search_max_range, false, is_min_support, centering_offset, &mut context, altitude_gap, min_altitude).pop() {
                Some(start) => {
                    transformed_start = start;
                    new_start = true;
                },
                None => {
                    return (Vec::new(), String::new());
                }
            } 
        }

        if !self._freespace(&transformed_goal, altitude_gap, min_altitude){
            match self._get_freespace(transformed_goal, freespace_search_max_range, true, is_min_support, centering_offset, &mut context, altitude_gap, min_altitude).pop() {
                Some(goal) => {
                    transformed_goal = goal;
                },
                None => {
                    return (Vec::new(), context);
                }
            }
        }

        if transformed_start == transformed_goal {
            context.push_str(&format!(
                "GF: {} {} {}\n", 
                transformed_goal.0 as f64 + centering_offset.0, 
                transformed_goal.1 as f64 + centering_offset.1, 
                transformed_goal.2 as f64 * -1.0
            ));
            let final_path = vec![(transformed_goal.0 as f64 + centering_offset.0, transformed_goal.1 as f64 + centering_offset.1, transformed_goal.2 as f64 * -1.0)];
            return (final_path, context);
        }
        
        let mut path = self._search(transformed_start, transformed_goal, altitude_gap, min_altitude, z_penalty);
        if path.is_empty() {
            context.push_str(&format!(
                "IG: {} {} {}\n", 
                transformed_goal.0 as f64 + centering_offset.0, 
                transformed_goal.1 as f64 + centering_offset.1, 
                transformed_goal.2 as f64 * -1.0
            ));
            return (Vec::new(), context);
        }

        if new_start {
            path.insert(0, transformed_start);
        }

        context.push_str(&format!(
            "GF: {} {} {}\n", 
            transformed_goal.0 as f64 + centering_offset.0, 
            transformed_goal.1 as f64 + centering_offset.1, 
            transformed_goal.2 as f64 * -1.0
        ));

        let transformed_path: Vec<(f64, f64, f64)> = self._remove_collinear_points(&path).iter().map(|node| {
            (
                node.0 as f64 + centering_offset.0,
                node.1 as f64 + centering_offset.1,
                node.2 as f64 * -1.0,
            )
        }).collect();

        return (transformed_path, context);
    }

}

impl ThetaStarBi {
    fn _validate_world_map(world_map: Vec<Vec<Vec<usize>>>, dims: (usize, usize, usize)) -> bool {
        if world_map.len() != dims.0 {
            return false;
        }
        for i in 0..dims.0 {
            if world_map[i].len() != dims.1 {
                return false;
            }
            for j in 0..dims.1 {
                if world_map[i][j].len() != dims.2 {
                    return false;
                }
            }
        }
        true
    }

    fn _valid(&self, node: &(usize, usize, usize), altitude_gap: usize, min_altitude: usize) -> bool {
        let alt_idx = ((node.2 as f32) - (min_altitude as f32)) / (altitude_gap as f32);
        if alt_idx < 0.0 {
            return false;
        }
        node.0 < self.dims.0 && node.1 < self.dims.1 && (alt_idx as usize) < self.dims.2
    }

    fn _freespace(&self, node: &(usize, usize, usize), altitude_gap: usize, min_altitude: usize) -> bool {
        // True if freespace, False if obstacle
        if !self._valid(node, altitude_gap, min_altitude) {
            return false;
        }
        if self.world_map[node.0][node.1][(node.2 - min_altitude) / altitude_gap] == 1 {
            return false;
        }
        true
    }

    fn _bresenham_3d(&self, state_1: &(usize, usize, usize), state_2: &(usize, usize, usize)) -> Vec<(usize, usize, usize)> {
        let mut list_of_points = vec![state_1.clone()];
        let dx = (state_2.0 as isize - state_1.0 as isize).abs();
        let dy = (state_2.1 as isize - state_1.1 as isize).abs();
        let dz = (state_2.2 as isize - state_1.2 as isize).abs();
        let xs = if state_2.0 > state_1.0 { 1 } else { -1 };
        let ys = if state_2.1 > state_1.1 { 1 } else { -1 };
        let zs = if state_2.2 > state_1.2 { 1 } else { -1 };

        // Driving axis is X-axis
        if dx >= dy && dx >= dz {
            let mut p1 = 2 * dy - dx;
            let mut p2 = 2 * dz - dx;
            let mut x1 = state_1.0 as isize;
            let mut y1 = state_1.1 as isize;
            let mut z1 = state_1.2 as isize;
            while x1 != state_2.0 as isize {
                x1 += xs;
                if p1 >= 0 {
                    y1 += ys;
                    p1 -= 2 * dx;
                }
                if p2 >= 0 {
                    z1 += zs;
                    p2 -= 2 * dx;
                }
                p1 += 2 * dy;
                p2 += 2 * dz;
                list_of_points.push((x1 as usize, y1 as usize, z1 as usize));
            }
        }
        // Driving Axis is Y-axis
        else if dy >= dx && dy >= dz {
            let mut p1 = 2 * dx - dy;
            let mut p2 = 2 * dz - dy;
            let mut x1 = state_1.0 as isize;
            let mut y1 = state_1.1 as isize;
            let mut z1 = state_1.2 as isize;
            while y1 != state_2.1 as isize {
                y1 += ys;
                if p1 >= 0 {
                    x1 += xs;
                    p1 -= 2 * dy;
                }
                if p2 >= 0 {
                    z1 += zs;
                    p2 -= 2 * dy;
                }
                p1 += 2 * dx;
                p2 += 2 * dz;
                list_of_points.push((x1 as usize, y1 as usize, z1 as usize));
            }
        }
        // Driving axis is Z axis
        else{
            let mut p1 = 2 * dy - dz;
            let mut p2 = 2 * dx - dz;
            let mut x1 = state_1.0 as isize;
            let mut y1 = state_1.1 as isize;
            let mut z1 = state_1.2 as isize;
            while z1 != state_2.2 as isize {
                z1 += zs;
                if p1 >= 0 {
                    y1 += ys;
                    p1 -= 2 * dz;
                }
                if p2 >= 0 {
                    x1 += xs;
                    p2 -= 2 * dz;
                }
                p1 += 2 * dy;
                p2 += 2 * dx;
                list_of_points.push((x1 as usize, y1 as usize, z1 as usize));
            }
        }
        list_of_points
    }

    fn _get_neighbors_freespace(&self, state: &(usize, usize, usize), altitude_gap: usize, min_altitude: usize) -> Vec<(usize, usize, usize)> {
        let mut list_of_neighbors = vec![];
        let deltas_2d = [-1, 0, 1];

        // Generate all 2D deltas and append (x, y, 0)
        let mut deltas_3d: Vec<(i32, i32, i32)> = deltas_2d
            .iter()
            .flat_map(|&x| deltas_2d.iter().map(move |&y| (x, y, 0)))
            .collect();

        // Add the additional deltas (0, 0, altitude_gap) and (0, 0, -altitude_gap)
        deltas_3d.push((0, 0, altitude_gap as i32));
        deltas_3d.push((0, 0, -1 * altitude_gap as i32));

        // Generate all neighbors
        for delta in deltas_3d {
            if delta != (0, 0, 0) {
                let neighbor = (
                    (state.0 as i32 + delta.0),
                    (state.1 as i32 + delta.1),
                    (state.2 as i32 + delta.2),
                );
                if neighbor.0 >= 0 && neighbor.1 >= 0 && neighbor.2 >= 0 {
                    let neighbor_u = (neighbor.0 as usize, neighbor.1 as usize, neighbor.2 as usize);
                    if self._freespace(&neighbor_u, altitude_gap, min_altitude) {
                        list_of_neighbors.push(neighbor_u);
                    }
                }
            }
        }
        list_of_neighbors
    }
    
    fn _get_neighbors_valid(&self, state: &(usize, usize, usize), altitude_gap: usize, min_altitude: usize) -> Vec<(usize, usize, usize)> {
        let mut list_of_neighbors = vec![];
        let deltas_2d = [-1, 0, 1];

        // Generate all 2D deltas and append (x, y, 0)
        let mut deltas_3d: Vec<(i32, i32, i32)> = deltas_2d
            .iter()
            .flat_map(|&x| deltas_2d.iter().map(move |&y| (x, y, 0)))
            .collect();

        // Add the additional deltas (0, 0, altitude_gap) and (0, 0, -altitude_gap)
        deltas_3d.push((0, 0, altitude_gap as i32));
        deltas_3d.push((0, 0, -1 * altitude_gap as i32));

        // Generate all neighbors
        for delta in deltas_3d {
            if delta != (0, 0, 0) {
                let neighbor = (
                    (state.0 as i32 + delta.0),
                    (state.1 as i32 + delta.1),
                    (state.2 as i32 + delta.2),
                );
                if neighbor.0 >= 0 && neighbor.1 >= 0 && neighbor.2 >= 0 {
                    let neighbor_u = (neighbor.0 as usize, neighbor.1 as usize, neighbor.2 as usize);
                    if self._valid(&neighbor_u, altitude_gap, min_altitude) {
                        list_of_neighbors.push(neighbor_u);
                    }
                }
            }
        }
        list_of_neighbors
    } 
    
    fn _line_of_sight(&self, state_1: &(usize, usize, usize), state_2: &(usize, usize, usize), altitude_gap: usize, min_altitude: usize) -> bool {
        // 2.5D Constraint: The line of sight must be in the same z plane or vertically above or below
        if state_1.2 != state_2.2 && (state_1.0 != state_2.0 || state_1.1 != state_2.1) {
            return false;
        }

        let path = self._bresenham_3d(state_1, state_2);
        for node in path {
            if !self._freespace(&node, altitude_gap, min_altitude) {
                return false;
            }
        }
        true
    }

    fn _penalized_l2_norm(&self, point1: &(usize, usize, usize), point2: &(usize, usize, usize), z_penalty: f64) -> f64 {
        let dx = point2.0 as f64 - point1.0 as f64;
        let dy = point2.1 as f64 - point1.1 as f64;
        let dz = (point2.2 as f64 - point1.2 as f64) * z_penalty; // Apply the penalty to the z component
    
        // Calculate the penalized norm
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn _heuristic(&self, state: &(usize, usize, usize), goal: &(usize, usize, usize), z_penalty: f64) -> f64 {
        self._penalized_l2_norm(state, goal, z_penalty)
    }
    
    fn _is_middle_point_on_line_xy(&self, point1: (usize, usize, usize), point2: (usize, usize, usize), middle_point: (usize, usize, usize)) -> bool {
        // Calculate the differences to avoid division by zero
        let delta_x1 = point2.0 as isize - point1.0 as isize;
        let delta_y1 = point2.1 as isize - point1.1 as isize;
        let delta_x2 = middle_point.0 as isize - point1.0 as isize;
        let delta_y2 = middle_point.1 as isize - point1.1 as isize;

        // Check if the slopes are equal using cross multiplication to avoid division
        delta_y1 * delta_x2 == delta_y2 * delta_x1
    }

    fn _remove_collinear_points(&self, points: &Vec<(usize, usize, usize)>) -> Vec<(usize, usize, usize)> {
        if points.len() < 2 {
            return points.clone(); // Return early if not enough points
        }

        // Initialize the list of non-collinear points with the first point
        let mut non_collinear_points = vec![points[0]];

        // Iterate through the points and check for collinearity
        // Only remove points that are collinear in the same z plane
        for i in 1..points.len() - 1 {
            // Points share same z plane and are collinear
            let guard_1 = points[i].2 == points[i-1].2 && points[i].2 == points[i+1].2;
            let guard_2 = self._is_middle_point_on_line_xy(points[i-1], points[i+1], points[i]);
            
            // Points share same x and y coordinates and are in different z planes
            let guard_3 = points[i].0 == points[i-1].0 && points[i].0 == points[i+1].0 && points[i].1 == points[i-1].1 && points[i].1 == points[i+1].1;
            let guard_4 = points[i].2 != points[i-1].2 && points[i].2 != points[i+1].2;
            if !((guard_1 && guard_2) || (guard_3 && guard_4)) {
                non_collinear_points.push(points[i]);
            }
        }

        // Add the last point
        non_collinear_points.push(points[points.len() - 1]);

        non_collinear_points
    }
    
    fn _reconstruct(&self,
        node: &(usize, usize, usize),
        pathmap: &HashMap<(usize, usize, usize), (usize, usize, usize)>,
        backward_pathmap: &HashMap<(usize, usize, usize), (usize, usize, usize)>,
    ) -> Vec<(usize, usize, usize)>{
        let mut path = vec![];
        let mut current_node = node;
        path.push(*current_node);
        while pathmap.contains_key(current_node) {
            current_node = pathmap.get(current_node).unwrap();
            path.push(*current_node);
        }
        path.reverse();
        current_node = node;
        while backward_pathmap.contains_key(current_node) {
            current_node = backward_pathmap.get(current_node).unwrap();
            path.push(*current_node);
        }
        path
    }

    fn _get_freespace(
        &self,
        state: (usize, usize, usize),
        max_search_dist: usize,
        is_goal: bool,
        is_min_support: bool,
        centering_offset: (f64, f64),
        context: &mut String,
        altitude_gap: usize,
        min_altitude: usize, 
    ) -> Vec<(usize, usize, usize)> { 
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        queue.push_back(state);
        while !queue.is_empty() {
            let current = queue.pop_front().unwrap();
            visited.insert(current);
            if self._freespace(&current, altitude_gap, min_altitude){
                return vec![current];
            } else if is_min_support && is_goal{
                context.push_str(&format!(
                    "IG: {} {} {}\n", 
                    current.0 as f64 + centering_offset.0, 
                    current.1 as f64 + centering_offset.1, 
                    current.2 as f64 * -1.0
                ));
            }

            // Calculate distance to break the loop if greater than max_search_dist
            let state_vec = Vector3::new(state.0 as isize, state.1 as isize, state.2 as isize);
            let current_vec = Vector3::new(current.0 as isize, current.1 as isize, current.2 as isize);
            let dist = (state_vec - current_vec).abs().sum();

            if dist > max_search_dist as isize {
                return Vec::new(); // Return empty if out of search distance
            }

            // Process valid neighbors
            for neighbor in self._get_neighbors_valid(&current, altitude_gap, min_altitude) { 
                let neighbor_vec = Vector3::new(neighbor.0 as isize, neighbor.1 as isize, neighbor.2 as isize);
                let dist_to_neighbor = (state_vec - neighbor_vec).abs().sum();

                if !visited.contains(&neighbor) && !queue.contains(&neighbor) && dist_to_neighbor <= max_search_dist as isize {
                    queue.push_back(neighbor);
                }
            }
        }
        Vec::new()
    }
    
    fn _search(&self, 
        start: (usize, usize, usize), 
        goal: (usize, usize, usize),
        altitude_gap: usize,
        min_altitude: usize,
        z_penalty: f64
    ) -> Vec<(usize, usize, usize)> {
        // Initialize data structures
        let mut forward_closed_set = HashSet::new(); // The set of nodes already evaluated
        let mut forward_open_pq = BinaryHeap::new(); // Candidates for evaluation
        let mut forward_open_set = HashSet::new();
        let mut forward_came_from = HashMap::new(); // Keys are nodes, values are the predecessor node
        let mut forward_g_score = HashMap::new(); // cost from start to node structure

        
        let mut backward_closed_set = HashSet::new(); // The set of nodes already evaluated
        let mut backward_open_pq = BinaryHeap::new(); // Candidates for evaluation
        let mut backward_open_set = HashSet::new();
        let mut backward_came_from = HashMap::new(); // Keys are nodes, values are the predecessor node
        let mut backward_g_score = HashMap::new(); // cost from start to node structure

        // Set up the forward problem by expanding the start node
        forward_g_score.insert(start, 0.0);
        let forward_initial_nodes = self._get_neighbors_freespace(&start, altitude_gap, min_altitude);
        for n in forward_initial_nodes {
            let g = self._penalized_l2_norm(&start, &n, z_penalty);
            let h = self._heuristic(&n, &goal, z_penalty);
            forward_g_score.insert(n, g);
            forward_came_from.insert(n, start);
            forward_open_set.insert(n);
            forward_open_pq.push(Node{state: n, g, h: h, f: g + h});
        }
        forward_closed_set.insert(start);

        // Set up the backward problem by expanding the goal node
        backward_g_score.insert(goal, 0.0);
        let backward_initial_nodes = self._get_neighbors_freespace(&goal, altitude_gap, min_altitude);
        for n in backward_initial_nodes {
            let g = self._penalized_l2_norm(&goal, &n, z_penalty);
            let h = self._heuristic(&n, &start, z_penalty);
            backward_g_score.insert(n, g);
            backward_came_from.insert(n, goal);
            backward_open_set.insert(n);
            backward_open_pq.push(Node{state: n, g, h: h, f: g + h});
        }
        backward_closed_set.insert(goal);

        while !forward_open_pq.is_empty() && !backward_open_pq.is_empty(){
            let forward_current = forward_open_pq.pop().unwrap();
            if forward_closed_set.contains(&forward_current.state) {
                /*  
                    Because we leave old copies instead of removing from openPQ
                    (instead of properly updating priorities)
                    we need to skip (continue) if we find current in closedSet
                 */
                continue;
            }

            // Lazy theta*, check line of sight when expanding and fix parent if problem arises
            if self.is_theta_star{
                if !self._line_of_sight(&forward_current.state, &forward_came_from.get(&forward_current.state).unwrap(), altitude_gap, min_altitude){
                    let my_neighbors: Vec<(usize, usize, usize)> = self._get_neighbors_freespace(&forward_current.state, altitude_gap, min_altitude).into_iter().filter(|item| forward_closed_set.contains(item)).collect();
                    let values: Vec<f64> = my_neighbors.iter().map(|item| forward_g_score.get(item).unwrap_or(&0.0) + self._penalized_l2_norm(item, &forward_current.state, z_penalty)).collect();
                    if let Some((min_index, _)) = values.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                        let min_neighbor = &my_neighbors[min_index];
                        forward_came_from.insert(forward_current.state.clone(), min_neighbor.clone());
                        forward_g_score.insert(forward_current.state.clone(), values[min_index]);
                    }
                }
            }

            forward_open_set.remove(&forward_current.state);
            forward_closed_set.insert(forward_current.state);

            if backward_closed_set.contains(&forward_current.state) {
                return self._reconstruct(&forward_current.state, &forward_came_from, &backward_came_from);
            }

            // Get the neighbors, and for each:
            let forward_neighbors = self._get_neighbors_freespace(&forward_current.state, altitude_gap, min_altitude);
            for neighbor in forward_neighbors{
                // If it isn't already in the closedSet (if it is, ignore it):
                if !forward_closed_set.contains(&neighbor) {
                    let g = forward_g_score.get(&forward_current.state).unwrap() + self._penalized_l2_norm(&forward_current.state, &neighbor, z_penalty);
                    let h = self._heuristic(&neighbor, &goal, z_penalty);
                    if !forward_open_set.contains(&neighbor) {
                        forward_g_score.insert(forward_current.state, g);
                        forward_open_pq.push(Node{state: neighbor, g, h, f: g + h});
                        forward_open_set.insert(neighbor);
                        forward_came_from.insert(neighbor, forward_current.state);
                    } else {
                        // neighbor is already in open set
                        if g < *forward_g_score.get(&neighbor).unwrap_or(&INFINITY) { // We're doing better with this path; update entry in openPQ
                            forward_g_score.insert(neighbor, g);
                            forward_came_from.insert(neighbor, forward_current.state);
                            // remove old copy won't be necessary because
                            // the lower priority copy will get taken off first and added to open set
                            forward_open_pq.push(Node{state: neighbor, g, h, f: g + h});
                        }
                    }

                    if self.is_theta_star{
                        if true{ // delay line of sight check lazy theta*
                            // If the gScore for neighbor thru parent of current is better than gscore of neighbor otherwise
                            let parent = forward_came_from.get(&forward_current.state).unwrap();
                            let g_parent = forward_g_score.get(parent).unwrap() + self._penalized_l2_norm(parent, &neighbor, z_penalty);
                            if g_parent < *forward_g_score.get(&neighbor).unwrap_or(&INFINITY) {
                                forward_came_from.insert(neighbor, parent.clone());
                                forward_g_score.insert(neighbor, g_parent);
                                forward_open_pq.push(Node{state: neighbor, g: g_parent, h: h, f: g_parent + h});
                            }
                        }
                    }
                    
                }
            }

            let backward_current = backward_open_pq.pop().unwrap();
            if backward_closed_set.contains(&backward_current.state) {
                continue;
            }

            // Lazy theta*, check line of sight when expanding and fix parent if problem arises
            if self.is_theta_star{
                if !self._line_of_sight(&backward_current.state, &backward_came_from.get(&backward_current.state).unwrap(), altitude_gap, min_altitude){
                    let my_neighbors: Vec<(usize, usize, usize)> = self._get_neighbors_freespace(&backward_current.state, altitude_gap, min_altitude).into_iter().filter(|item| backward_closed_set.contains(item)).collect();
                    let values: Vec<f64> = my_neighbors.iter().map(|item| backward_g_score.get(item).unwrap_or(&0.0) + self._penalized_l2_norm(item, &backward_current.state, z_penalty)).collect();
                    if let Some((min_index, _)) = values.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()) {
                        let min_neighbor = &my_neighbors[min_index];
                        backward_came_from.insert(backward_current.state.clone(), min_neighbor.clone());
                        backward_g_score.insert(backward_current.state.clone(), values[min_index]);
                    }
                }
            }

            backward_open_set.remove(&backward_current.state);
            backward_closed_set.insert(backward_current.state);

            if forward_closed_set.contains(&backward_current.state) {
                return self._reconstruct(&backward_current.state, &forward_came_from, &backward_came_from);
            }

            // Get the neighbors, and for each:
            let backward_neighbors = self._get_neighbors_freespace(&backward_current.state, altitude_gap, min_altitude);
            for neighbor in backward_neighbors{
                // If it isn't already in the closedSet (if it is, ignore it):
                if !backward_closed_set.contains(&neighbor) {
                    let g = backward_g_score.get(&backward_current.state).unwrap() + self._penalized_l2_norm(&backward_current.state, &neighbor, z_penalty);
                    let h = self._heuristic(&neighbor, &start, z_penalty);
                    if !backward_open_set.contains(&neighbor) {
                        backward_g_score.insert(backward_current.state, g);
                        backward_open_pq.push(Node{state: neighbor, g, h, f: g + h});
                        backward_open_set.insert(neighbor);
                        backward_came_from.insert(neighbor, backward_current.state);
                    } else {
                        // neighbor is already in open set
                        if g < *backward_g_score.get(&neighbor).unwrap_or(&INFINITY) { // We're doing better with this path; update entry in openPQ
                            backward_g_score.insert(neighbor, g);
                            backward_came_from.insert(neighbor, backward_current.state);
                            // remove old copy won't be necessary because
                            // the lower priority copy will get taken off first and added to open set
                            backward_open_pq.push(Node{state: neighbor, g, h, f: g + h});
                        }
                    }

                    if self.is_theta_star{
                        if true{ // delay line of sight check lazy theta*
                            // If the gScore for neighbor thru parent of current is better than gscore of neighbor otherwise
                            let parent = backward_came_from.get(&backward_current.state).unwrap();
                            let g_parent = backward_g_score.get(parent).unwrap() + self._penalized_l2_norm(parent, &neighbor, z_penalty);
                            if g_parent < *backward_g_score.get(&neighbor).unwrap_or(&INFINITY) {
                                backward_came_from.insert(neighbor, parent.clone());
                                backward_g_score.insert(neighbor, g_parent);
                                backward_open_pq.push(Node{state: neighbor, g: g_parent, h: h, f: g_parent + h});
                            }
                        }
                    }
                    
                }
            }

        }
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_binary_heap(){
        let mut heap = BinaryHeap::new();
        heap.push(Node{state: (1, 1, 1), g: 1.0, h: 1.0, f: 2.0});
        heap.push(Node{state: (2, 2, 2), g: 2.0, h: 2.0, f: 4.0});
        heap.push(Node{state: (3, 3, 3), g: 3.0, h: 3.0, f: 6.0});
        let node = heap.pop().unwrap();
        assert_eq!(node.state, (1, 1, 1));
    }

    #[test]
    fn test_validate_world_map(){
        let world_map = vec![vec![vec![0; 3]; 3]; 3];
        let dims = (3, 3, 3);
        assert!(ThetaStarBi::_validate_world_map(world_map, dims));
    }

    #[test]
    fn test_set_world_map(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let world_map = vec![vec![vec![0; 3]; 3]; 3];
        theta_star_bi.set_world_map(world_map);
    }

    #[test]
    fn test_valid(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let node = (1, 1, 1);
        let altitude_gap = 5;
        let min_altitude = 2;
        assert!(!theta_star_bi._valid(&node, altitude_gap, min_altitude));

        let invalid_node = (3, 3, 3);
        assert!(!theta_star_bi._valid(&invalid_node, altitude_gap, min_altitude));
    }

    #[test]
    fn test_freespace(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let node = (1, 1, 1);
        let altitude_gap = 1;
        let min_altitude = 0;
        assert!(theta_star_bi._freespace(&node, altitude_gap, min_altitude));

        theta_star_bi.world_map[1][1][1] = 1;
        let obstacle_node = (1, 1, 1);
        assert!(!theta_star_bi._freespace(&obstacle_node, altitude_gap, min_altitude));
    }

    #[test]
    fn test_bresenham_3d(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let state_1 = (0, 0, 0);
        let state_2 = (2, 2, 2);
        let path = theta_star_bi._bresenham_3d(&state_1, &state_2);
        assert_eq!(path, vec![(0, 0, 0), (1, 1, 1), (2, 2, 2)]);

        let state_1 = (0, 2, 2);
        let state_2 = (6, 4, 0);
        let path = theta_star_bi._bresenham_3d(&state_1, &state_2);
        assert_eq!(path, vec![(0, 2, 2), (1, 2, 2), (2, 3, 1), (3, 3, 1), (4, 3, 1), (5, 4, 0), (6, 4, 0)]);
    }

    #[test]
    fn test_get_neighbors_valid(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let state = (1, 1, 1);
        let altitude_gap = 1;
        let min_altitude = 0;
        let neighbors = theta_star_bi._get_neighbors_valid(&state, altitude_gap, min_altitude);
        assert_eq!(neighbors, vec![(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (1, 2, 1), (2, 0, 1), (2, 1, 1), (2, 2, 1), (1, 1, 2), (1, 1, 0)]);

        let altitude_gap = 5;
        let min_altitude = 2;
        let neighbors = theta_star_bi._get_neighbors_valid(&state, altitude_gap, min_altitude);
        assert_eq!(neighbors, vec![(1, 1, 6)]);
    }

    #[test]
    fn test_get_neighbors_freespace(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let state = (1, 1, 1);
        let altitude_gap = 1;
        let min_altitude = 0;
        theta_star_bi.world_map[0][0][1] = 1;
        theta_star_bi.world_map[0][1][1] = 1;
        let neighbors = theta_star_bi._get_neighbors_freespace(&state, altitude_gap, min_altitude);
        assert_eq!(neighbors, vec![(0, 2, 1), (1, 0, 1), (1, 2, 1), (2, 0, 1), (2, 1, 1), (2, 2, 1), (1, 1, 2), (1, 1, 0)]);
    }
    
    #[test]
    fn test_line_of_sight(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let state_1 = (0, 0, 0);
        let state_2 = (2, 2, 0);
        let altitude_gap = 1;
        let min_altitude = 0;
        assert!(theta_star_bi._line_of_sight(&state_1, &state_2, altitude_gap, min_altitude));

        theta_star_bi.world_map[1][1][0] = 1;
        let state_1 = (0, 0, 0);
        let state_2 = (2, 2, 0);
        let altitude_gap = 1;
        let min_altitude = 0;
        assert!(!theta_star_bi._line_of_sight(&state_1, &state_2, altitude_gap, min_altitude));
    }

    #[test]
    fn test_penalized_l2_norm(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let point1 = (0, 0, 0);
        let point2 = (2, 2, 2);
        let z_penalty = 1.0;
        let penalized_norm = theta_star_bi._penalized_l2_norm(&point1, &point2, z_penalty);
        assert_eq!(penalized_norm, 3.4641016151377544);

        let point1 = (0, 0, 0);
        let point2 = (2, 2, 2);
        let z_penalty = 2.0;
        let penalized_norm = theta_star_bi._penalized_l2_norm(&point1, &point2, z_penalty);
        assert_eq!(penalized_norm, 4.898979485566356);
    }
 
    #[test]
    fn test_is_middle_point_on_line_xy(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let point1 = (0, 0, 1);
        let point2 = (2, 2, 1);
        let middle_point = (1, 1, 1);
        assert!(theta_star_bi._is_middle_point_on_line_xy(point1, point2, middle_point));

        let point1 = (0, 0, 1);
        let point2 = (2, 2, 1);
        let middle_point = (1, 0, 1);
        assert!(!theta_star_bi._is_middle_point_on_line_xy(point1, point2, middle_point));
    }

    #[test]
    fn test_remove_collinear_points(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        // Only remove collinear points on same z-plane
        let points = vec![(0, 0, 0), (1, 1, 1), (2, 2, 2)];
        let non_collinear_points = theta_star_bi._remove_collinear_points(&points);
        assert_eq!(non_collinear_points, vec![(0, 0, 0), (1, 1, 1), (2, 2, 2)]);

        let points = vec![(0, 0, 0), (0, 0, 1), (2, 2, 1), (3, 3, 1)];
        let non_collinear_points = theta_star_bi._remove_collinear_points(&points);
        assert_eq!(non_collinear_points, vec![(0, 0, 0), (0, 0, 1), (3, 3, 1)]);

        let points = vec![(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1)];
        let non_collinear_points = theta_star_bi._remove_collinear_points(&points);
        assert_eq!(non_collinear_points, vec![(0, 0, 1), (4, 4, 1)]);

        let points = vec![(0, 0, 0), (0, 0, 1), (0, 0, 2)];
        let non_collinear_points = theta_star_bi._remove_collinear_points(&points);
        assert_eq!(non_collinear_points, vec![(0, 0, 0), (0, 0, 2)]);
    }
    
    #[test]
    fn test_reconstruct(){
        let theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let node = (2, 2, 2);
        let mut pathmap = HashMap::new();
        pathmap.insert((2, 2, 2), (1, 1, 1));
        pathmap.insert((1, 1, 1), (0, 0, 0));
        let mut backward_pathmap = HashMap::new();
        backward_pathmap.insert((2, 2, 2), (2, 1, 1));
        backward_pathmap.insert((2, 1, 1), (2, 0, 0));
        let path = theta_star_bi._reconstruct(&node, &pathmap, &backward_pathmap);
        assert_eq!(path, vec![(0, 0, 0), (1, 1, 1), (2, 2, 2), (2, 1, 1), (2, 0, 0)]);
    }

    #[test]
    fn test_get_freespace(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let state = (1, 1, 1);
        let max_search_dist = 5;
        let is_goal = true;
        let is_min_support = true;
        let centering_offset = (1.5, 1.5);
        let mut context = String::new();
        let altitude_gap = 1;
        let min_altitude = 0;
        let freespace = theta_star_bi._get_freespace(state, max_search_dist, is_goal, is_min_support, centering_offset, &mut context, altitude_gap, min_altitude);
        assert_eq!(freespace, vec![(1, 1, 1)]);
        assert!(context.is_empty());

        theta_star_bi.world_map[1][1][1] = 1;
        context.clear();
        let freespace = theta_star_bi._get_freespace(state, max_search_dist, is_goal, is_min_support, centering_offset, &mut context, altitude_gap, min_altitude);
        assert_eq!(freespace, vec![(0, 0, 1)]);
        assert_eq!(context, "IG: 2.5 2.5 -1\n");

        theta_star_bi.world_map = vec![vec![vec![1; 3]; 3]; 3];
        context.clear();
        let freespace = theta_star_bi._get_freespace(state, max_search_dist, is_goal, is_min_support, centering_offset, &mut context, altitude_gap, min_altitude);
        assert!(freespace.is_empty());
        assert_eq!(context.len(), 405);
    }

    #[test]
    fn test_internal_search(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let start = (0, 0, 0);
        let goal = (2, 2, 2);
        let altitude_gap = 1;
        let min_altitude = 0;
        let z_penalty = 1.0;
        let path = theta_star_bi._search(start, goal, altitude_gap, min_altitude, z_penalty);
        assert_eq!(path, vec![(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 2, 2)]);

        theta_star_bi.world_map = vec![vec![vec![1; 3]; 3]; 3];
        let path = theta_star_bi._search(start, goal, altitude_gap, min_altitude, z_penalty);
        assert!(path.is_empty());
    }

    #[test]
    fn test_external_search(){
        let mut theta_star_bi = ThetaStarBi::new(vec![vec![vec![0; 3]; 3]; 3], (3, 3, 3), true);
        let start = (0., 0., 0.);
        let goal = (2., 2., -2.);
        let altitude_gap = 1;
        let min_altitude = 0;
        let z_penalty = 1.0;
        let is_min_support = true;
        let centering_offset = (0., 0.);
        let freespace_search_max_dist = 5;
        let (path, context) = theta_star_bi.search(start, goal, altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, freespace_search_max_dist);
        let expected_path = vec![(0., 0., 0.), (1., 1., 0.), (1., 1., -2.), (2., 2., -2.)];
        let expected_context = "GF: 2 2 -2\n".to_string();
        assert_eq!(path, expected_path);
        assert_eq!(context, expected_context);

        theta_star_bi.world_map[0][0][0] = 1;
        theta_star_bi.world_map[2][2][2] = 1;
        let (path, context) = theta_star_bi.search(start, goal, altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, freespace_search_max_dist);
        assert_eq!(path, vec![(0.0, 1.0, -0.0), (0.0, 1.0, -0.0), (0.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, 1.0, -2.0)]);
        assert_eq!(context, "IG: 2 2 -2\nGF: 1 1 -2\n");

        theta_star_bi.world_map = vec![vec![vec![1; 3]; 3]; 3];
        let (path, context) = theta_star_bi.search(start, goal, altitude_gap, min_altitude, z_penalty, is_min_support, centering_offset, freespace_search_max_dist);
        assert!(path.is_empty());
        assert!(context.is_empty());
    }
}