import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import random
from collections import defaultdict
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import differential_evolution, dual_annealing
from functools import lru_cache
import heapq
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# RL imports (with error handling)
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Create dummy classes for when RL is not available
    class DummyEnv:
        pass
    gym = type('gym', (), {'Env': DummyEnv})()
    spaces = type('spaces', (), {'Discrete': lambda x: None, 'Box': lambda **kwargs: None})()
    # print("Info: Stable-Baselines3 not available. RL optimizer will use fallback SA.")

class SegmentType(Enum):
    """Types of path segments in grid navigation"""
    ALONG_ROW = "row"
    ALONG_HEADLAND = "headland"
    SPUR = "spur"  # Combined spur entry and exit
    TURN = "turn"

# Global color mapping for segment types (avoid duplication)
SEGMENT_COLORS = {
    SegmentType.ALONG_ROW: 'blue',
    SegmentType.ALONG_HEADLAND: 'purple',
    SegmentType.SPUR: 'orange'
}

# Helper functions for common operations
def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def interpolate_segment(seg: 'GridSegment', num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate points along a segment for smooth visualization.

    Returns:
        x_path, y_path, t_values (interpolation parameters)
    """
    if num_points is None:
        num_points = max(10, int(calculate_distance(seg.start, seg.end)))

    t = np.linspace(0, 1, num_points)
    x_path = seg.start[0] + t * (seg.end[0] - seg.start[0])
    y_path = seg.start[1] + t * (seg.end[1] - seg.start[1])

    return x_path, y_path, t

class ElevationModel:
    """Models vineyard terrain elevation"""

    def __init__(self, grid_bounds: Tuple[float, float, float, float],
                 resolution: float = 1.0,
                 terrain_type: str = 'mosel'):
        """
        Initialize elevation model.

        Parameters:
        -----------
        grid_bounds: (x_min, x_max, y_min, y_max) in meters
        resolution: Grid resolution for elevation sampling (meters)
        terrain_type: 'mosel' for Mosel Valley, 'gentle' for gentle slopes
        """
        self.bounds = grid_bounds
        self.resolution = resolution
        self.terrain_type = terrain_type

        # Generate elevation grid
        x = np.arange(grid_bounds[0], grid_bounds[1], resolution)
        y = np.arange(grid_bounds[2], grid_bounds[3], resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Generate realistic vineyard terrain
        self.elevation = self._generate_terrain()

        # Create interpolation function
        self.elevation_func = RectBivariateSpline(y, x, self.elevation)

    def _generate_terrain(self) -> np.ndarray:
        """
        Generate realistic vineyard terrain based on terrain type.

        Mosel Valley: Steep terraced hillside vineyards (30-60% grade)
        Gentle: Gentle rolling slopes (5-10% grade)
        """
        if self.terrain_type == 'mosel':
            return self._generate_mosel_terrain()
        else:
            return self._generate_gentle_terrain()

  

    def _generate_mosel_terrain(self) -> np.ndarray:
        """
        Generate Mosel Valley vineyard terrain with realistic hill profile.

        Characteristics:
        - Base elevation: 150m (river level, from mosel_occupancy_grid.yaml)
        - Hill-shaped profile with steeper slopes at base and summit
        - Random variations for natural appearance
        - Elevation range: 150m (river) to ~280m (hilltop)
        - Elevation changes along row direction (X axis)
        """
        # Base parameters from Mosel Valley
        base_elevation = 150.0  # River level (from YAML origin)
        max_elevation_gain = 30.0  # Typical hilltop elevation gain

        # Normalize coordinates to [0, 1] range
        x_norm = (self.X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        y_norm = (self.Y - self.bounds[2]) / (self.bounds[3] - self.bounds[2])

        # Create realistic hill profile using sigmoid-like curve
        # This creates steeper slopes at the base and top, gentler in middle
        # Using a combination of polynomial and exponential for natural hill shape

        # S-curve (sigmoid-like) for gradual start and end
        hill_profile = 1 / (1 + np.exp(-10 * (x_norm - 0.5)))  # Sigmoid centered at 0.5

        # Combine sigmoid and power function for realistic hill
        # Higher at start (steeper base), gentle middle, steeper top
        hill_shape = 0.3 * hill_profile + 0.7 * (x_norm ** 1.5)

        # Apply to elevation gain
        hillside_slope = base_elevation + max_elevation_gain * hill_shape

        # Add realistic random variations (roughness)
        # Use multiple frequency components for natural terrain
        np.random.seed(42)  # For reproducibility

        # Large-scale variations (terrain features like gullies, ridges)
        large_scale = 2.0 * np.sin(0.4 * x_norm * 2 * np.pi + np.random.rand()) * \
                      np.cos(0.3 * y_norm * 2 * np.pi + np.random.rand())

        # Medium-scale variations (small hills and valleys)
        medium_scale = 1.2 * np.sin(1.2 * x_norm * 2 * np.pi + np.random.rand()) * \
                       np.cos(0.8 * y_norm * 2 * np.pi + np.random.rand())

        # Small-scale variations (surface roughness)
        small_scale = 0.6 * np.sin(3.0 * x_norm * 2 * np.pi + np.random.rand()) * \
                      np.cos(2.5 * y_norm * 2 * np.pi + np.random.rand())

        # Add cross-row slope variation (drainage channels between rows)
        # Slight 3-5% cross-slope for water drainage
        cross_slope = 0.04 * max_elevation_gain * np.sin(0.3 * y_norm * np.pi)

        # Valley shape - slight curvature (river valley bottom)
        valley_curvature = 1.5 * ((y_norm - 0.5) ** 2)

        # Combine all components
        elevation = (hillside_slope +
                    large_scale +
                    medium_scale +
                    small_scale +
                    cross_slope +
                    valley_curvature)

        return elevation

    def get_elevation(self, x: float, y: float) -> float:
        """Get elevation at specific point"""
        x_clipped = np.clip(x, self.bounds[0], self.bounds[1])
        y_clipped = np.clip(y, self.bounds[2], self.bounds[3])
        return float(self.elevation_func(y_clipped, x_clipped)[0, 0])

    def get_elevations_vectorized(self, x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Get elevations for arrays of points (optimized for bulk queries)"""
        x_clipped = np.clip(x_array, self.bounds[0], self.bounds[1])
        y_clipped = np.clip(y_array, self.bounds[2], self.bounds[3])
        return self.elevation_func(y_clipped, x_clipped, grid=False)

@dataclass
class Waypoint:
    """Represents an inspection waypoint in the vineyard"""
    x: float  # Position along row
    y: float  # Position between rows (in the alley)
    row_id: int  # Which row pair the waypoint is between (0 means between row 0 and 1)
    offset: float = 0.0  # Offset from center of alley
    z: float = 0.0  # Elevation (optional)

@dataclass
class GridSegment:
    """Represents a segment of the path"""
    type: SegmentType
    start: Tuple[float, float]
    end: Tuple[float, float]
    row_id: Optional[int] = None
    elevation_start: Optional[float] = None
    elevation_end: Optional[float] = None

    def distance(self) -> float:
        """
        Calculate segment distance in 2D (horizontal) using interpolated points
        for more accurate path length calculation
        """
        # Get interpolated points along the segment
        x_path, y_path, _ = self.interpolate()

        # Calculate cumulative distance along interpolated path
        total_dist = 0.0
        for i in range(1, len(x_path)):
            dx = x_path[i] - x_path[i-1]
            dy = y_path[i] - y_path[i-1]
            total_dist += np.sqrt(dx**2 + dy**2)

        # For SPUR segments, double the distance (round trip)
        if self.type == SegmentType.SPUR:
            total_dist *= 2

        return total_dist

    def distance_3d(self, elevation_model: Optional[ElevationModel] = None) -> float:
        """
        Calculate 3D segment distance considering elevation using interpolated points
        for realistic path length on hilly terrain.

        If elevation_model is provided, queries actual elevation at each point.
        Otherwise uses linear interpolation between start and end elevations.
        """
        # Get interpolated points along the segment
        x_path, y_path, t = self.interpolate()

        # Calculate elevation at each interpolated point
        if elevation_model is not None:
            # Query actual elevation from model at each interpolated point
            z_path = elevation_model.get_elevations_vectorized(x_path, y_path)
        elif self.elevation_start is not None and self.elevation_end is not None:
            # Linearly interpolate elevation
            z_path = self.elevation_start + t * (self.elevation_end - self.elevation_start)
        else:
            # No elevation data, fall back to 2D
            return self.distance()

        # Calculate cumulative 3D distance along interpolated path
        total_dist_3d = 0.0
        for i in range(1, len(x_path)):
            dx = x_path[i] - x_path[i-1]
            dy = y_path[i] - y_path[i-1]
            dz = z_path[i] - z_path[i-1]
            total_dist_3d += np.sqrt(dx**2 + dy**2 + dz**2)

        # For SPUR segments, double the distance (round trip)
        if self.type == SegmentType.SPUR:
            total_dist_3d *= 2

        return total_dist_3d

    def interpolate(self, num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate points along this segment"""
        return interpolate_segment(self, num_points)
    
class VineyardGrid:
    """Represents the vineyard grid structure"""

    def __init__(self, num_rows=10, row_spacing=3.0, row_length=100.0, tree_spacing=2.0):
        self.num_rows = num_rows
        self.row_spacing = row_spacing
        self.row_length = row_length
        self.tree_spacing = tree_spacing  # Distance between vine trees along a row
        self.left_headland_x = 0
        self.right_headland_x = row_length

        # Generate tree positions along each row
        self.tree_positions = self._generate_tree_positions()

    def _generate_tree_positions(self) -> dict:
        """Generate positions of all vine trees in the vineyard"""
        trees = {}
        for row_id in range(self.num_rows):
            row_y = self.get_row_y(row_id)
            # Trees positioned every tree_spacing meters along the row
            tree_x_positions = np.arange(0, self.row_length + self.tree_spacing, self.tree_spacing)
            trees[row_id] = [(x, row_y) for x in tree_x_positions if x <= self.row_length]
        return trees

    def get_nearest_tree(self, x: float, row_id: int) -> Tuple[float, float]:
        """Get the nearest tree position along a row"""
        if row_id not in self.tree_positions:
            return (x, self.get_row_y(row_id))

        trees_in_row = self.tree_positions[row_id]
        # Find nearest tree by x position
        nearest = min(trees_in_row, key=lambda t: abs(t[0] - x))
        return nearest
        
    def get_row_y(self, row_id: int) -> float:
        """Get y-coordinate of a row"""
        return row_id * self.row_spacing

    def get_alley_center_y(self, alley_id: int) -> float:
        """Get y-coordinate of the center of an alley (space between two rows)"""
        # Alley 0 is between row 0 and row 1
        row1_y = self.get_row_y(alley_id)
        row2_y = self.get_row_y(alley_id + 1)
        return (row1_y + row2_y) / 2.0

    def get_nearest_row(self, y: float) -> int:
        """Get the nearest row ID to a y-coordinate"""
        return int(round(y / self.row_spacing))

    def get_nearest_alley(self, y: float) -> int:
        """Get the nearest alley ID (space between rows) to a y-coordinate"""
        # Find which alley this y position is closest to
        alley_id = int(y / self.row_spacing)
        # Clamp to valid range (0 to num_rows-2)
        return max(0, min(self.num_rows - 2, alley_id))

    def is_valid_row(self, row_id: int) -> bool:
        """Check if row ID is valid"""
        return 0 <= row_id < self.num_rows

    def is_valid_alley(self, alley_id: int) -> bool:
        """Check if alley ID is valid"""
        return 0 <= alley_id < self.num_rows - 1

class EnergyAwareOptimizer:
    """Advanced optimization algorithms for energy-aware waypoint sequencing"""

    @staticmethod
    def cluster_by_elevation(waypoints: List[Waypoint],
                            num_clusters: int = 3,
                            elevation_model: Optional['ElevationModel'] = None) -> List[List[Waypoint]]:
        """
        Cluster waypoints by elevation zones to minimize climbing.

        Parameters:
        -----------
        waypoints: List of waypoints to cluster
        num_clusters: Number of elevation zones
        elevation_model: Optional elevation model

        Returns:
        --------
        List of waypoint clusters, sorted by elevation
        """
        if not waypoints:
            return []

        # Get elevations
        elevations = np.array([wp.z for wp in waypoints])

        # Simple k-means clustering by elevation
        min_elev = elevations.min()
        max_elev = elevations.max()

        # Create elevation bands
        band_size = (max_elev - min_elev) / num_clusters

        # Handle edge case: all waypoints at same elevation
        if band_size == 0:
            # All waypoints in single cluster
            return [waypoints]

        clusters = [[] for _ in range(num_clusters)]

        for wp in waypoints:
            # Assign to cluster based on elevation band
            cluster_idx = int((wp.z - min_elev) / band_size)
            cluster_idx = min(cluster_idx, num_clusters - 1)  # Handle edge case
            clusters[cluster_idx].append(wp)

        # Remove empty clusters and sort by mean elevation
        clusters = [c for c in clusters if c]
        clusters.sort(key=lambda c: np.mean([wp.z for wp in c]))

        return clusters

    @staticmethod
    def simulated_annealing_tsp(waypoints: List[Waypoint],
                                start_pos: Tuple[float, float],
                                planner: 'GridConstrainedPlanner',
                                objective: str = 'energy',
                                max_iterations: int = 1000,
                                initial_temp: float = 100.0,
                                cooling_rate: float = 0.995) -> List[Waypoint]:
        """
        Use Simulated Annealing to find near-optimal waypoint sequence.

        Parameters:
        -----------
        waypoints: List of waypoints to sequence
        start_pos: Starting position
        planner: Planner instance for cost calculation
        objective: 'energy', 'distance', or 'time'
        max_iterations: Number of iterations
        initial_temp: Starting temperature
        cooling_rate: Temperature decay rate

        Returns:
        --------
        Optimized waypoint sequence
        """
        if not waypoints:
            return []

        # Handle edge case: only 1 or 2 waypoints
        if len(waypoints) <= 2:
            return waypoints

        # Start with random sequence
        current_sequence = waypoints.copy()
        random.shuffle(current_sequence)

        # Calculate initial cost
        current_cost = EnergyAwareOptimizer._calculate_sequence_cost(
            current_sequence, start_pos, planner, objective
        )

        best_sequence = current_sequence.copy()
        best_cost = current_cost

        # Debug: Print what we're optimizing
        objective_unit = "J" if objective == "energy" else ("m" if objective == "distance" else "s")
        print(f"         Optimizing for: {objective} (initial: {current_cost:.2f} {objective_unit})")

        temperature = initial_temp

        for iteration in range(max_iterations):
            # Generate neighbor solution (2-opt swap)
            new_sequence = current_sequence.copy()

            # Random 2-opt swap (need at least 2 elements)
            if len(new_sequence) < 2:
                break

            i, j = sorted(random.sample(range(len(new_sequence)), 2))
            new_sequence[i:j+1] = reversed(new_sequence[i:j+1])

            # Calculate new cost
            new_cost = EnergyAwareOptimizer._calculate_sequence_cost(
                new_sequence, start_pos, planner, objective
            )

            # Check for invalid cost (inf means calculation failed)
            if new_cost == float('inf'):
                continue

            # Acceptance probability
            cost_diff = new_cost - current_cost

            if cost_diff < 0 or random.random() < np.exp(-cost_diff / temperature):
                current_sequence = new_sequence
                current_cost = new_cost

                # Update best if improved
                if current_cost < best_cost:
                    best_sequence = current_sequence.copy()
                    best_cost = current_cost

            # Cool down
            temperature *= cooling_rate

            # Periodic garbage collection to prevent memory buildup (more frequent)
            if iteration % 5 == 0:
                gc.collect()

        gc.collect()  # Final cleanup

        # Debug: Print final best cost
        print(f"         Final best {objective}: {best_cost:.2f} {objective_unit}")

        return best_sequence

    @staticmethod
    def _calculate_sequence_cost(sequence: List[Waypoint],
                                 start_pos: Tuple[float, float],
                                 planner: 'GridConstrainedPlanner',
                                 objective: str = 'energy') -> float:
        """Calculate cost (energy, distance, or time) for a waypoint sequence"""
        try:
            _, metrics, _ = planner.plan_complete_tour(
                sequence, start_pos, sequencing_mode='custom', custom_sequence=sequence
            )

            if objective == 'energy':
                return metrics['total_energy_j']
            elif objective == 'distance':
                return metrics['total_distance']
            elif objective == 'time':
                return metrics['total_time']
            else:
                return metrics['total_energy_j']
        except Exception as e:
            # If calculation fails, return a large penalty
            print(f"Warning: Cost calculation failed: {e}")
            return float('inf')

    @staticmethod
    def two_opt_improve(waypoints: List[Waypoint],
                       start_pos: Tuple[float, float],
                       planner: 'GridConstrainedPlanner',
                       objective: str = 'energy',
                       max_iterations: int = 100) -> List[Waypoint]:
        """
        Apply 2-opt local search to improve waypoint sequence.

        Returns:
        --------
        Improved waypoint sequence
        """
        if len(waypoints) < 3:
            return waypoints

        current_sequence = waypoints.copy()
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(current_sequence) - 1):
                for j in range(i + 2, len(current_sequence)):
                    # Create new sequence with reversed segment
                    new_sequence = current_sequence.copy()
                    new_sequence[i:j] = reversed(new_sequence[i:j])

                    # Evaluate both sequences
                    current_cost = EnergyAwareOptimizer._calculate_sequence_cost(
                        current_sequence, start_pos, planner, objective
                    )
                    new_cost = EnergyAwareOptimizer._calculate_sequence_cost(
                        new_sequence, start_pos, planner, objective
                    )

                    # Accept if improved
                    if new_cost < current_cost:
                        current_sequence = new_sequence
                        improved = True
                        break

                if improved:
                    break

        return current_sequence

    @staticmethod
    def multi_objective_optimize(waypoints: List[Waypoint],
                                start_pos: Tuple[float, float],
                                planner: 'GridConstrainedPlanner',
                                energy_weight: float = 0.7,
                                time_weight: float = 0.3,
                                max_iterations: int = 500) -> Tuple[List[Waypoint], dict]:
        """
        Multi-objective optimization balancing energy and time.
        FIXED: Baseline metrics are now calculated once to avoid segmentation fault.

        Parameters:
        -----------
        waypoints: List of waypoints
        start_pos: Starting position
        planner: Planner instance
        energy_weight: Weight for energy (0-1)
        time_weight: Weight for time (0-1)
        max_iterations: SA iterations

        Returns:
        --------
        Tuple of (optimized sequence, pareto_solutions)
        """
        # Normalize weights
        total_weight = energy_weight + time_weight
        energy_weight /= total_weight
        time_weight /= total_weight

        # FIX: Calculate baseline metrics ONCE before the loop (not 500 times!)
        baseline_seq = planner.sequence_waypoints(
            waypoints, start_pos, mode='nearest_neighbor'
        )
        _, baseline_metrics, _ = planner.plan_complete_tour(
            baseline_seq, start_pos, sequencing_mode='custom',
            custom_sequence=baseline_seq
        )
        baseline_energy = baseline_metrics['total_energy_j']
        baseline_time = baseline_metrics['total_time']

        # Find sequences optimized for each objective
        energy_optimal = EnergyAwareOptimizer.simulated_annealing_tsp(
            waypoints, start_pos, planner, objective='energy',
            max_iterations=max_iterations
        )

        time_optimal = EnergyAwareOptimizer.simulated_annealing_tsp(
            waypoints, start_pos, planner, objective='time',
            max_iterations=max_iterations
        )

        # Weighted objective function (now uses cached baseline)
        def weighted_cost(sequence):
            segments, metrics, _ = planner.plan_complete_tour(
                sequence, start_pos, sequencing_mode='custom',
                custom_sequence=sequence
            )

            # Use pre-calculated baseline metrics with safety checks
            norm_energy = metrics['total_energy_j'] / baseline_energy if baseline_energy > 0 else 1.0
            norm_time = metrics['total_time'] / baseline_time if baseline_time > 0 else 1.0

            return energy_weight * norm_energy + time_weight * norm_time

        # Simulated annealing with weighted objective
        current_sequence = waypoints.copy()
        random.shuffle(current_sequence)
        current_cost = weighted_cost(current_sequence)

        best_sequence = current_sequence.copy()
        best_cost = current_cost

        temperature = 100.0
        cooling_rate = 0.995

        for iteration in range(max_iterations):
            new_sequence = current_sequence.copy()
            i, j = sorted(random.sample(range(len(new_sequence)), 2))
            new_sequence[i:j+1] = reversed(new_sequence[i:j+1])

            new_cost = weighted_cost(new_sequence)
            cost_diff = new_cost - current_cost

            if cost_diff < 0 or random.random() < np.exp(-cost_diff / temperature):
                current_sequence = new_sequence
                current_cost = new_cost

                if current_cost < best_cost:
                    best_sequence = current_sequence.copy()
                    best_cost = current_cost

            temperature *= cooling_rate

        # Return best weighted solution and pareto solutions
        pareto_solutions = {
            'energy_optimal': energy_optimal,
            'time_optimal': time_optimal,
            'weighted_optimal': best_sequence
        }

        return best_sequence, pareto_solutions

    @staticmethod
    def scipy_optimize_tsp(waypoints: List[Waypoint],
                          start_pos: Tuple[float, float],
                          planner: 'GridConstrainedPlanner',
                          objective: str = 'energy',
                          method: str = 'dual_annealing') -> List[Waypoint]:
        """
        Use SciPy's global optimization algorithms for TSP.

        Parameters:
        -----------
        waypoints: List of waypoints to sequence
        start_pos: Starting position
        planner: Planner instance
        objective: 'energy', 'distance', or 'time'
        method: 'dual_annealing' or 'differential_evolution'

        Returns:
        --------
        Optimized waypoint sequence
        """
        if not waypoints:
            return []

        n = len(waypoints)

        # Objective function that takes a permutation vector
        def cost_function(x):
            # Convert continuous variables to permutation
            indices = np.argsort(x)
            sequence = [waypoints[i] for i in indices]

            # Calculate cost
            segments, metrics, _ = planner.plan_complete_tour(
                sequence, start_pos, sequencing_mode='custom',
                custom_sequence=sequence
            )

            if objective == 'energy':
                return metrics['total_energy_j']
            elif objective == 'distance':
                return metrics['total_distance']
            elif objective == 'time':
                return metrics['total_time']
            else:
                return metrics['total_energy_j']

        # Bounds for each variable (used to generate permutations)
        bounds = [(0, n) for _ in range(n)]

        if method == 'dual_annealing':
            result = dual_annealing(cost_function, bounds, maxiter=100, seed=42)
        else:  # differential_evolution
            result = differential_evolution(cost_function, bounds, maxiter=50, seed=42, workers=1)

        # Convert result back to sequence
        indices = np.argsort(result.x)
        optimized_sequence = [waypoints[i] for i in indices]

        return optimized_sequence


class WaypointSequencingEnv(gym.Env):
    """
    Gymnasium environment for waypoint sequencing optimization using RL.
    Compatible with Stable-Baselines3.
    """

    def __init__(self, waypoints: List[Waypoint], start_pos: Tuple[float, float],
                 planner: 'GridConstrainedPlanner', objective: str = 'energy'):
        super().__init__()

        self.waypoints = waypoints
        self.start_pos = start_pos
        self.planner = planner
        self.objective = objective
        self.n_waypoints = len(waypoints)

        # Action space: select next waypoint (discrete)
        self.action_space = spaces.Discrete(self.n_waypoints)

        # Observation space: current position + unvisited waypoints mask + elevation info
        # [current_x, current_y, current_z] + [mask for each waypoint] + [waypoint features]
        obs_dim = 3 + self.n_waypoints + (self.n_waypoints * 3)  # position + mask + waypoint coords
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.current_pos = None
        self.visited = None
        self.sequence = None
        self.total_cost = 0.0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_pos = self.start_pos
        self.visited = np.zeros(self.n_waypoints, dtype=bool)
        self.sequence = []
        self.total_cost = 0.0
        self.steps = 0

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Build observation vector"""
        # Current position
        current_z = self.planner.elevation.get_elevation(self.current_pos[0], self.current_pos[1])
        pos_features = [self.current_pos[0], self.current_pos[1], current_z]

        # Visited mask
        mask = (~self.visited).astype(np.float32)

        # Waypoint features (x, y, z for each waypoint)
        wp_features = []
        for wp in self.waypoints:
            wp_features.extend([wp.x, wp.y, wp.z])

        obs = np.concatenate([pos_features, mask, wp_features], dtype=np.float32)
        return obs

    def step(self, action):
        """Execute action (select next waypoint)"""
        # Check if action is valid (not already visited)
        if self.visited[action]:
            # Invalid action - penalize heavily
            reward = -1000.0
            terminated = True
            obs = self._get_observation()
            return obs, reward, terminated, False, {}

        # Mark as visited
        self.visited[action] = True
        selected_wp = self.waypoints[action]
        self.sequence.append(selected_wp)

        # Calculate cost of reaching this waypoint
        segments = self.planner.plan_grid_route(self.current_pos, (selected_wp.x, selected_wp.y))
        metrics = self.planner.calculate_metrics(segments)

        if self.objective == 'energy':
            step_cost = metrics['total_energy_j']
        elif self.objective == 'distance':
            step_cost = metrics['total_distance']
        else:  # time
            step_cost = metrics['total_time']

        self.total_cost += step_cost
        self.current_pos = (selected_wp.x, selected_wp.y)
        self.steps += 1

        # Reward is negative cost (we want to minimize)
        # Scale reward to reasonable range
        reward = -step_cost / 10000.0  # Normalize

        # Check if done
        terminated = np.all(self.visited)

        # Reward structure to ensure all waypoints are visited
        if terminated:
            # Large bonus ONLY if all waypoints visited
            completion_bonus = 100.0  # Large positive reward for completing the tour
            # Penalty for total cost
            cost_penalty = self.total_cost / 10000.0
            reward = completion_bonus - cost_penalty
        else:
            # Small step reward to encourage progress
            if self.visited is not None:
                remaining_waypoints = int(np.sum(~self.visited))
                progress_reward = 1.0 / (remaining_waypoints + 1)
                reward += progress_reward

        obs = self._get_observation()
        return obs, reward, terminated, False, {}


class RLOptimizer:
    """
    Reinforcement Learning-based optimizer using Stable-Baselines3.
    Provides state-of-the-art RL algorithms for waypoint sequencing.
    """

    @staticmethod
    def train_rl_agent(waypoints: List[Waypoint],
                      start_pos: Tuple[float, float],
                      planner: 'GridConstrainedPlanner',
                      objective: str = 'energy',
                      algorithm: str = 'PPO',
                      total_timesteps: int = 1000) -> List[Waypoint]:
        """
        Train an RL agent to optimize waypoint sequencing.

        Parameters:
        -----------
        waypoints: List of waypoints
        start_pos: Starting position
        planner: Planner instance
        objective: 'energy', 'distance', or 'time'
        algorithm: 'PPO', 'A2C', or 'SAC'
        total_timesteps: Training timesteps

        Returns:
        --------
        Optimized waypoint sequence
        """
        if not SB3_AVAILABLE:
            print("   Stable-Baselines3 not available, using fallback SA optimizer...")
            return EnergyAwareOptimizer.simulated_annealing_tsp(
                waypoints, start_pos, planner, objective=objective, max_iterations=500
            )

        if not waypoints:
            return []

        # Create environment
        env = WaypointSequencingEnv(waypoints, start_pos, planner, objective)

        # Select algorithm
        if algorithm == 'PPO':
            model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003)
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=0, learning_rate=0.0003)
        else:  # SAC (for continuous but we adapt)
            model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003)

        # Train the agent (with periodic GC to prevent memory issues)
        model.learn(total_timesteps=total_timesteps)
        gc.collect()  # Clean up training memory

        # Generate optimized sequence
        obs, _ = env.reset()
        sequence = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        return env.sequence

class GridConstrainedPlanner:
    """Plans grid-compliant paths for vineyard navigation"""

    def __init__(self, grid: VineyardGrid, elevation_model: Optional[ElevationModel] = None):
        self.grid = grid
        self.elevation = elevation_model
        self.velocity = 1.0  # m/s
        self.turn_time = 6.0  # seconds per 90° turn

        # UGV physical parameters for energy calculation
        self.ugv_mass = 120.0  # kg (typical agricultural UGV)
        self.g = 9.81  # m/s^2 (gravitational acceleration)
        self.rolling_resistance = 0.15  # coefficient for rough terrain
        self.mechanical_efficiency = 0.75  # motor/drivetrain efficiency
        
    def sequence_waypoints(self, waypoints: List[Waypoint], start_pos: Tuple[float, float], mode: str = 'nearest_neighbor', **kwargs) -> List[Waypoint]:
        """
        Phase 1: Sequence waypoints using different strategies

        Parameters:
        -----------
        waypoints: List of waypoints to sequence
        start_pos: Starting position
        mode: Sequencing strategy:
            - 'nearest_neighbor': Greedy alley clustering
            - 'elevation': Greedy elevation minimization
            - 'simulated_annealing': SA optimization for energy/time/distance
            - 'clustering': Elevation-zone clustering with SA
            - 'two_opt': 2-opt local search improvement
            - 'multi_objective': Multi-objective energy+time optimization
            - 'rl': Reinforcement Learning with Stable-Baselines3 (PPO/A2C)
            - 'scipy': SciPy global optimization (dual_annealing/differential_evolution)
            - 'custom': Use provided custom_sequence
        **kwargs: Additional parameters for specific modes

        Returns:
        --------
        Ordered list of waypoints
        """
        if mode == 'custom' and 'custom_sequence' in kwargs:
            return kwargs['custom_sequence']
        elif mode == 'elevation':
            return self._sequence_by_elevation(waypoints, start_pos)
        elif mode == 'simulated_annealing':
            objective = kwargs.get('objective', 'energy')
            max_iterations = kwargs.get('max_iterations', 1000)
            return EnergyAwareOptimizer.simulated_annealing_tsp(
                waypoints, start_pos, self, objective, max_iterations
            )
          
        elif mode == 'two_opt':
            # Start with nearest neighbor and improve
            initial_seq = self._sequence_by_nearest_neighbor(waypoints, start_pos)
            return EnergyAwareOptimizer.two_opt_improve(
                initial_seq, start_pos, self, objective='energy'
            )
        elif mode == 'multi_objective':
            energy_weight = kwargs.get('energy_weight', 0.7)
            time_weight = kwargs.get('time_weight', 0.3)
            best_seq, pareto = EnergyAwareOptimizer.multi_objective_optimize(
                waypoints, start_pos, self, energy_weight, time_weight
            )
            return best_seq
        elif mode == 'rl':
            # Reinforcement Learning optimization
            objective = kwargs.get('objective', 'energy')
            algorithm = kwargs.get('algorithm', 'PPO')
            timesteps = kwargs.get('timesteps', 10000)
            return RLOptimizer.train_rl_agent(
                waypoints, start_pos, self, objective, algorithm, timesteps
            )
        elif mode == 'scipy':
            # SciPy global optimization
            objective = kwargs.get('objective', 'energy')
            method = kwargs.get('scipy_method', 'dual_annealing')
            return EnergyAwareOptimizer.scipy_optimize_tsp(
                waypoints, start_pos, self, objective, method
            )
        else:  # nearest_neighbor (default)
            return self._sequence_by_nearest_neighbor(waypoints, start_pos)

    def _sequence_by_nearest_neighbor(self, waypoints: List[Waypoint], start_pos: Tuple[float, float]) -> List[Waypoint]:
        """
        Sequence waypoints using alley clustering and nearest neighbor
        """
        # Cluster waypoints by alley (row_id now represents alley between rows)
        alley_clusters = defaultdict(list)
        for wp in waypoints:
            alley_clusters[wp.row_id].append(wp)

        # Sort waypoints within each alley by x-coordinate
        for alley_id in alley_clusters:
            alley_clusters[alley_id].sort(key=lambda w: w.x)

        # Determine alley visitation order using nearest neighbor
        start_alley = self.grid.get_nearest_alley(start_pos[1])
        unvisited_alleys = set(alley_clusters.keys())
        alley_order = []
        current_alley = start_alley

        # Find nearest unvisited alley iteratively
        while unvisited_alleys:
            if current_alley in unvisited_alleys:
                next_alley = current_alley
            else:
                # Find nearest unvisited alley
                next_alley = min(unvisited_alleys,
                             key=lambda a: abs(a - current_alley))
            alley_order.append(next_alley)
            unvisited_alleys.remove(next_alley)
            current_alley = next_alley

        # Build final sequence alternating direction for efficiency
        sequence = []
        for i, alley_id in enumerate(alley_order):
            alley_waypoints = alley_clusters[alley_id]
            if i % 2 == 1:  # Reverse every other alley for zigzag
                alley_waypoints = alley_waypoints[::-1]
            sequence.extend(alley_waypoints)

        return sequence

    def _sequence_by_elevation(self, waypoints: List[Waypoint], start_pos: Tuple[float, float]) -> List[Waypoint]:
        """
        Sequence waypoints to minimize elevation changes (greedy approach)
        """
        if not waypoints:
            return []

        # Get starting elevation
        current_z = self.elevation.get_elevation(start_pos[0], start_pos[1]) if self.elevation else 0

        unvisited = set(range(len(waypoints)))
        sequence = []

        # Greedy selection: always choose waypoint with smallest elevation change
        while unvisited:
            # Find waypoint with minimum elevation difference from current position
            min_diff = float('inf')
            best_idx = None

            for idx in unvisited:
                wp = waypoints[idx]
                elev_diff = abs(wp.z - current_z)

                if elev_diff < min_diff:
                    min_diff = elev_diff
                    best_idx = idx

            # Add best waypoint to sequence
            sequence.append(waypoints[best_idx])
            current_z = waypoints[best_idx].z
            unvisited.remove(best_idx)

        return sequence
    
    def plan_grid_route(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> List[GridSegment]:
        """
        Plan grid-compliant route between two points with elevation data.
        Routes now travel in alleys (between rows) instead of on rows.
        """
        x1, y1 = p1
        x2, y2 = p2
        alley1 = self.grid.get_nearest_alley(y1)
        alley2 = self.grid.get_nearest_alley(y2)

        segments = []

        if alley1 == alley2:
            # Same alley - direct horizontal travel along the alley
            z1 = self.elevation.get_elevation(x1, y1) if self.elevation else 0
            z2 = self.elevation.get_elevation(x2, y2) if self.elevation else 0
            segments.append(GridSegment(
                type=SegmentType.ALONG_ROW,  # Keep same type for compatibility
                start=p1,
                end=p2,
                row_id=alley1,
                elevation_start=z1,
                elevation_end=z2
            ))
        else:
            # Different alleys - must use headland
            cost_left = abs(x1 - 0) + abs(alley2 - alley1) * self.grid.row_spacing + abs(x2 - 0)
            cost_right = abs(x1 - self.grid.row_length) + \
                        abs(alley2 - alley1) * self.grid.row_spacing + \
                        abs(x2 - self.grid.row_length)

            if cost_left < cost_right:
                headland_x = 0
            else:
                headland_x = self.grid.row_length

            # Navigate to headland (stay in current alley)
            z1 = self.elevation.get_elevation(x1, y1) if self.elevation else 0
            z_h1 = self.elevation.get_elevation(headland_x, y1) if self.elevation else 0
            segments.append(GridSegment(
                type=SegmentType.ALONG_ROW,
                start=p1,
                end=(headland_x, y1),
                row_id=alley1,
                elevation_start=z1,
                elevation_end=z_h1
            ))

            # Travel along headland to reach target alley
            alley2_y = self.grid.get_alley_center_y(alley2)
            z_h2 = self.elevation.get_elevation(headland_x, alley2_y) if self.elevation else 0
            segments.append(GridSegment(
                type=SegmentType.ALONG_HEADLAND,
                start=(headland_x, y1),
                end=(headland_x, alley2_y),
                elevation_start=z_h1,
                elevation_end=z_h2
            ))

            # Enter target alley
            z2 = self.elevation.get_elevation(x2, y2) if self.elevation else 0
            segments.append(GridSegment(
                type=SegmentType.ALONG_ROW,
                start=(headland_x, alley2_y),
                end=(x2, y2),
                row_id=alley2,
                elevation_start=z_h2,
                elevation_end=z2
            ))

        return segments
    
    def generate_spur_maneuver(self, waypoint: Waypoint) -> List[GridSegment]:
        """
        Generate spur maneuver to access waypoint with elevation.
        Waypoints are in alleys (between rows), so spur moves from alley center to inspection point and back.
        Returns a single round-trip segment.
        """
        # Get the center of the alley
        alley_center_y = self.grid.get_alley_center_y(waypoint.row_id)
        entry_point = (waypoint.x, alley_center_y)
        waypoint_pos = (waypoint.x, waypoint.y)

        z_entry = self.elevation.get_elevation(waypoint.x, alley_center_y) if self.elevation else 0
        z_wp = waypoint.z if waypoint.z != 0 else (self.elevation.get_elevation(waypoint.x, waypoint.y) if self.elevation else 0)

        # Single round-trip segment (entry + exit combined)
        # The segment goes from entry_point to waypoint_pos and back
        # Distance calculation will account for both directions
        segments = [
            GridSegment(
                type=SegmentType.SPUR,
                start=entry_point,
                end=waypoint_pos,
                elevation_start=z_entry,
                elevation_end=z_wp
            )
        ]

        return segments
    
    def plan_complete_tour(self, waypoints: List[Waypoint],
                          start_pos: Tuple[float, float],
                          sequencing_mode: str = 'elevation',
                          **kwargs) -> Tuple[List[GridSegment], dict, List[Waypoint]]:
        """
        Plan complete tour visiting all waypoints in alleys between rows

        Parameters:
        -----------
        waypoints: List of waypoints to visit
        start_pos: Starting position
        sequencing_mode: Waypoint ordering strategy (see sequence_waypoints for options)
        **kwargs: Additional parameters passed to sequence_waypoints

        Returns:
        --------
        Tuple of (segments, metrics, sequenced_waypoints)
        """
        # Phase 1: Sequence waypoints
        sequence = self.sequence_waypoints(waypoints, start_pos, mode=sequencing_mode, **kwargs)

        # Phase 2: Generate grid-compliant path
        all_segments = []
        current_pos = start_pos

        for wp in sequence:
            # Navigate to waypoint alley center position
            alley_center_y = self.grid.get_alley_center_y(wp.row_id)
            access_point = (wp.x, alley_center_y)

            # Grid route to access point
            grid_segments = self.plan_grid_route(current_pos, access_point)
            all_segments.extend(grid_segments)

            # Spur maneuver (move from alley center to inspection point and back)
            spur_segments = self.generate_spur_maneuver(wp)
            all_segments.extend(spur_segments)

            current_pos = access_point

        # Return to start position (complete the tour)
        return_segments = self.plan_grid_route(current_pos, start_pos)
        all_segments.extend(return_segments)

        # Calculate metrics
        metrics = self.calculate_metrics(all_segments)

        return all_segments, metrics, sequence
    
    def calculate_energy_for_segment(self, seg: GridSegment) -> dict:
        """
        Calculate energy consumption for a single segment with improved accuracy.

        Improvements:
        - Uses 3D distance for rolling resistance
        - Considers slope angle effect on rolling resistance
        - Variable motor efficiency based on load
        - Includes terrain-specific coefficients

        Returns:
        --------
        dict with energy components in Joules (J)
        """
        # Use 3D distance for more accurate calculations
        distance_2d = seg.distance()

        # Elevation change
        if seg.elevation_start is not None and seg.elevation_end is not None:
            elevation_change = seg.elevation_end - seg.elevation_start
            # Calculate 3D distance (accounting for slope)
            distance_3d = np.sqrt(distance_2d**2 + elevation_change**2)
        else:
            elevation_change = 0.0
            distance_3d = distance_2d

        # Calculate slope angle (for improved physics)
        slope_angle = np.arctan2(abs(elevation_change), distance_2d) if distance_2d > 0 else 0.0

        # 1. Potential energy change (climbing/descending)
        # E_potential = m * g * Δh
        potential_energy = self.ugv_mass * self.g * elevation_change

        # 2. Rolling resistance energy (uses 3D distance and slope effect)
        # On slopes, normal force = m * g * cos(θ), so rolling resistance = m * g * cos(θ) * C_rr * distance
        # For small angles, cos(θ) ≈ 1, but we calculate it for accuracy
        normal_force_factor = np.cos(slope_angle)
        rolling_energy = self.ugv_mass * self.g * self.rolling_resistance * distance_3d * normal_force_factor

        # 3. Air resistance is disabled (set to 0 for low-speed UGV)
        air_resistance_energy = 0.0

        # 4. Climbing vs Descending Energy
        if elevation_change > 0:
            # CLIMBING: Need to provide gravitational potential energy
            climbing_energy = potential_energy

            # Variable motor efficiency based on load
            # Higher loads (climbing steep slopes) = lower efficiency
            load_factor = 1.0 + abs(elevation_change) / distance_3d if distance_3d > 0 else 1.0
            motor_efficiency = max(0.60, self.mechanical_efficiency - 0.05 * (load_factor - 1.0))

            # Total mechanical energy
            mechanical_energy = climbing_energy + rolling_energy + air_resistance_energy
            total_energy = mechanical_energy / motor_efficiency

        else:
            # DESCENDING: Only rolling resistance energy consumed (no climbing energy)
            climbing_energy = 0.0

            # Total mechanical energy (rolling resistance only on descents)
            mechanical_energy = rolling_energy + air_resistance_energy

            # Use standard motor efficiency
            motor_efficiency = self.mechanical_efficiency
            total_energy = mechanical_energy / motor_efficiency

        return {
            'distance': distance_2d,
            'distance_3d': distance_3d,
            'elevation_change': elevation_change,
            'slope_angle_deg': np.degrees(slope_angle),
            'potential_energy': potential_energy,
            'climbing_energy': climbing_energy,
            'rolling_energy': rolling_energy,
            'air_resistance_energy': air_resistance_energy,
            'mechanical_energy': mechanical_energy,
            'total_energy': total_energy,
            'motor_efficiency': motor_efficiency if elevation_change > 0 else self.mechanical_efficiency
        }

    def calculate_metrics(self, segments: List[GridSegment]) -> dict:
        """
        Calculate path metrics including distance, time, and energy.

        Parameters:
        -----------
        segments: List of path segments

        Returns:
        --------
        dict with comprehensive metrics
        """
        # Calculate both 2D and 3D distances (use elevation model for accurate 3D calculation)
        total_distance_2d = sum(seg.distance() for seg in segments)
        total_distance_3d = sum(seg.distance_3d(self.elevation) for seg in segments)
        total_distance = total_distance_3d  # Use 3D distance as actual travel distance

        # Count turns between segments
        num_turns = 0
        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            seg = segments[i]
            # Check if direction changed (simplified)
            if prev_seg.type != seg.type:
                if prev_seg.type in [SegmentType.ALONG_ROW, SegmentType.ALONG_HEADLAND] and \
                   seg.type in [SegmentType.ALONG_ROW, SegmentType.ALONG_HEADLAND]:
                    num_turns += 1
                elif seg.type == SegmentType.SPUR:
                    num_turns += 2  # SPUR has entry and exit turns

        # Calculate time
        travel_time = total_distance / self.velocity
        turn_time = num_turns * self.turn_time
        total_time = travel_time + turn_time

        # Calculate energy consumption
        total_energy = 0.0
        total_climbing_energy = 0.0
        total_rolling_energy = 0.0
        total_air_energy = 0.0
        total_elevation_gain = 0.0
        total_elevation_loss = 0.0

        for seg in segments:
            energy_data = self.calculate_energy_for_segment(seg)
            total_energy += energy_data['total_energy']
            total_climbing_energy += energy_data['climbing_energy']
            total_rolling_energy += energy_data['rolling_energy']
            total_air_energy += energy_data['air_resistance_energy']

            # Track elevation changes
            if energy_data['elevation_change'] > 0:
                total_elevation_gain += energy_data['elevation_change']
            else:
                total_elevation_loss += abs(energy_data['elevation_change'])

        # Energy for turns (motors working during rotation)
        # Assume ~50W power during a turn
        turn_energy = num_turns * 50.0 * self.turn_time  # Watts * seconds = Joules

        total_energy += turn_energy

        # Convert to more readable units
        energy_kj = total_energy / 1000.0  # Kilojoules
        energy_kwh = total_energy / 3600000.0  # Kilowatt-hours

        metrics = {
            'total_distance': total_distance,
            'total_distance_2d': total_distance_2d,
            'total_distance_3d': total_distance_3d,
            'num_turns': num_turns,
            'travel_time': travel_time,
            'turn_time': turn_time,
            'total_time': total_time,
            'total_energy_j': total_energy,
            'total_energy_kj': energy_kj,
            'total_energy_kwh': energy_kwh,
            'climbing_energy_j': total_climbing_energy,
            'rolling_energy_j': total_rolling_energy,
            'air_resistance_energy_j': total_air_energy,
            'turn_energy_j': turn_energy,
            'total_elevation_gain': total_elevation_gain,
            'total_elevation_loss': total_elevation_loss
        }

        return metrics

class VineyardVisualizer:
    """Visualizes the vineyard and planned paths"""
    
    def __init__(self, grid: VineyardGrid):
        self.grid = grid
        
    def plot_solution(self, waypoints: List[Waypoint],
                     segments: List[GridSegment],
                     metrics: dict,
                     start_pos: Tuple[float, float],
                     save_as: Optional[str] = 'vineyard_solution.png'):
        """Create comprehensive visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Vineyard structure and path
        self._plot_vineyard_structure(ax1)
        self._plot_waypoints(ax1, waypoints)
        self._plot_path(ax1, segments)
        self._plot_start(ax1, start_pos)

        ax1.set_xlabel('Distance along row (m) - Upslope Direction')
        ax1.set_ylabel('Cross-row distance (m)')
        ax1.set_title('Mosel Valley Vineyard Navigation')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right')

        # Right plot: Path segments breakdown
        self._plot_segment_analysis(ax2, segments)
        self._add_metrics_text(ax2, metrics)

        ax2.set_title('Path Segment Analysis')
        ax2.set_xlabel('Segment Index')
        ax2.set_ylabel('Distance (m)')

        plt.tight_layout()

        # Save figure
        if save_as:
            fig.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"     ✓ Saved: {save_as}")

        plt.close(fig)
        return fig
    
    def _plot_vineyard_structure(self, ax, show_trees=True):
        """Plot vineyard rows, headlands, trees, and alleys"""
        # Plot rows (vine rows)
        for i in range(self.grid.num_rows):
            y = self.grid.get_row_y(i)
            ax.plot([0, self.grid.row_length], [y, y],
                   'g-', alpha=0.3, linewidth=0.5)
            # Add row labels
            if i % 5 == 0:
                ax.text(-5, y, f'Row {i}', fontsize=8, ha='right')

        # Plot alleys (spaces between rows where UGV travels)
        for i in range(self.grid.num_rows - 1):
            alley_y = self.grid.get_alley_center_y(i)
            ax.plot([0, self.grid.row_length], [alley_y, alley_y],
                   'b--', alpha=0.2, linewidth=1,
                   label='Alleys (UGV path)' if i == 0 else '')

        # Plot vine trees
        if show_trees:
            for row_id, trees in self.grid.tree_positions.items():
                tree_x = [t[0] for t in trees]
                tree_y = [t[1] for t in trees]
                ax.plot(tree_x, tree_y, 'o', color='darkgreen',
                       markersize=3, alpha=0.4,
                       label='Vine Trees' if row_id == 0 else '')

        # Plot headlands
        ax.plot([0, 0], [0, self.grid.get_row_y(self.grid.num_rows-1)],
               'k-', linewidth=2, label='Headlands')
        ax.plot([self.grid.row_length, self.grid.row_length],
               [0, self.grid.get_row_y(self.grid.num_rows-1)],
               'k-', linewidth=2)

        # Shade headland areas
        ax.fill_between([-5, 0], 0, self.grid.get_row_y(self.grid.num_rows-1),
                       alpha=0.1, color='gray')
        ax.fill_between([self.grid.row_length, self.grid.row_length+5],
                       0, self.grid.get_row_y(self.grid.num_rows-1),
                       alpha=0.1, color='gray')
    
    def _plot_waypoints(self, ax, waypoints, waypoint_visit_order=None):
        """Plot waypoint locations with optional visit order labels"""
        for i, wp in enumerate(waypoints):
            ax.plot(wp.x, wp.y, 'ro', markersize=8, label='Waypoints' if i == 0 else '')
            # Use visit order if available, otherwise use original index
            if waypoint_visit_order and (wp.x, wp.y, wp.row_id) in waypoint_visit_order:
                label_num = waypoint_visit_order[(wp.x, wp.y, wp.row_id)]
            else:
                label_num = i + 1
            ax.text(wp.x + 1, wp.y + 0.5, f'W{label_num}', fontsize=8)
    
    def _plot_path(self, ax, segments, show_interpolated=True):
        """Plot the planned path with optional interpolated waypoints"""
        plotted_types = set()

        for seg in segments:
            color = SEGMENT_COLORS.get(seg.type, 'black')
            style = '-' if seg.type in [SegmentType.ALONG_ROW, SegmentType.ALONG_HEADLAND] else '--'

            label = None
            if seg.type not in plotted_types:
                label = seg.type.value.replace('_', ' ').title()
                plotted_types.add(seg.type)

            # Plot interpolated path instead of straight line
            if show_interpolated:
                x_path, y_path, _ = seg.interpolate()
                ax.plot(x_path, y_path,
                       color=color, linestyle=style, linewidth=2,
                       label=label, alpha=0.7)

                # Plot interpolated waypoints as small dots
                ax.plot(x_path, y_path, 'o', color=color,
                       markersize=2, alpha=0.3)
            else:
                ax.plot([seg.start[0], seg.end[0]],
                       [seg.start[1], seg.end[1]],
                       color=color, linestyle=style, linewidth=2,
                       label=label, alpha=0.7)

            # Add direction arrows (optimized)
            if seg.distance() > 5:  # Only for longer segments
                mid_x = (seg.start[0] + seg.end[0]) / 2
                mid_y = (seg.start[1] + seg.end[1]) / 2
                dx = seg.end[0] - seg.start[0]
                dy = seg.end[1] - seg.start[1]
                ax.arrow(mid_x - dx*0.1, mid_y - dy*0.1,
                        dx*0.2, dy*0.2,
                        head_width=1, head_length=0.5,
                        fc=color, ec=color, alpha=0.5)
    
    def _plot_start(self, ax, start_pos):
        """Plot starting position"""
        ax.plot(start_pos[0], start_pos[1], 'gs', markersize=12, label='Start')
    
    def _plot_segment_analysis(self, ax, segments):
        """Plot segment distance breakdown (optimized)"""
        # Vectorized distance calculation
        segment_distances = [seg.distance() for seg in segments]
        segment_types = [seg.type for seg in segments]

        # Create stacked bar chart by type
        indices = np.arange(len(segments))

        for seg_type in SegmentType:
            distances = [d if t == seg_type else 0
                        for d, t in zip(segment_distances, segment_types)]
            if any(distances):
                ax.bar(indices, distances, label=seg_type.value,
                      color=SEGMENT_COLORS.get(seg_type, 'gray'), alpha=0.7)

        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _add_metrics_text(self, ax, metrics):
        """Add metrics text to plot"""
        text = f"Performance Metrics:\n"
        text += f"━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"Distance 2D: {metrics['total_distance_2d']:.1f} m\n"
        text += f"Distance 3D: {metrics['total_distance_3d']:.1f} m\n"
        text += f"Turns: {metrics['num_turns']}\n"
        text += f"Travel Time: {metrics['travel_time']:.1f} s\n"
        text += f"Turn Time: {metrics['turn_time']:.1f} s\n"
        text += f"Total Time: {metrics['total_time']:.1f} s ({metrics['total_time']/60:.1f} min)\n"
        text += f"Avg Speed: {metrics['total_distance']/metrics['total_time']:.2f} m/s\n\n"

        text += f"Energy Consumption:\n"
        text += f"━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"Total Energy: {metrics['total_energy_kj']:.2f} kJ\n"
        text += f"             ({metrics['total_energy_kwh']*1000:.2f} Wh)\n"
        text += f"Climbing: {metrics['climbing_energy_j']/1000:.2f} kJ\n"
        text += f"Rolling: {metrics['rolling_energy_j']/1000:.2f} kJ\n"
        text += f"Turns: {metrics['turn_energy_j']/1000:.2f} kJ\n"

        text += f"\nElevation:\n"
        text += f"━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"Total Gain: {metrics['total_elevation_gain']:.1f} m\n"
        text += f"Total Loss: {metrics['total_elevation_loss']:.1f} m"

        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def animate_ugv_path(self, waypoints: List[Waypoint],
                        segments: List[GridSegment],
                        start_pos: Tuple[float, float],
                        elevation_model: Optional[ElevationModel] = None,
                        planner: Optional['GridConstrainedPlanner'] = None,
                        save_as: Optional[str] = None,
                        sequenced_waypoints: Optional[List[Waypoint]] = None):
        """
        Create animation of UGV traveling along the planned path

        Parameters:
        -----------
        waypoints: List of waypoints to visit (original unordered list)
        segments: Path segments to follow
        start_pos: Starting position
        elevation_model: Optional elevation model for terrain visualization
        planner: Optional planner for UGV parameters (mass, velocity, etc.)
        save_as: Optional filename to save animation (e.g., 'ugv_animation.gif')
        sequenced_waypoints: Optional list of waypoints in visit order (for labeling)
        """
        # Default UGV parameters if planner not provided
        if planner is None:
            ugv_mass = 10.0
            ugv_g = 9.81
            rolling_resistance = 0.15
            mechanical_efficiency = 0.75
            velocity = 1.5
        else:
            ugv_mass = planner.ugv_mass
            ugv_g = planner.g
            rolling_resistance = planner.rolling_resistance
            mechanical_efficiency = planner.mechanical_efficiency
            velocity = planner.velocity
        # Create waypoint visit order mapping
        waypoint_visit_order = {}
        if sequenced_waypoints is not None:
            for visit_order, wp in enumerate(sequenced_waypoints, start=1):
                # Create a unique key for each waypoint based on position
                waypoint_visit_order[(wp.x, wp.y, wp.row_id)] = visit_order

        # Create figure with 4 subplots (2D path, elevation, energy, distance)
        fig = plt.figure(figsize=(24, 12))
        ax1 = plt.subplot(2, 2, 1)  # 2D top view
        ax2 = plt.subplot(2, 2, 2)  # Elevation profile
        ax3 = plt.subplot(2, 2, 3)  # Energy profile
        ax4 = plt.subplot(2, 2, 4)  # Distance profile

        # Plot static vineyard structure on ax1
        self._plot_vineyard_structure(ax1)
        self._plot_waypoints(ax1, waypoints, waypoint_visit_order)
        self._plot_path(ax1, segments)

        ax1.set_xlabel('Distance along row (m) - Upslope Direction')
        ax1.set_ylabel('Cross-row distance (m)')
        ax1.set_title('Mosel Valley UGV Path Animation - Top View')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize=8)

        # Initialize UGV marker
        ugv, = ax1.plot([], [], 'go', markersize=15, label='UGV', markeredgecolor='darkgreen', markeredgewidth=2)
        ugv_trail, = ax1.plot([], [], 'g-', linewidth=1, alpha=0.5)

        # Add text for real-time metrics
        metrics_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes,
                               fontsize=11, verticalalignment='bottom',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                               family='monospace')

        # Prepare elevation profile
        if elevation_model:
            # Calculate path distances and elevations
            path_distances = [0]
            path_elevations = []
            cumulative_dist = 0

            for seg in segments:
                dx = seg.end[0] - seg.start[0]
                dy = seg.end[1] - seg.start[1]
                dist = np.sqrt(dx**2 + dy**2)
                cumulative_dist += dist
                path_distances.append(cumulative_dist)

                if seg.elevation_start is not None:
                    path_elevations.append(seg.elevation_start)
                else:
                    path_elevations.append(elevation_model.get_elevation(seg.start[0], seg.start[1]))

            # Add final elevation
            last_seg = segments[-1]
            if last_seg.elevation_end is not None:
                path_elevations.append(last_seg.elevation_end)
            else:
                path_elevations.append(elevation_model.get_elevation(last_seg.end[0], last_seg.end[1]))

            # Plot elevation profile
            ax2.plot(path_distances, path_elevations, 'b-', linewidth=2, alpha=0.7)
            ax2.fill_between(path_distances, min(path_elevations), path_elevations, alpha=0.3)
            ax2.set_xlabel('Distance along path (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Elevation Profile')
            ax2.grid(True, alpha=0.3)

            # Current position marker on elevation profile
            elev_marker, = ax2.plot([], [], 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        else:
            ax2.text(0.5, 0.5, 'No Elevation Data', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            elev_marker = None
            path_distances = None
            path_elevations = None

        # Setup Energy Profile (ax3) - will be populated during animation
        ax3.set_xlabel('Distance along path (m)')
        ax3.set_ylabel('Cumulative Energy (kJ)')
        ax3.set_title('Energy Consumption Profile')
        ax3.grid(True, alpha=0.3)
        energy_line, = ax3.plot([], [], 'r-', linewidth=2, label='Energy')
        energy_marker, = ax3.plot([], [], 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
        ax3.legend(loc='upper left')

        # Setup Distance Profile (ax4) - will show cumulative distance over time
        ax4.set_xlabel('Time (frames)')
        ax4.set_ylabel('Cumulative Distance (m)')
        ax4.set_title('Distance Traveled Profile')
        ax4.grid(True, alpha=0.3)
        distance_line, = ax4.plot([], [], 'g-', linewidth=2, label='Distance')
        distance_marker, = ax4.plot([], [], 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2)
        ax4.legend(loc='upper left')

        # Build interpolated path points for smooth animation (optimized)
        path_points = []
        path_point_distances = []
        path_point_energies = []
        cumulative_dist = 0
        cumulative_energy = 0

        for seg in segments:
            # Use fewer interpolation points for faster animation
            num_points = max(5, int(seg.distance() / 2))  # Reduced from default
            x_path, y_path, t = seg.interpolate(num_points=num_points)

            # Calculate elevation for energy computation
            if seg.elevation_start is not None and seg.elevation_end is not None:
                z_path = seg.elevation_start + t * (seg.elevation_end - seg.elevation_start)
            else:
                z_path = np.zeros_like(x_path)

            for i, (x, y, z) in enumerate(zip(x_path, y_path, z_path)):
                path_points.append((x, y))

                # Calculate incremental distance and energy
                if len(path_points) > 1:
                    dx = x - path_points[-2][0]
                    dy = y - path_points[-2][1]
                    dz = z - z_path[i-1] if i > 0 else 0

                    # 3D distance
                    dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
                    cumulative_dist += dist_3d

                    # Energy calculation for this small segment
                    # Potential energy
                    potential_energy = max(0, ugv_mass * ugv_g * dz)
                    # Rolling resistance
                    rolling_energy = ugv_mass * ugv_g * rolling_resistance * dist_3d
                    # Total mechanical energy with efficiency (air resistance removed)
                    segment_energy = (potential_energy + rolling_energy) / mechanical_efficiency

                    cumulative_energy += segment_energy

                path_point_distances.append(cumulative_dist)
                path_point_energies.append(cumulative_energy)

        # Animation function
        trail_x, trail_y = [], []

        def animate(frame):
            if frame < len(path_points):
                x, y = path_points[frame]
                trail_x.append(x)
                trail_y.append(y)

                ugv.set_data([x], [y])
                ugv_trail.set_data(trail_x, trail_y)

                # Update metrics text
                dist_traveled = path_point_distances[frame]
                energy_used = path_point_energies[frame]
                progress_pct = (frame / len(path_points)) * 100

                metrics_str = f"Distance: {dist_traveled:.1f} m\n"
                metrics_str += f"Energy: {energy_used/1000:.2f} kJ\n"
                metrics_str += f"Progress: {progress_pct:.0f}%"
                metrics_text.set_text(metrics_str)

                # Update elevation marker
                if elev_marker and path_distances:
                    dist = path_point_distances[frame]
                    # Interpolate elevation at this distance
                    elev = np.interp(dist, path_distances, path_elevations)
                    elev_marker.set_data([dist], [elev])

                # Update energy profile (ax3)
                current_distances = path_point_distances[:frame+1]
                current_energies = [e/1000 for e in path_point_energies[:frame+1]]  # Convert to kJ
                energy_line.set_data(current_distances, current_energies)
                energy_marker.set_data([dist_traveled], [energy_used/1000])
                ax3.set_xlim(0, max(path_point_distances) if path_point_distances else 1)
                ax3.set_ylim(0, max(current_energies) * 1.1 if current_energies else 1)

                # Update distance profile (ax4)
                frames_so_far = list(range(frame+1))
                distance_line.set_data(frames_so_far, current_distances)
                distance_marker.set_data([frame], [dist_traveled])
                ax4.set_xlim(0, len(path_points))
                ax4.set_ylim(0, max(path_point_distances) * 1.1 if path_point_distances else 1)

            artists = [ugv, ugv_trail, metrics_text, energy_line, energy_marker, distance_line, distance_marker]
            if elev_marker:
                artists.append(elev_marker)
            return artists

        # Create animation (optimized with faster frame rate)
        anim = FuncAnimation(fig, animate, frames=len(path_points),
                           interval=30, blit=True, repeat=True)

        # Save if requested
        if save_as:
            print(f"Saving animation to {save_as}...")
            # Faster frame rate and reduced DPI for faster saving and less memory
            writer = PillowWriter(fps=20)
            anim.save(save_as, writer=writer, dpi=60)
            print(f"Animation saved!")
            gc.collect()  # Clean up after saving

        plt.tight_layout()
        return fig, anim

    def plot_3d_terrain(self, waypoints: List[Waypoint],
                       segments: List[GridSegment],
                       start_pos: Tuple[float, float],
                       elevation_model: ElevationModel,
                       save_as: Optional[str] = 'vineyard_3d_terrain.png',
                       sequenced_waypoints: Optional[List[Waypoint]] = None):
        """
        Create 3D visualization of terrain with UGV path

        Parameters:
        -----------
        waypoints: List of waypoints to visit (original unordered list)
        segments: Path segments to follow
        start_pos: Starting position
        elevation_model: Elevation model for terrain surface
        save_as: Optional filename to save figure
        sequenced_waypoints: Optional list of waypoints in visit order (for labeling)
        """
        fig = plt.figure(figsize=(16, 12))

        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Create waypoint visit order mapping
        waypoint_visit_order = {}
        if sequenced_waypoints is not None:
            for visit_order, wp in enumerate(sequenced_waypoints, start=1):
                waypoint_visit_order[(wp.x, wp.y, wp.row_id)] = visit_order

        # Plot terrain surface
        surf = ax.plot_surface(elevation_model.X, elevation_model.Y,
                              elevation_model.elevation,
                              cmap='terrain', alpha=0.6,
                              linewidth=0, antialiased=True,
                              vmin=elevation_model.elevation.min(),
                              vmax=elevation_model.elevation.max())

        # Add colorbar for elevation
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Elevation (m)', rotation=270, labelpad=20)

        # Plot vineyard rows as lines on terrain (vectorized)
        for i in range(self.grid.num_rows):
            y_row = self.grid.get_row_y(i)
            x_row = np.linspace(0, self.grid.row_length, 50)
            y_vals = np.full_like(x_row, y_row)
            z_vals = elevation_model.get_elevations_vectorized(x_row, y_vals)
            ax.plot(x_row, y_vals, z_vals, 'g-', alpha=0.3, linewidth=0.5)

        # Plot vine trees on terrain
        for row_id, trees in self.grid.tree_positions.items():
            tree_x = np.array([t[0] for t in trees])
            tree_y = np.array([t[1] for t in trees])
            tree_z = elevation_model.get_elevations_vectorized(tree_x, tree_y)
            ax.scatter(tree_x, tree_y, tree_z, c='darkgreen', s=10,
                      marker='o', alpha=0.4, zorder=2,
                      label='Vine Trees' if row_id == 0 else '')

        # Plot headlands (vectorized)
        for headland_x in [0, self.grid.row_length]:
            y_headland = np.linspace(0, self.grid.get_row_y(self.grid.num_rows-1), 50)
            x_vals = np.full_like(y_headland, headland_x)
            z_vals = elevation_model.get_elevations_vectorized(x_vals, y_headland)
            ax.plot(x_vals, y_headland, z_vals, 'k-', alpha=0.5, linewidth=2)

        # Plot waypoints in 3D with numbers
        for i, wp in enumerate(waypoints):
            ax.scatter(wp.x, wp.y, wp.z, c='red', s=100, marker='o',
                      edgecolors='darkred', linewidth=2, zorder=5)
            # Use visit order if available, otherwise use original index
            if waypoint_visit_order and (wp.x, wp.y, wp.row_id) in waypoint_visit_order:
                label_num = waypoint_visit_order[(wp.x, wp.y, wp.row_id)]
            else:
                label_num = i + 1
            # Add waypoint number label with better styling
            ax.text(wp.x, wp.y, wp.z + 2, f'{label_num}', fontsize=9,
                   color='white', weight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red',
                           edgecolor='darkred', alpha=0.8), zorder=6)

        # Plot start position
        z_start = elevation_model.get_elevation(start_pos[0], start_pos[1])
        ax.scatter(start_pos[0], start_pos[1], z_start, c='lime', s=200,
                  marker='s', edgecolors='darkgreen', linewidth=2, label='Start', zorder=5)

        # Plot path segments in 3D (optimized)
        plotted_types = set()

        for seg in segments:
            # Use optimized interpolation
            x_path, y_path, _ = seg.interpolate()

            # Always query actual elevation from the model for accurate terrain following
            # This ensures the path follows the hill contours instead of being a straight line
            z_path = elevation_model.get_elevations_vectorized(x_path, y_path)

            # Slightly elevate path above terrain for visibility
            z_path = z_path + 0.2

            color = SEGMENT_COLORS.get(seg.type, 'black')
            style = '-' if seg.type in [SegmentType.ALONG_ROW, SegmentType.ALONG_HEADLAND] else '--'

            label = None
            if seg.type not in plotted_types:
                label = seg.type.value.replace('_', ' ').title()
                plotted_types.add(seg.type)

            ax.plot(x_path, y_path, z_path, color=color, linestyle=style,
                   linewidth=3, label=label, alpha=0.8, zorder=4)

        # Set labels and title
        ax.set_xlabel('Distance along row (m) - Upslope Direction', fontsize=10, labelpad=10)
        ax.set_ylabel('Cross-row distance (m)', fontsize=10, labelpad=10)
        ax.set_zlabel('Elevation (m ASL)', fontsize=10, labelpad=10)
        ax.set_title('Mosel Valley 3D Terrain with UGV Path', fontsize=14, fontweight='bold', pad=20)

        # Set viewing angle - rotated 60 degrees from side view for better perspective
        ax.view_init(elev=30, azim=120)

        # Add legend
        ax.legend(loc='upper left', fontsize=9)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if save_as:
            fig.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"     ✓ Saved: {save_as}")

        plt.close(fig)
        return fig

    def animate_3d_path(self, waypoints: List[Waypoint],
                       segments: List[GridSegment],
                       start_pos: Tuple[float, float],
                       elevation_model: ElevationModel,
                       save_as: Optional[str] = None,
                       sequenced_waypoints: Optional[List[Waypoint]] = None):
        """
        Create 3D animation of UGV traveling on terrain

        Parameters:
        -----------
        waypoints: List of waypoints
        segments: Path segments
        start_pos: Starting position
        elevation_model: Elevation model
        save_as: Optional filename to save (e.g., 'ugv_3d_animation.gif')
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create waypoint visit order mapping
        waypoint_visit_order = {}
        if sequenced_waypoints is not None:
            for visit_order, wp in enumerate(sequenced_waypoints, start=1):
                waypoint_visit_order[(wp.x, wp.y, wp.row_id)] = visit_order

        # Plot static terrain
        surf = ax.plot_surface(elevation_model.X, elevation_model.Y,
                              elevation_model.elevation,
                              cmap='terrain', alpha=0.5,
                              linewidth=0, antialiased=True)

        # Plot vine trees on terrain
        for row_id, trees in self.grid.tree_positions.items():
            tree_x = np.array([t[0] for t in trees])
            tree_y = np.array([t[1] for t in trees])
            tree_z = elevation_model.get_elevations_vectorized(tree_x, tree_y)
            ax.scatter(tree_x, tree_y, tree_z, c='darkgreen', s=8,
                      marker='o', alpha=0.3, zorder=2)

        # Plot waypoints (larger and more visible) with numbers
        for i, wp in enumerate(waypoints):
            ax.scatter(wp.x, wp.y, wp.z, c='red', s=80, marker='o',
                      edgecolors='darkred', linewidth=1.5, zorder=5)
            # Use visit order if available, otherwise use original index
            if waypoint_visit_order and (wp.x, wp.y, wp.row_id) in waypoint_visit_order:
                label_num = waypoint_visit_order[(wp.x, wp.y, wp.row_id)]
            else:
                label_num = i + 1
            # Add waypoint number label
            ax.text(wp.x, wp.y, wp.z + 2, f'{label_num}', fontsize=9,
                   color='white', weight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red',
                           edgecolor='darkred', alpha=0.8), zorder=6)

        # Plot path with intermediate waypoints (optimized)
        for seg in segments:
            x_path, y_path, _ = seg.interpolate()

            # Always query actual elevation from the model for accurate terrain following
            z_path = elevation_model.get_elevations_vectorized(x_path, y_path)

            z_path = z_path + 0.2

            color = SEGMENT_COLORS.get(seg.type, 'black')
            ax.plot(x_path, y_path, z_path, color=color, linewidth=2,
                   alpha=0.4, zorder=3)

            # Plot intermediate waypoints along the path
            ax.scatter(x_path, y_path, z_path, c=color, s=5,
                      marker='o', alpha=0.3, zorder=3)

        # Build interpolated path for animation (optimized)
        path_points_3d = []
        for seg in segments:
            # Use fewer points for faster animation
            num_points = max(3, int(seg.distance() / 3))  # Reduced from 15
            x_path, y_path, _ = seg.interpolate(num_points=num_points)

            # Always query actual elevation from the model for accurate terrain following
            z_path = elevation_model.get_elevations_vectorized(x_path, y_path)

            z_path = z_path + 0.3  # Slightly above terrain for ugv center point of gravity

            # Append all points from this segment
            path_points_3d.extend(zip(x_path, y_path, z_path))

        # Initialize UGV marker and trail
        ugv, = ax.plot([], [], [], 'go', markersize=12, markeredgecolor='darkgreen',
                      markeredgewidth=2, zorder=10, label='UGV')
        trail, = ax.plot([], [], [], 'g-', linewidth=2, alpha=0.7, zorder=9)

        # Set labels
        ax.set_xlabel('X (m) - Upslope Direction', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Elevation (m ASL)', fontsize=10)
        ax.set_title('Mosel Valley 3D UGV Path Animation', fontsize=12, fontweight='bold')

        # Set initial view (60 degree rotation for better perspective)
        ax.view_init(elev=25, azim=60)

        trail_x, trail_y, trail_z = [], [], []

        def animate(frame):
            if frame < len(path_points_3d):
                x, y, z = path_points_3d[frame]
                trail_x.append(x)
                trail_y.append(y)
                trail_z.append(z)

                ugv.set_data([x], [y])
                ugv.set_3d_properties([z])

                trail.set_data(trail_x, trail_y)
                trail.set_3d_properties(trail_z)

                # Slowly rotate view
                azim = 60 + (frame / len(path_points_3d)) * 90  # Rotate 90 degrees during animation
                ax.view_init(elev=25, azim=azim)

            return ugv, trail

        # Create animation (optimized with faster frame rate)
        anim = FuncAnimation(fig, animate, frames=len(path_points_3d),
                           interval=30, blit=False, repeat=True)

        # Save if requested
        if save_as:
            print(f"Saving 3D animation to {save_as}...")
            # Faster frame rate and reduced DPI for faster saving and less memory
            writer = PillowWriter(fps=20)
            anim.save(save_as, writer=writer, dpi=60)
            print(f"3D animation saved!")
            gc.collect()  # Clean up after saving

        plt.tight_layout()
        return fig, anim

    def plot_strategy_comparison(self, strategies: dict, save_as: Optional[str] = 'strategy_comparison.png'):
        """
        Create comprehensive comparison visualization of different sequencing strategies.

        Parameters:
        -----------
        strategies: Dict mapping strategy name to metrics dict
        save_as: Optional filename to save figure
        """
        fig = plt.figure(figsize=(20, 12))

        # Create 2x3 grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Energy comparison bar chart
        ax2 = fig.add_subplot(gs[1, 0])  # Time comparison
        ax3 = fig.add_subplot(gs[1, 1])  # Distance comparison
        ax4 = fig.add_subplot(gs[1, 2])  # Energy breakdown pie
        ax5 = fig.add_subplot(gs[2, :])  # Performance heatmap

        strategy_names = list(strategies.keys())
        n_strategies = len(strategy_names)

        # 1. Energy Consumption Comparison (stacked bar)
        energy_components = ['climbing_energy_j', 'rolling_energy_j', 'turn_energy_j']
        component_labels = ['Climbing', 'Rolling', 'Turning']
        colors = ['#ff6b6b', '#4ecdc4', '#ffa07a']

        x_pos = np.arange(n_strategies)
        bottom = np.zeros(n_strategies)

        for i, (component, label) in enumerate(zip(energy_components, component_labels)):
            values = [strategies[name][component] / 1000 for name in strategy_names]  # Convert to kJ
            ax1.bar(x_pos, values, label=label, bottom=bottom, color=colors[i], alpha=0.8)
            bottom += values

        ax1.set_xlabel('Strategy', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Energy (kJ)', fontsize=11, fontweight='bold')
        ax1.set_title('Energy Consumption Breakdown by Strategy', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(strategy_names, rotation=15, ha='right', fontsize=9)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Time Comparison
        times = [strategies[name]['total_time'] / 60 for name in strategy_names]  # Convert to minutes
        bars2 = ax2.bar(x_pos, times, color='#95e1d3', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Strategy', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Time (min)', fontsize=10, fontweight='bold')
        ax2.set_title('Total Time Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(strategy_names, rotation=15, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        # 3. Distance Comparison
        distances = [strategies[name]['total_distance_3d'] for name in strategy_names]
        bars3 = ax3.bar(x_pos, distances, color='#f38181', alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Strategy', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Distance (m)', fontsize=10, fontweight='bold')
        ax3.set_title('Total Distance Comparison (3D)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(strategy_names, rotation=15, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)

        # 4. Energy Breakdown Pie (for best strategy)
        best_strategy = min(strategies.items(), key=lambda x: x[1]['total_energy_kj'])
        best_name, best_metrics = best_strategy

        pie_values = [
            best_metrics['climbing_energy_j'] / 1000,
            best_metrics['rolling_energy_j'] / 1000,
            best_metrics['turn_energy_j'] / 1000
        ]

        _, _, autotexts = ax4.pie(pie_values, labels=component_labels, autopct='%1.1f%%',
                                   colors=colors, startangle=90)
        ax4.set_title(f'Energy Breakdown\n{best_name}\n(Lowest Energy)',
                     fontsize=12, fontweight='bold')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 5. Performance Heatmap
        metrics_for_heatmap = ['total_energy_kj', 'total_time', 'total_distance_3d',
                              'num_turns', 'total_elevation_gain']
        metric_labels = ['Energy\n(kJ)', 'Time\n(s)', 'Distance\n(m)', 'Turns', 'Elev Gain\n(m)']

        # Create normalized heatmap data
        heatmap_data = np.zeros((len(metrics_for_heatmap), n_strategies))

        for i, metric in enumerate(metrics_for_heatmap):
            values = [strategies[name][metric] for name in strategy_names]
            # Normalize to 0-1 range (lower is better, so invert)
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                normalized = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                normalized = [1.0] * len(values)
            heatmap_data[i, :] = normalized

        im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax5.set_xticks(np.arange(n_strategies))
        ax5.set_yticks(np.arange(len(metric_labels)))
        ax5.set_xticklabels(strategy_names, rotation=15, ha='right', fontsize=9)
        ax5.set_yticklabels(metric_labels, fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Performance\n(Green=Better)', rotation=270, labelpad=20, fontsize=10)

        # Add text annotations
        for i in range(len(metric_labels)):
            for j in range(n_strategies):
                metric = metrics_for_heatmap[i]
                value = strategies[strategy_names[j]][metric]
                if metric == 'total_time':
                    text = f'{value:.0f}s'
                elif metric == 'total_energy_kj':
                    text = f'{value:.1f}'
                else:
                    text = f'{value:.0f}'
                ax5.text(j, i, text, ha='center', va='center',
                        color='black' if heatmap_data[i, j] > 0.5 else 'white',
                        fontsize=8, fontweight='bold')

        ax5.set_title('Performance Heatmap (Green = Better Performance)',
                     fontsize=12, fontweight='bold', pad=10)

        fig.suptitle('Vineyard Routing Strategy Comparison', fontsize=16, fontweight='bold', y=0.995)

        if save_as:
            fig.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"     ✓ Saved: {save_as}")

        plt.close(fig)
        return fig

    def plot_elevation_profiles(self, strategies_data: dict, elevation_model: ElevationModel,
                               save_as: Optional[str] = 'elevation_profiles.png'):
        """
        Plot elevation profiles for different strategies.

        Parameters:
        -----------
        strategies_data: Dict mapping strategy name to (segments, metrics) tuple
        elevation_model: Elevation model
        save_as: Optional filename to save figure
        """
        fig, axes = plt.subplots(len(strategies_data), 1, figsize=(16, 4*len(strategies_data)))

        if len(strategies_data) == 1:
            axes = [axes]

        for idx, (strategy_name, (segments, metrics)) in enumerate(strategies_data.items()):
            ax = axes[idx]

            # Calculate elevation profile along path
            cumulative_distance = 0
            distances = [0]
            elevations = [segments[0].start_z if hasattr(segments[0], 'start_z')
                         else elevation_model.get_elevation(*segments[0].start)]

            for seg in segments:
                # Interpolate along segment
                x_path, y_path, _ = seg.interpolate()
                z_path = elevation_model.get_elevations_vectorized(x_path, y_path)

                seg_distances = np.linspace(cumulative_distance,
                                           cumulative_distance + seg.distance(),
                                           len(x_path))

                distances.extend(seg_distances[1:])
                elevations.extend(z_path[1:])
                cumulative_distance += seg.distance()

            # Plot elevation profile with base elevation fill
            # Get base elevation from the terrain minimum
            if elevation_model:
                base_elev = float(np.min(elevation_model.elevation))
            else:
                base_elev = 150.0  # Default Mosel Valley base elevation

            ax.fill_between(distances, base_elev, elevations, alpha=0.3, color='brown')
            ax.plot(distances, elevations, 'b-', linewidth=2, label='Path elevation')

            # Add climb/descent indicators
            climbs = []
            descents = []
            for i in range(1, len(elevations)):
                if elevations[i] > elevations[i-1]:
                    climbs.append(i)
                elif elevations[i] < elevations[i-1]:
                    descents.append(i)

            if climbs:
                ax.scatter([distances[i] for i in climbs],
                          [elevations[i] for i in climbs],
                          c='red', s=1, alpha=0.3, label='Climbing')
            if descents:
                ax.scatter([distances[i] for i in descents],
                          [elevations[i] for i in descents],
                          c='green', s=1, alpha=0.3, label='Descending')

            # Add metrics text
            text = (f"Energy: {metrics['total_energy_kj']:.2f} kJ  |  "
                   f"Climb: {metrics['total_elevation_gain']:.1f} m  |  "
                   f"Descent: {metrics['total_elevation_loss']:.1f} m")

            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Cumulative Distance (m)', fontsize=11)
            ax.set_ylabel('Elevation (m)', fontsize=11)
            ax.set_title(f'{strategy_name} - Elevation Profile', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            # Set y-axis to start from base elevation
            ax.set_ylim(bottom=base_elev)

        plt.tight_layout()

        if save_as:
            fig.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"     ✓ Saved: {save_as}")

        plt.close(fig)
        return fig

    def plot_energy_over_path(self, segments: List[GridSegment],
                             planner: 'GridConstrainedPlanner',
                             save_as: Optional[str] = 'energy_over_path.png'):
        """
        Plot cumulative energy consumption and elevation profile over the path.

        Parameters:
        -----------
        segments: Path segments
        planner: Planner with UGV parameters
        save_as: Optional filename to save figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

        # Calculate energy consumption and elevation along path
        cumulative_distance = []
        cumulative_energy = []
        energy_rate = []
        elevations = []

        dist = 0
        energy = 0

        # Add starting elevation
        first_seg = segments[0] if segments else None
        if first_seg and first_seg.elevation_start is not None:
            elevations.append(first_seg.elevation_start)
        else:
            elevations.append(0)
        cumulative_distance.append(0)
        cumulative_energy.append(0)
        energy_rate.append(0)

        # Store segment boundaries for step plot
        segment_boundaries = [0]  # Start position
        segment_rates = []  # Energy rate for each segment (constant per segment)

        for seg in segments:
            seg_dist = seg.distance()
            energy_data = planner.calculate_energy_for_segment(seg)
            seg_energy = energy_data['total_energy']

            dist += seg_dist
            energy += seg_energy

            cumulative_distance.append(dist)
            cumulative_energy.append(energy / 1000)  # Convert to kJ

            # Calculate constant energy rate for this segment
            seg_rate = seg_energy / seg_dist if seg_dist > 0 else 0
            segment_rates.append(seg_rate)
            segment_boundaries.append(dist)

            # Track elevation at end of segment
            if seg.elevation_end is not None:
                elevations.append(seg.elevation_end)
            else:
                elevations.append(elevations[-1])  # Keep previous elevation

        # Plot 1: Cumulative Energy Consumption
        ax1.plot(cumulative_distance, cumulative_energy, 'b-', linewidth=2, label='Total Energy')
        ax1.fill_between(cumulative_distance, cumulative_energy, alpha=0.3, color='blue')
        ax1.set_xlabel('Distance (m)', fontsize=11)
        ax1.set_ylabel('Cumulative Energy (kJ)', fontsize=11)
        ax1.set_title('Cumulative Energy Consumption Along Path', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)

        # Add final energy annotation
        ax1.annotate(f'Total: {cumulative_energy[-1]:.2f} kJ',
                    xy=(cumulative_distance[-1], cumulative_energy[-1]),
                    xytext=(-60, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, fontweight='bold')

        # Plot 2: Energy Rate (J/m) - Step plot with constant rate per segment
        for i in range(len(segment_rates)):
            ax2.hlines(segment_rates[i], segment_boundaries[i], segment_boundaries[i+1],
                      colors='r', linewidth=2, label='Energy per meter' if i == 0 else '')
            # Add vertical lines to show segment transitions
            if i < len(segment_rates) - 1:
                ax2.vlines(segment_boundaries[i+1], segment_rates[i], segment_rates[i+1],
                          colors='r', linewidth=2, linestyles='--', alpha=0.3)

        ax2.axhline(y=np.mean(segment_rates), color='g', linestyle='--',
                   label=f'Average: {np.mean(segment_rates):.1f} J/m', linewidth=2)
        ax2.set_xlabel('Distance (m)', fontsize=11)
        ax2.set_ylabel('Energy Rate (J/m)', fontsize=11)
        ax2.set_title('Energy Consumption Rate (Constant per Segment)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)

        # Plot 3: Elevation Profile
        # Get base elevation for fill (use minimum elevation as base)
        # For Mosel Valley, base is 150m; for gentle terrain, base is 100m
        if hasattr(planner, 'elevation_model'):
            # Use the minimum elevation from the terrain as base
            base_elev = float(np.min(planner.elevation_model.elevation))
        else:
            base_elev = 150.0  # Default Mosel Valley base elevation

        ax3.plot(cumulative_distance, elevations, 'brown', linewidth=2, label='Elevation')
        ax3.fill_between(cumulative_distance, base_elev, elevations, alpha=0.3, color='brown')

        # Mark climbing and descending sections
        for i in range(1, len(elevations)):
            if elevations[i] > elevations[i-1]:
                ax3.plot([cumulative_distance[i-1], cumulative_distance[i]],
                        [elevations[i-1], elevations[i]],
                        'r-', linewidth=3, alpha=0.6, label='Climbing' if i == 1 else '')
            elif elevations[i] < elevations[i-1]:
                ax3.plot([cumulative_distance[i-1], cumulative_distance[i]],
                        [elevations[i-1], elevations[i]],
                        'g-', linewidth=3, alpha=0.6, label='Descending' if i == 1 else '')

        ax3.set_xlabel('Distance (m)', fontsize=11)
        ax3.set_ylabel('Elevation (m)', fontsize=11)
        ax3.set_title('Elevation Profile Along Path', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)

        # Set y-axis to start from base elevation
        ax3.set_ylim(bottom=base_elev)

        # Add elevation change annotation
        total_gain = sum(max(0, elevations[i] - elevations[i-1]) for i in range(1, len(elevations)))
        total_loss = sum(max(0, elevations[i-1] - elevations[i]) for i in range(1, len(elevations)))
        ax3.text(0.02, 0.98, f'Gain: {total_gain:.1f}m\nLoss: {total_loss:.1f}m',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_as:
            fig.savefig(save_as, dpi=150, bbox_inches='tight')
            print(f"     ✓ Saved: {save_as}")

        plt.close(fig)
        return fig

    def create_algorithm_comparison_animations(self,
                                               waypoints: List[Waypoint],
                                               strategies_data: dict,
                                               start_pos: Tuple[float, float],
                                               elevation_model: Optional[ElevationModel] = None,
                                               planner: Optional['GridConstrainedPlanner'] = None):
        """
        Create individual 2D animations for each algorithm for side-by-side comparison.

        Parameters:
        -----------
        waypoints: List of waypoints
        strategies_data: Dict mapping strategy name to (segments, metrics) tuple
        start_pos: Starting position
        elevation_model: Optional elevation model
        planner: Optional planner instance

        Creates separate animation files:
        - algorithm_nearest_neighbor.gif
        - algorithm_elevation.gif
        - algorithm_simulated_annealing.gif
        etc.
        """
        print("\n   Creating individual algorithm animations...")

        for strategy_name, (segments, _) in strategies_data.items():
            # Clean filename
            safe_name = strategy_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            filename = f'algorithm_{safe_name}.gif'

            print(f"     - {strategy_name}...")

            try:
                # Create animation
                fig, _ = self.animate_ugv_path(
                    waypoints,
                    segments,
                    start_pos,
                    elevation_model=elevation_model,
                    planner=planner,
                    save_as=filename
                )
                plt.close(fig)
                print(f"       ✓ Saved: {filename}")
            except Exception as e:
                print(f"       ✗ Failed: {e}")

def generate_random_waypoints(grid: VineyardGrid, num_waypoints: int = 10, elevation_model: Optional[ElevationModel] = None) -> List[Waypoint]:
    """
    Generate random waypoints for testing with elevation.
    Waypoints are positioned in alleys (between rows) near tree positions.
    """
    waypoints = []

    # We have num_rows-1 alleys (spaces between rows)
    num_alleys = grid.num_rows - 1

    for i in range(num_waypoints):
        # Select a random alley (avoid edges)
        alley_id = random.randint(1, num_alleys - 2)

        # Select a random tree x-position (same for both adjacent rows)
        row_for_trees = alley_id  # Use lower row for tree reference
        trees_in_row = grid.tree_positions[row_for_trees]
        if len(trees_in_row) > 2:
            # Avoid first and last tree
            tree_pos = random.choice(trees_in_row[1:-1])
        else:
            tree_pos = random.choice(trees_in_row)

        x = tree_pos[0]

        # Position in the alley - offset from center to be closer to one row or the other
        alley_center_y = grid.get_alley_center_y(alley_id)
        # Small offset (0.3 to 0.8 m from center, toward one of the rows)
        offset = random.uniform(0.3, 0.8) * random.choice([-1, 1])
        y = alley_center_y + offset

        # Get elevation if model is provided
        z = elevation_model.get_elevation(x, y) if elevation_model else 0.0

        waypoints.append(Waypoint(x=x, y=y, row_id=alley_id, offset=abs(offset), z=z))

    return waypoints

def run_statistical_analysis(grid, elevation_model, planner, num_trials=10, num_waypoints=15,
                            run_rl=False, start_pos=(5.0, 1.5)):
    """
    Run multiple trials and collect statistical performance data for each strategy.

    Returns comprehensive statistics including mean, std, min, max, median, and percentiles.
    """
    import statistics
    from collections import defaultdict

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS - MULTIPLE TRIALS")
    print("="*70)
    print(f"Running {num_trials} trials with {num_waypoints} waypoints each...")
    print()

    # Store results for each strategy
    results = defaultdict(lambda: {
        'energy': [], 'time': [], 'distance': [],
        'climbing_energy': [], 'rolling_energy': [], 'turn_energy': []
    })

    strategies_to_test = [
        ('Nearest Neighbor', 'nearest_neighbor', {}),
        ('Elevation Greedy', 'elevation', {}),
        ('SA (Energy)', 'simulated_annealing', {'objective': 'energy', 'max_iterations': 2000}),
        ('SA (Distance)', 'simulated_annealing', {'objective': 'distance', 'max_iterations': 2000}),
    ]

    if run_rl:
        strategies_to_test.append(('RL (PPO)', 'rl', {'objective': 'energy', 'timesteps': 50000}))

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        print("-" * 70)

        # Generate random waypoints for this trial
        waypoints = []
        random.seed(trial * 42)  # Reproducible random waypoints
        for i in range(num_waypoints):
            row_id = random.randint(0, grid.num_rows - 2)
            x = random.uniform(5, grid.row_length - 5)
            alley_y = grid.get_alley_center_y(row_id)
            offset_y = random.uniform(-0.3, 0.3)
            y = alley_y + offset_y
            z = elevation_model.get_elevation(x, y)
            waypoints.append(Waypoint(x, y, z, row_id))

        # Test each strategy
        for strategy_name, mode, kwargs in strategies_to_test:
            try:
                print(f"   Testing {strategy_name}...", end=" ")
                sys.stdout.flush()

                segments, metrics, sequence = planner.plan_complete_tour(
                    waypoints, start_pos, sequencing_mode=mode, **kwargs
                )

                # Store results
                results[strategy_name]['energy'].append(metrics['total_energy_kj'])
                results[strategy_name]['time'].append(metrics['total_time'] / 60.0)  # minutes
                results[strategy_name]['distance'].append(metrics['total_distance_3d'])
                results[strategy_name]['climbing_energy'].append(metrics['climbing_energy_j'] / 1000.0)
                results[strategy_name]['rolling_energy'].append(metrics['rolling_energy_j'] / 1000.0)
                results[strategy_name]['turn_energy'].append(metrics['turn_energy_j'] / 1000.0)

                print(f"✓ (Energy: {metrics['total_energy_kj']:.2f} kJ)")

                gc.collect()  # Clean up memory

            except Exception as e:
                print(f"✗ Failed: {e}")
                # Use NaN for failed trials
                for metric in ['energy', 'time', 'distance', 'climbing_energy', 'rolling_energy', 'turn_energy']:
                    results[strategy_name][metric].append(float('nan'))

    # Calculate statistics
    print("\n" + "="*70)
    print("STATISTICAL RESULTS")
    print("="*70)

    def calc_stats(data):
        """Calculate statistics, filtering out NaN values"""
        valid_data = [x for x in data if not np.isnan(x)]
        if not valid_data:
            return {
                'mean': float('nan'), 'std': float('nan'), 'min': float('nan'),
                'max': float('nan'), 'median': float('nan'),
                'q25': float('nan'), 'q75': float('nan'),
                'success_rate': 0.0
            }
        return {
            'mean': statistics.mean(valid_data),
            'std': statistics.stdev(valid_data) if len(valid_data) > 1 else 0.0,
            'min': min(valid_data),
            'max': max(valid_data),
            'median': statistics.median(valid_data),
            'q25': np.percentile(valid_data, 25),
            'q75': np.percentile(valid_data, 75),
            'success_rate': len(valid_data) / len(data) * 100.0
        }

    # Print Energy Statistics
    print("\n1. ENERGY CONSUMPTION (kJ)")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Mean':>10} {'±Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 70)

    for strategy_name in results.keys():
        stats = calc_stats(results[strategy_name]['energy'])
        print(f"{strategy_name:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{stats['min']:>10.2f} {stats['max']:>10.2f} {stats['median']:>10.2f}")

    # Print Time Statistics
    print("\n2. TRAVEL TIME (minutes)")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Mean':>10} {'±Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 70)

    for strategy_name in results.keys():
        stats = calc_stats(results[strategy_name]['time'])
        print(f"{strategy_name:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{stats['min']:>10.2f} {stats['max']:>10.2f} {stats['median']:>10.2f}")

    # Print Distance Statistics
    print("\n3. TRAVEL DISTANCE (m)")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Mean':>10} {'±Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 70)

    for strategy_name in results.keys():
        stats = calc_stats(results[strategy_name]['distance'])
        print(f"{strategy_name:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{stats['min']:>10.2f} {stats['max']:>10.2f} {stats['median']:>10.2f}")

    # Energy Component Breakdown
    print("\n4. ENERGY COMPONENT BREAKDOWN (kJ) - Mean ± Std")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Climbing':>18} {'Rolling':>18} {'Turning':>18}")
    print("-" * 70)

    for strategy_name in results.keys():
        climb_stats = calc_stats(results[strategy_name]['climbing_energy'])
        roll_stats = calc_stats(results[strategy_name]['rolling_energy'])
        turn_stats = calc_stats(results[strategy_name]['turn_energy'])
        print(f"{strategy_name:<25} {climb_stats['mean']:>8.2f}±{climb_stats['std']:>7.2f} "
              f"{roll_stats['mean']:>8.2f}±{roll_stats['std']:>7.2f} "
              f"{turn_stats['mean']:>8.2f}±{turn_stats['std']:>7.2f}")

    # Success Rate
    print("\n5. SUCCESS RATE (%)")
    print("-" * 70)
    for strategy_name in results.keys():
        stats = calc_stats(results[strategy_name]['energy'])
        print(f"{strategy_name:<25} {stats['success_rate']:>10.1f}%")

    # Statistical Comparison (Relative Performance)
    print("\n6. RELATIVE PERFORMANCE (compared to Nearest Neighbor baseline)")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Energy':>15} {'Time':>15} {'Distance':>15}")
    print("-" * 70)

    nn_energy_mean = statistics.mean(results['Nearest Neighbor']['energy'])
    nn_time_mean = statistics.mean(results['Nearest Neighbor']['time'])
    nn_dist_mean = statistics.mean(results['Nearest Neighbor']['distance'])

    for strategy_name in results.keys():
        if strategy_name == 'Nearest Neighbor':
            print(f"{strategy_name:<25} {'0.0% (baseline)':>15} {'0.0% (baseline)':>15} {'0.0% (baseline)':>15}")
        else:
            energy_mean = statistics.mean(results[strategy_name]['energy'])
            time_mean = statistics.mean(results[strategy_name]['time'])
            dist_mean = statistics.mean(results[strategy_name]['distance'])

            energy_diff = ((energy_mean - nn_energy_mean) / nn_energy_mean) * 100
            time_diff = ((time_mean - nn_time_mean) / nn_time_mean) * 100
            dist_diff = ((dist_mean - nn_dist_mean) / nn_dist_mean) * 100

            print(f"{strategy_name:<25} {energy_diff:>+14.2f}% {time_diff:>+14.2f}% {dist_diff:>+14.2f}%")

    # Coefficient of Variation (stability metric)
    print("\n7. COEFFICIENT OF VARIATION (lower = more stable)")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Energy':>15} {'Time':>15} {'Distance':>15}")
    print("-" * 70)

    for strategy_name in results.keys():
        energy_stats = calc_stats(results[strategy_name]['energy'])
        time_stats = calc_stats(results[strategy_name]['time'])
        dist_stats = calc_stats(results[strategy_name]['distance'])

        cv_energy = (energy_stats['std'] / energy_stats['mean']) * 100 if energy_stats['mean'] > 0 else 0
        cv_time = (time_stats['std'] / time_stats['mean']) * 100 if time_stats['mean'] > 0 else 0
        cv_dist = (dist_stats['std'] / dist_stats['mean']) * 100 if dist_stats['mean'] > 0 else 0

        print(f"{strategy_name:<25} {cv_energy:>14.2f}% {cv_time:>14.2f}% {cv_dist:>14.2f}%")

    print("\n" + "="*70)

    return results


def main():
    """Main function to demonstrate the vineyard routing solution with elevation, energy, and animation"""

    # ========================================================================
    # USER CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
    # ========================================================================

    # Vineyard Grid Parameters
    NUM_ROWS = 20              # Number of vine rows
    ROW_SPACING = 3.0          # Distance between rows (meters)
    ROW_LENGTH = 100.0         # Length of each row (meters)
    TREE_SPACING = 1.5         # Distance between vine trees along row (meters)

    # Terrain Parameters
    TERRAIN_TYPE = 'mosel'     # 'mosel' for hill-shaped or
    TERRAIN_RESOLUTION = 0.5   # Grid resolution for elevation sampling (meters)
    BASE_ELEVATION = 150.0     # Base elevation in meters (configured in ElevationModel)
    MAX_ELEVATION_GAIN = 30.0  # Maximum elevation gain along row (configured in ElevationModel)

    # Waypoint Generation
    NUM_WAYPOINTS = 15         # Number of random waypoints to generate

    # UGV Parameters (defined in GridConstrainedPlanner, listed here for reference)
    # UGV_MASS = 150.0         # kg (typical agricultural UGV)
    # UGV_VELOCITY = 1.5       # m/s (average travel velocity)
    # ROLLING_RESISTANCE = 0.15  # coefficient for rough terrain
    # MECHANICAL_EFFICIENCY = 0.75  # motor/drivetrain efficiency (0-1)
 
    # TURN_TIME = 6.0          # seconds per 90° turn
    RUN_RL_OPTIMIZER = False  # Set to True to enable PPO training

    # Statistical Analysis
    RUN_STATISTICAL_ANALYSIS = True  # Run multiple trials for statistical analysis
    NUM_TRIALS = 10           # Number of trials to run for statistical analysis
    NUM_WAYPOINTS_PER_TRIAL = 15  # Number of waypoints per trial

    # Sequencing Strategy
    COMPARE_STRATEGIES = True  # Compare both sequencing strategies
    DEFAULT_STRATEGY = 'elevation'  # 'nearest_neighbor' or 'elevation'

    # Visualization Parameters
    SAVE_2D_PLOT = True        # Save 2D static visualization
    SAVE_3D_PLOT = True        # Save 3D terrain visualization
    SAVE_2D_ANIMATION = False   # Save 2D animation (GIF)
    SAVE_3D_ANIMATION = False   # Save 3D animation (GIF)
    SAVE_ALGORITHM_ANIMATIONS = False  # Save individual algorithm animations (GIF)

    # Output File Names
    OUTPUT_2D_PLOT = 'vineyard_solution.png'
    OUTPUT_3D_PLOT = 'vineyard_3d_terrain.png'
    OUTPUT_2D_ANIMATION = 'ugv_animation_2d.gif'
    OUTPUT_3D_ANIMATION = 'ugv_animation_3d.gif'

    # ========================================================================
    # END OF USER CONFIGURATION
    # ========================================================================

    print("=" * 70)
    print("MOSEL VALLEY VINEYARD UGV ROUTING")
    print("Hill-Shaped Terrain Navigation with Energy Analysis")
    print("=" * 70)

    # Create vineyard grid with vine trees
    grid = VineyardGrid(num_rows=NUM_ROWS, row_spacing=ROW_SPACING,
                       row_length=ROW_LENGTH, tree_spacing=TREE_SPACING)
    print(f"   Grid: {grid.num_rows} rows, {grid.row_length}m length")
    print(f"   Vine trees: {grid.tree_spacing}m spacing, {len(grid.tree_positions[0])} trees per row")

    # Create Mosel Valley elevation model
    print("\n1. Creating terrain model...")
    elevation_model = ElevationModel(
        grid_bounds=(0, grid.row_length, 0, grid.get_row_y(grid.num_rows-1)),
        resolution=TERRAIN_RESOLUTION,
        terrain_type=TERRAIN_TYPE
    )

    print("   Terrain type: Mosel Valley (hill-shaped profile)")
    print("   Base elevation: 150m (river level)")
    print("   Elevation gain: ~30m (along row direction)")
    print("   Profile: Non-linear hill with random variations")

    # Create planner with elevation
    planner = GridConstrainedPlanner(grid, elevation_model)

    # Generate random waypoints with elevation (in alleys between rows)
    print("2. Generating waypoints in alleys between rows...")
    waypoints = generate_random_waypoints(grid, num_waypoints=NUM_WAYPOINTS, elevation_model=elevation_model)
    print(f"   Generated {len(waypoints)} waypoints in alleys (between vine rows)")

    # Starting position (in an alley between rows)
    # With 20 rows and 3m spacing, alley 9 is between row 9 and 10 (middle of field)
    start_alley = 9
    start_pos = (0, grid.get_alley_center_y(start_alley))  # Start in middle alley

    # Run statistical analysis if enabled
    if RUN_STATISTICAL_ANALYSIS:
        statistical_results = run_statistical_analysis(
            grid=grid,
            elevation_model=elevation_model,
            planner=planner,
            num_trials=NUM_TRIALS,
            num_waypoints=NUM_WAYPOINTS_PER_TRIAL,
            run_rl=RUN_RL_OPTIMIZER,
            start_pos=start_pos
        )
        print("\n✓ Statistical analysis completed!")
        print("\nNote: Skipping single-run comparison since statistical analysis was performed.")
        print("="*70)
        return  # Exit after statistical analysis

    # Plan tours with sequencing strategies
    print("3. Planning grid-constrained tours...")

    if COMPARE_STRATEGIES:
        print("\n   3a. Nearest Neighbor sequencing (baseline)...")
        segments_nn, metrics_nn, seq_nn = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='nearest_neighbor')

        print("   3b. Elevation-based greedy sequencing...")
        segments_elev, metrics_elev, seq_elev = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='elevation')

        print("   3c. Simulated Annealing (energy optimization with 2000 iterations)...")
        sys.stdout.flush()
        # Using reduced iterations with periodic garbage collection to prevent segfault
        segments_sa_energy, metrics_sa_energy, seq_sa_energy = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='simulated_annealing', objective='energy', max_iterations=1500)
        print("      ✓ SA (energy) completed")
        gc.collect()  # Clean up after SA
        import time
        time.sleep(0.5)  # Brief pause to allow memory cleanup
        gc.collect()  # Second cleanup

        print("   3d. Simulated Annealing (distance optimization with 2000 iterations)...")
        sys.stdout.flush()
        segments_sa_dist, metrics_sa_dist, seq_sa_dist = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='simulated_annealing', objective='distance', max_iterations=1500)
        print("      ✓ SA (distance) completed")
        gc.collect()  # Clean up after SA

        # RL Optimizer (optional - can cause memory issues)
        if RUN_RL_OPTIMIZER:
            print("   3e. RL Optimizer (PPO algorithm)...")
            sys.stdout.flush()
            try:
                segments_rl, metrics_rl, seq_rl = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='rl', objective='energy', timesteps=50000)
                print("      ✓ PPO completed")
                gc.collect()  # Clean up after PPO
            except Exception as e:
                print(f"      ⚠ PPO training failed: {e}")
                print(f"      Using SA energy result as fallback...")
                segments_rl, metrics_rl, seq_rl = segments_sa_energy, metrics_sa_energy, seq_sa_energy
        else:
            print("   3e. RL Optimizer (DISABLED - set RUN_RL_OPTIMIZER=True to enable)")

        # Compare sequencing strategies
        num_strategies = 5 if RUN_RL_OPTIMIZER else 4
        print("\n" + "="*70)
        print(f"SEQUENCING STRATEGY COMPARISON ({num_strategies} STRATEGIES)")
        print("="*70)

        # Create comparison table
        strategies = {
            'Nearest Neighbor': metrics_nn,
            'Elevation Greedy': metrics_elev,
            'Simulated Annealing (Energy)': metrics_sa_energy,
            'Simulated Annealing (Distance)': metrics_sa_dist
        }

        if RUN_RL_OPTIMIZER:
            strategies['RL (PPO)'] = metrics_rl

        print("\n{:<30} {:>12} {:>12} {:>12}".format(
            "Strategy", "Energy (kJ)", "Time (min)", "Distance (m)"
        ))
        print("-" * 70)

        for name, m in strategies.items():
            print("{:<30} {:>12.2f} {:>12.1f} {:>12.1f}".format(
                name,
                m['total_energy_kj'],
                m['total_time'] / 60.0,
                m['total_distance_3d']
            ))

        # Find best strategies
        best_energy = min(strategies.items(), key=lambda x: x[1]['total_energy_kj'])
        best_time = min(strategies.items(), key=lambda x: x[1]['total_time'])
        best_distance = min(strategies.items(), key=lambda x: x[1]['total_distance_3d'])

        print("\n" + "="*70)
        print("BEST STRATEGIES:")
        print("="*70)
        print(f"Lowest Energy: {best_energy[0]} ({best_energy[1]['total_energy_kj']:.2f} kJ)")
        print(f"Fastest Time: {best_time[0]} ({best_time[1]['total_time']/60:.1f} min)")
        print(f"Shortest Distance: {best_distance[0]} ({best_distance[1]['total_distance_3d']:.1f} m)")

        # Calculate improvements over baseline
        print("\n" + "="*70)
        print("IMPROVEMENTS OVER BASELINE (Nearest Neighbor):")
        print("="*70)

        for name, m in strategies.items():
            if name == 'Nearest Neighbor (Baseline)':
                continue

            energy_save = ((metrics_nn['total_energy_kj'] - m['total_energy_kj']) / metrics_nn['total_energy_kj']) * 100
            time_diff = ((m['total_time'] - metrics_nn['total_time']) / metrics_nn['total_time']) * 100

            print(f"\n{name}:")
            print(f"  Energy: {energy_save:+.1f}%")
            print(f"  Time: {time_diff:+.1f}%")

        # Select segments/waypoints for best energy strategy
        best_energy_name = best_energy[0].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        if 'energy' in best_energy_name and 'simulated' in best_energy_name:
            segments_energy = segments_sa_energy
            metrics_energy = metrics_sa_energy
            seq_energy = seq_sa_energy
        elif 'distance' in best_energy_name and 'simulated' in best_energy_name:
            segments_energy = segments_sa_dist
            metrics_energy = metrics_sa_dist
            seq_energy = seq_sa_dist
        elif 'rl' in best_energy_name or 'ppo' in best_energy_name:
            segments_energy = segments_rl
            metrics_energy = metrics_rl
            seq_energy = seq_rl
        elif 'elevation' in best_energy_name:
            segments_energy = segments_elev
            metrics_energy = metrics_elev
            seq_energy = seq_elev
        else:
            segments_energy = segments_nn
            metrics_energy = metrics_nn
            seq_energy = seq_nn

        # Select segments/waypoints for best distance strategy
        best_distance_name = best_distance[0].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        if 'energy' in best_distance_name and 'simulated' in best_distance_name:
            segments_distance = segments_sa_energy
            metrics_distance = metrics_sa_energy
            seq_distance = seq_sa_energy
        elif 'distance' in best_distance_name and 'simulated' in best_distance_name:
            segments_distance = segments_sa_dist
            metrics_distance = metrics_sa_dist
            seq_distance = seq_sa_dist
        elif 'rl' in best_distance_name or 'ppo' in best_distance_name:
            segments_distance = segments_rl
            metrics_distance = metrics_rl
            seq_distance = seq_rl
        elif 'elevation' in best_distance_name:
            segments_distance = segments_elev
            metrics_distance = metrics_elev
            seq_distance = seq_elev
        else:
            segments_distance = segments_nn
            metrics_distance = metrics_nn
            seq_distance = seq_nn

        # Use best energy strategy as default visualization
        segments = segments_energy
        metrics = metrics_energy
        sequenced_waypoints = seq_energy
        better_strategy = best_energy_name
    else:
        # Use default strategy without comparison
        print(f"\n   Using {DEFAULT_STRATEGY} strategy...")
        segments, metrics, sequenced_waypoints = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode=DEFAULT_STRATEGY)
        better_strategy = DEFAULT_STRATEGY

    # Print metrics for selected strategy
    print("\n" + "="*70)
    print(f"TOUR METRICS - {better_strategy.upper().replace('_', ' ')} STRATEGY")
    print("="*70)
    print(f"Number of waypoints: {len(waypoints)}")
    print(f"Total path distance (2D): {metrics['total_distance_2d']:.1f} m")
    print(f"Total path distance (3D): {metrics['total_distance_3d']:.1f} m")
    print(f"Distance increase from elevation: {(metrics['total_distance_3d']/metrics['total_distance_2d']-1)*100:.1f}%")
    print(f"Number of turns: {metrics['num_turns']}")
    print(f"Travel time: {metrics['travel_time']:.1f} s")
    print(f"Turn time: {metrics['turn_time']:.1f} s")
    print(f"Total execution time: {metrics['total_time']:.1f} s ({metrics['total_time']/60:.1f} min)")
    print(f"Average effective speed: {metrics['total_distance']/metrics['total_time']:.2f} m/s")

    print(f"\n{'='*70}")
    print("ENERGY CONSUMPTION")
    print("="*70)
    print(f"Total Energy: {metrics['total_energy_kj']:.2f} kJ ({metrics['total_energy_kwh']*1000:.2f} Wh)")
    print(f"  - Climbing hills: {metrics['climbing_energy_j']/1000:.2f} kJ ({metrics['climbing_energy_j']/metrics['total_energy_j']*100:.1f}%)")
    print(f"  - Rolling resistance: {metrics['rolling_energy_j']/1000:.2f} kJ ({metrics['rolling_energy_j']/metrics['total_energy_j']*100:.1f}%)")
    print(f"  - Turning maneuvers: {metrics['turn_energy_j']/1000:.2f} kJ ({metrics['turn_energy_j']/metrics['total_energy_j']*100:.1f}%)")
    print(f"Energy per distance: {metrics['total_energy_kj']/metrics['total_distance']*1000:.2f} J/m")
    print(f"Average power: {metrics['total_energy_j']/metrics['total_time']:.2f} W")

    # Calculate elevation statistics
    elevations = [wp.z for wp in waypoints]
    print(f"\n{'='*70}")
    print("ELEVATION STATISTICS")
    print("="*70)
    print(f"Waypoint elevations:")
    print(f"  Min elevation: {min(elevations):.1f} m")
    print(f"  Max elevation: {max(elevations):.1f} m")
    print(f"  Elevation range: {max(elevations) - min(elevations):.1f} m")
    print(f"\nPath elevation changes:")
    print(f"  Total elevation gain: {metrics['total_elevation_gain']:.1f} m")
    print(f"  Total elevation loss: {metrics['total_elevation_loss']:.1f} m")
    print(f"  Net elevation change: {metrics['total_elevation_gain'] - metrics['total_elevation_loss']:.1f} m")

    # Calculate terrain slope statistics
    avg_grade = (max(elevations) - min(elevations)) / grid.row_length * 100
    print(f"\nTerrain characteristics:")
    print(f"  Average hillside grade: {avg_grade:.1f}%")
    print(f"  Terrain type: Mosel Valley (hill-shaped profile)")

    # Calculate theoretical minimum (Euclidean) using the selected strategy
    euclidean_dist = 0
    wp_sequence = planner.sequence_waypoints(waypoints, start_pos, mode=better_strategy)
    current = start_pos
    for wp in wp_sequence:
        euclidean_dist += np.sqrt((wp.x - current[0])**2 + (wp.y - current[1])**2)
        euclidean_dist += 2 * wp.offset  # Spur in and out from alley center
        current = (wp.x, grid.get_alley_center_y(wp.row_id))

    print(f"\nTheoretical Euclidean minimum ({better_strategy}): {euclidean_dist:.1f} m")
    print(f"Grid overhead ratio: {metrics['total_distance']/euclidean_dist:.2f}x")

    # Visualize
    print("\n4. Creating visualizations...")
    visualizer = VineyardVisualizer(grid)

    # 2D static visualization
    if SAVE_2D_PLOT:
        print("   - 2D path visualization...")
        try:
            fig_static = visualizer.plot_solution(waypoints, segments, metrics, start_pos,
                                                  save_as=OUTPUT_2D_PLOT)
        except Exception as e:
            print(f"     ✗ 2D visualization failed: {e}")

    # 3D terrain visualization
    if SAVE_3D_PLOT:
        print("   - 3D terrain visualization...")
        try:
            fig_3d = visualizer.plot_3d_terrain(
                waypoints,
                segments,
                start_pos,
                elevation_model,
                save_as=OUTPUT_3D_PLOT,
                sequenced_waypoints=sequenced_waypoints
            )
        except Exception as e:
            print(f"     ✗ 3D terrain visualization failed: {e}")

    # Create 2D animation
    if SAVE_2D_ANIMATION:
        print("\n5. Creating 2D animation...")
        try:
            print("   - 2D animation...")
            fig_anim_2d, anim_2d = visualizer.animate_ugv_path(
                waypoints,
                segments,
                start_pos,
                elevation_model=elevation_model,
                planner=planner,
                save_as=OUTPUT_2D_ANIMATION,
                sequenced_waypoints=sequenced_waypoints
            )
            plt.close(fig_anim_2d)
            gc.collect()  # Clean up after 2D animation
            print(f"     ✓ 2D animation saved as: {OUTPUT_2D_ANIMATION}")
        except Exception as e:
            print(f"     ✗ 2D animation failed: {e}")
            import traceback
            traceback.print_exc()

    # Create 3D animation
    if SAVE_3D_ANIMATION:
        print("\n6. Creating 3D animation...")
        try:
            print("   - 3D animation...")
            fig_anim_3d, anim_3d = visualizer.animate_3d_path(
                waypoints,
                segments,
                start_pos,
                elevation_model,
                save_as=OUTPUT_3D_ANIMATION,
                sequenced_waypoints=sequenced_waypoints
            )
            plt.close(fig_anim_3d)
            gc.collect()  # Clean up after 3D animation
            print(f"     ✓ 3D animation saved as: {OUTPUT_3D_ANIMATION}")
        except Exception as e:
            print(f"     ✗ 3D animation failed: {e}")

    # Create additional analysis visualizations
    print("\n7. Creating analysis visualizations...")

    # Strategy comparison chart (if strategies were compared)
    if COMPARE_STRATEGIES:
        print("   - Strategy comparison chart...")
        try:
            visualizer.plot_strategy_comparison(
                strategies,
                save_as='strategy_comparison.png'
            )
        except Exception as e:
            print(f"     ✗ Strategy comparison failed: {e}")

        # Elevation profiles for different strategies
        print("   - Elevation profiles...")
        try:
            strategies_data = {
                'Nearest Neighbor': (segments_nn, metrics_nn),
                'Elevation Greedy': (segments_elev, metrics_elev),
                'Simulated Annealing (Energy)': (segments_sa_energy, metrics_sa_energy),
                'Simulated Annealing (Distance)': (segments_sa_dist, metrics_sa_dist)
            }

            if RUN_RL_OPTIMIZER:
                strategies_data['RL (PPO)'] = (segments_rl, metrics_rl)
            visualizer.plot_elevation_profiles(
                strategies_data,
                elevation_model,
                save_as='elevation_profiles.png'
            )
        except Exception as e:
            print(f"     ✗ Elevation profiles failed: {e}")

        # Individual algorithm animations (optional)
        if SAVE_ALGORITHM_ANIMATIONS:
            print("   - Creating animations for each algorithm...")
            try:
                visualizer.create_algorithm_comparison_animations(
                    waypoints,
                    strategies_data,
                    start_pos,
                    elevation_model=elevation_model,
                    planner=planner
                )
            except Exception as e:
                print(f"     ✗ Algorithm animations failed: {e}")
        else:
            print("   - Skipping algorithm animations (SAVE_ALGORITHM_ANIMATIONS = False)")

    # Energy consumption over path (for best strategy)
    print("   - Energy consumption analysis...")
    try:
        visualizer.plot_energy_over_path(
            segments,
            planner,
            save_as='energy_over_path.png'
        )
    except Exception as e:
        print(f"     ✗ Energy analysis failed: {e}")

    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print("Visualizations saved:")
    if SAVE_2D_PLOT:
        print(f"  ✓ {OUTPUT_2D_PLOT} - 2D path visualization")
    if SAVE_3D_PLOT:
        print(f"  ✓ {OUTPUT_3D_PLOT} - 3D terrain with path")
    if SAVE_2D_ANIMATION:
        print(f"  ✓ {OUTPUT_2D_ANIMATION} - 2D animation")
    if SAVE_3D_ANIMATION:
        print(f"  ✓ {OUTPUT_3D_ANIMATION} - 3D animation with trees")
    if COMPARE_STRATEGIES:
        print(f"  ✓ strategy_comparison.png - Strategy performance comparison (4 strategies)")
        print(f"  ✓ elevation_profiles.png - Elevation profiles for all strategies")
        print(f"  ✓ algorithm_nearest_neighbor.gif - Nearest Neighbor animation")
        print(f"  ✓ algorithm_elevation_greedy.gif - Elevation Greedy animation")
        print(f"  ✓ algorithm_simulated_annealing_scipy.gif - Simulated Annealing animation")
        print(f"  ✓ algorithm_rl_ppo.gif - RL (PPO) animation")
    print(f"  ✓ energy_over_path.png - Energy consumption and elevation analysis")
    print("="*70)
    print("\nDone! UGV returns to start position after visiting all waypoints.")

if __name__ == "__main__":
    main()