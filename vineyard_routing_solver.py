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
from functools import lru_cache

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

    def _generate_gentle_terrain(self) -> np.ndarray:
        """Generate gentle vineyard terrain (original model)"""
        base_elevation = 100.0  # meters above sea level

        # Linear slope (8% in x, 3% in y)
        slope_x = 0.08
        slope_y = 0.03

        linear_elevation = (base_elevation +
                          slope_x * self.X +
                          slope_y * self.Y)

        # Add terrain undulations
        undulation1 = 2.0 * np.sin(0.05 * self.X) * np.cos(0.03 * self.Y)
        undulation2 = 1.5 * np.sin(0.08 * self.X + np.pi/4) * np.sin(0.06 * self.Y)
        undulation3 = 0.8 * np.cos(0.1 * self.X) * np.cos(0.1 * self.Y)

        elevation = linear_elevation + undulation1 + undulation2 + undulation3

        return elevation

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

class GridConstrainedPlanner:
    """Plans grid-compliant paths for vineyard navigation"""

    def __init__(self, grid: VineyardGrid, elevation_model: Optional[ElevationModel] = None):
        self.grid = grid
        self.elevation = elevation_model
        self.velocity = 1.5  # m/s
        self.turn_time = 6.0  # seconds per 90° turn

        # UGV physical parameters for energy calculation
        self.ugv_mass = 150.0  # kg (typical agricultural UGV)
        self.g = 9.81  # m/s^2 (gravitational acceleration)
        self.rolling_resistance = 0.15  # coefficient for rough terrain
        self.mechanical_efficiency = 0.75  # motor/drivetrain efficiency
        self.air_resistance_coeff = 0.5  # Cd * A (drag coefficient * frontal area)
        
    def sequence_waypoints(self, waypoints: List[Waypoint], start_pos: Tuple[float, float], mode: str = 'nearest_neighbor') -> List[Waypoint]:
        """
        Phase 1: Sequence waypoints using different strategies

        Parameters:
        -----------
        waypoints: List of waypoints to sequence
        start_pos: Starting position
        mode: 'nearest_neighbor' or 'elevation' - sequencing strategy

        Returns:
        --------
        Ordered list of waypoints
        """
        if mode == 'elevation':
            return self._sequence_by_elevation(waypoints, start_pos)
        else:  # nearest_neighbor
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
                          sequencing_mode: str = 'elevation') -> Tuple[List[GridSegment], dict]:
        """
        Plan complete tour visiting all waypoints in alleys between rows

        Parameters:
        -----------
        waypoints: List of waypoints to visit
        start_pos: Starting position
        sequencing_mode: 'nearest_neighbor' or 'elevation' - waypoint ordering strategy

        Returns:
        --------
        Tuple of (segments, metrics)
        """
        # Phase 1: Sequence waypoints
        sequence = self.sequence_waypoints(waypoints, start_pos, mode=sequencing_mode)

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

        # Calculate metrics
        metrics = self.calculate_metrics(all_segments)

        return all_segments, metrics
    
    def calculate_energy_for_segment(self, seg: GridSegment) -> dict:
        """
        Calculate energy consumption for a single segment.

        Returns:
            dict with energy components in Joules (J)
        """
        distance = seg.distance()

        # Elevation change
        if seg.elevation_start is not None and seg.elevation_end is not None:
            elevation_change = seg.elevation_end - seg.elevation_start
        else:
            elevation_change = 0.0

        # 1. Potential energy change (climbing/descending)
        # E_potential = m * g * Δh
        potential_energy = self.ugv_mass * self.g * elevation_change

        # Only count energy when climbing (positive elevation change)
        # Descending can use regenerative braking or just coasting
        climbing_energy = max(0, potential_energy)

        # 2. Rolling resistance energy
        # E_rolling = m * g * C_rr * distance
        rolling_energy = self.ugv_mass * self.g * self.rolling_resistance * distance

        # 3. Air resistance energy (at constant velocity)
        # E_air = 0.5 * rho * Cd * A * v^2 * distance
        # Simplified: using coefficient instead of full calculation
        air_resistance_energy = 0.5 * self.air_resistance_coeff * (self.velocity ** 2) * distance

        # 4. Total mechanical energy required
        mechanical_energy = climbing_energy + rolling_energy + air_resistance_energy

        # 5. Account for mechanical efficiency (motors, drivetrain losses)
        total_energy = mechanical_energy / self.mechanical_efficiency

        return {
            'distance': distance,
            'elevation_change': elevation_change,
            'potential_energy': potential_energy,
            'climbing_energy': climbing_energy,
            'rolling_energy': rolling_energy,
            'air_resistance_energy': air_resistance_energy,
            'mechanical_energy': mechanical_energy,
            'total_energy': total_energy
        }

    def calculate_metrics(self, segments: List[GridSegment]) -> dict:
        """Calculate path metrics including distance, time, and energy (optimized)"""
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

        return {
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
    
    def _plot_waypoints(self, ax, waypoints):
        """Plot waypoint locations"""
        for i, wp in enumerate(waypoints):
            ax.plot(wp.x, wp.y, 'ro', markersize=8, label='Waypoints' if i == 0 else '')
            ax.text(wp.x + 1, wp.y + 0.5, f'W{i+1}', fontsize=8)
    
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
        text += f"Air Drag: {metrics['air_resistance_energy_j']/1000:.2f} kJ\n"
        text += f"Turns: {metrics['turn_energy_j']/1000:.2f} kJ\n\n"

        text += f"Elevation:\n"
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
                        save_as: Optional[str] = None):
        """
        Create animation of UGV traveling along the planned path

        Parameters:
        -----------
        waypoints: List of waypoints to visit
        segments: Path segments to follow
        start_pos: Starting position
        elevation_model: Optional elevation model for terrain visualization
        planner: Optional planner for UGV parameters (mass, velocity, etc.)
        save_as: Optional filename to save animation (e.g., 'ugv_animation.gif')
        """
        # Default UGV parameters if planner not provided
        if planner is None:
            ugv_mass = 10.0
            ugv_g = 9.81
            rolling_resistance = 0.15
            mechanical_efficiency = 0.75
            air_resistance_coeff = 0.5
            velocity = 1.5
        else:
            ugv_mass = planner.ugv_mass
            ugv_g = planner.g
            rolling_resistance = planner.rolling_resistance
            mechanical_efficiency = planner.mechanical_efficiency
            air_resistance_coeff = planner.air_resistance_coeff
            velocity = planner.velocity
        # Create figure with 2 subplots (2D path and elevation profile)
        fig = plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1)  # 2D top view
        ax2 = plt.subplot(1, 2, 2)  # Elevation profile

        # Plot static vineyard structure on ax1
        self._plot_vineyard_structure(ax1)
        self._plot_waypoints(ax1, waypoints)
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
                    # Air resistance
                    air_energy = 0.5 * air_resistance_coeff * (velocity ** 2) * dist_3d
                    # Total mechanical energy with efficiency
                    segment_energy = (potential_energy + rolling_energy + air_energy) / mechanical_efficiency

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

            return ugv, ugv_trail, metrics_text, elev_marker if elev_marker else ugv

        # Create animation (optimized with faster frame rate)
        anim = FuncAnimation(fig, animate, frames=len(path_points),
                           interval=30, blit=True, repeat=True)

        # Save if requested
        if save_as:
            print(f"Saving animation to {save_as}...")
            # Faster frame rate and reduced DPI for faster saving
            writer = PillowWriter(fps=30)
            anim.save(save_as, writer=writer, dpi=80)
            print(f"Animation saved!")

        plt.tight_layout()
        return fig, anim

    def plot_3d_terrain(self, waypoints: List[Waypoint],
                       segments: List[GridSegment],
                       start_pos: Tuple[float, float],
                       elevation_model: ElevationModel,
                       save_as: Optional[str] = 'vineyard_3d_terrain.png'):
        """
        Create 3D visualization of terrain with UGV path

        Parameters:
        -----------
        waypoints: List of waypoints to visit
        segments: Path segments to follow
        start_pos: Starting position
        elevation_model: Elevation model for terrain surface
        save_as: Optional filename to save figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')

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

        # Plot waypoints in 3D
        for i, wp in enumerate(waypoints):
            ax.scatter(wp.x, wp.y, wp.z, c='red', s=100, marker='o',
                      edgecolors='darkred', linewidth=2, zorder=5)
            ax.text(wp.x, wp.y, wp.z + 0.5, f'W{i+1}', fontsize=8, zorder=5)

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

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

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
                       save_as: Optional[str] = None):
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

        # Plot waypoints (larger and more visible)
        for i, wp in enumerate(waypoints):
            ax.scatter(wp.x, wp.y, wp.z, c='red', s=80, marker='o',
                      edgecolors='darkred', linewidth=1.5, zorder=5)

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

        # Set initial view
        ax.view_init(elev=25, azim=45)

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
                azim = 45 + (frame / len(path_points_3d)) * 90  # Rotate 90 degrees during animation
                ax.view_init(elev=25, azim=azim)

            return ugv, trail

        # Create animation (optimized with faster frame rate)
        anim = FuncAnimation(fig, animate, frames=len(path_points_3d),
                           interval=30, blit=False, repeat=True)

        # Save if requested
        if save_as:
            print(f"Saving 3D animation to {save_as}...")
            # Faster frame rate and reduced DPI for faster saving
            writer = PillowWriter(fps=30)
            anim.save(save_as, writer=writer, dpi=80)
            print(f"3D animation saved!")

        plt.tight_layout()
        return fig, anim

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

def main():
    """Main function to demonstrate the vineyard routing solution with elevation, energy, and animation"""

    # ========================================================================
    # USER CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
    # ========================================================================

    # Vineyard Grid Parameters
    NUM_ROWS = 20              # Number of vine rows
    ROW_SPACING = 3.0          # Distance between rows (meters)
    ROW_LENGTH = 100.0         # Length of each row (meters)
    TREE_SPACING = 1.0         # Distance between vine trees along row (meters)

    # Terrain Parameters
    TERRAIN_TYPE = 'mosel'     # 'mosel' for hill-shaped or 'gentle' for gentle slopes
    TERRAIN_RESOLUTION = 0.5   # Grid resolution for elevation sampling (meters)
    BASE_ELEVATION = 150.0     # Base elevation in meters (configured in ElevationModel)
    MAX_ELEVATION_GAIN = 30.0  # Maximum elevation gain along row (configured in ElevationModel)

    # Waypoint Generation
    NUM_WAYPOINTS = 10         # Number of random waypoints to generate

    # UGV Parameters (defined in GridConstrainedPlanner, listed here for reference)
    # UGV_MASS = 150.0         # kg (typical agricultural UGV)
    # UGV_VELOCITY = 1.5       # m/s (average travel velocity)
    # ROLLING_RESISTANCE = 0.15  # coefficient for rough terrain
    # MECHANICAL_EFFICIENCY = 0.75  # motor/drivetrain efficiency (0-1)
    # AIR_RESISTANCE_COEFF = 0.5    # Cd * A (drag coefficient * frontal area)
    # TURN_TIME = 6.0          # seconds per 90° turn

    # Sequencing Strategy
    COMPARE_STRATEGIES = True  # Compare both sequencing strategies
    DEFAULT_STRATEGY = 'elevation'  # 'nearest_neighbor' or 'elevation'

    # Visualization Parameters
    SAVE_2D_PLOT = True        # Save 2D static visualization
    SAVE_3D_PLOT = True        # Save 3D terrain visualization
    SAVE_2D_ANIMATION = True   # Save 2D animation (GIF)
    SAVE_3D_ANIMATION = True   # Save 3D animation (GIF)

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

    # Plan tours with sequencing strategies
    print("3. Planning grid-constrained tours...")

    if COMPARE_STRATEGIES:
        print("\n   3a. Nearest Neighbor sequencing...")
        segments_nn, metrics_nn = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='nearest_neighbor')

        print("   3b. Elevation-based sequencing...")
        segments_elev, metrics_elev = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode='elevation')

        # Compare sequencing strategies
        print("\n" + "="*70)
        print("SEQUENCING STRATEGY COMPARISON")
        print("="*70)

        print("\nNearest Neighbor Strategy:")
        print(f"  Total Distance (3D): {metrics_nn['total_distance_3d']:.1f} m")
        print(f"  Total Energy: {metrics_nn['total_energy_kj']:.2f} kJ")
        print(f"  Elevation Gain: {metrics_nn['total_elevation_gain']:.1f} m")
        print(f"  Elevation Loss: {metrics_nn['total_elevation_loss']:.1f} m")
        print(f"  Total Time: {metrics_nn['total_time']:.1f} s ({metrics_nn['total_time']/60:.1f} min)")

        print("\nElevation-Optimized Strategy:")
        print(f"  Total Distance (3D): {metrics_elev['total_distance_3d']:.1f} m")
        print(f"  Total Energy: {metrics_elev['total_energy_kj']:.2f} kJ")
        print(f"  Elevation Gain: {metrics_elev['total_elevation_gain']:.1f} m")
        print(f"  Elevation Loss: {metrics_elev['total_elevation_loss']:.1f} m")
        print(f"  Total Time: {metrics_elev['total_time']:.1f} s ({metrics_elev['total_time']/60:.1f} min)")

        print("\nComparison:")
        dist_diff = ((metrics_elev['total_distance_3d'] - metrics_nn['total_distance_3d']) / metrics_nn['total_distance_3d']) * 100
        energy_diff = ((metrics_elev['total_energy_kj'] - metrics_nn['total_energy_kj']) / metrics_nn['total_energy_kj']) * 100
        elev_gain_diff = metrics_elev['total_elevation_gain'] - metrics_nn['total_elevation_gain']
        time_diff = ((metrics_elev['total_time'] - metrics_nn['total_time']) / metrics_nn['total_time']) * 100

        print(f"  Distance difference: {dist_diff:+.1f}%")
        print(f"  Energy difference: {energy_diff:+.1f}%")
        print(f"  Elevation gain difference: {elev_gain_diff:+.1f} m")
        print(f"  Time difference: {time_diff:+.1f}%")

        # Determine better strategy
        if metrics_elev['total_energy_kj'] < metrics_nn['total_energy_kj']:
            print(f"\n  ✓ Elevation-optimized strategy saves {abs(energy_diff):.1f}% energy!")
            better_strategy = "elevation"
            segments = segments_elev
            metrics = metrics_elev
        else:
            print(f"\n  ✓ Nearest neighbor strategy uses {abs(energy_diff):.1f}% less energy")
            better_strategy = "nearest_neighbor"
            segments = segments_nn
            metrics = metrics_nn
    else:
        # Use default strategy without comparison
        print(f"\n   Using {DEFAULT_STRATEGY} strategy...")
        segments, metrics = planner.plan_complete_tour(waypoints, start_pos, sequencing_mode=DEFAULT_STRATEGY)
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
    print(f"  - Air resistance: {metrics['air_resistance_energy_j']/1000:.2f} kJ ({metrics['air_resistance_energy_j']/metrics['total_energy_j']*100:.1f}%)")
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
                save_as=OUTPUT_3D_PLOT
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
                save_as=OUTPUT_2D_ANIMATION
            )
            plt.close(fig_anim_2d)
            print(f"     ✓ 2D animation saved as: {OUTPUT_2D_ANIMATION}")
        except Exception as e:
            print(f"     ✗ 2D animation failed: {e}")

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
                save_as=OUTPUT_3D_ANIMATION
            )
            plt.close(fig_anim_3d)
            print(f"     ✓ 3D animation saved as: {OUTPUT_3D_ANIMATION}")
        except Exception as e:
            print(f"     ✗ 3D animation failed: {e}")

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
    print("="*70)
    print("\nDone!")

if __name__ == "__main__":
    main()