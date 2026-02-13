from src.virtual_microscope.cell_normal import NormalCell
import numpy as np
from typing import Literal, Optional
import random

class CellCycleNormal(NormalCell):
    """Cell that loops through a cell cycle indefinitely"""

    def __init__(self, *args, initial_state: Optional[Literal['G1', 'S', 'G2', 'M']] = None,
                 initial_mitosis: Optional[Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']] = None,
                 initial_time: Optional[int] = None, initial_divisions: Optional[int] = None,
                 copy_chromatin_from: Optional['CellCycleNormal'] = None,
                 initial_apoptosis: Optional[Literal['Shrinkage', 'Blebbing', 'Apoptotic bodies', 'Phagocytosis']] = None,
                 death_time: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # This cell has no fluorescence
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(self.vertices)

        # Initialize max_nb_div FIRST before using in _initial_number_division()
        self.max_nb_div: int = 10
        # Store original base_r before any cycle changes to restore after division
        self.original_base_r = self.base_r
        # G1, S, G2 -> interphase
        # M -> prophase, metaphase, anaphase, telophase
        # M -> cytokinesis
        if initial_state is not None:
            self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S'] = initial_state  # type: ignore
        else:
            self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S'] = self._initial_cycle_state()

        if initial_mitosis is not None:
            self.cell_mitosis_state: Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase'] = initial_mitosis
        else:
            self.cell_mitosis_state: Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase'] = self._initial_mitosis_state()
        
        if initial_divisions is not None:
            self.n_div: int = initial_divisions
        else:
            self.n_div: int = self._initial_number_division()
            
        self.is_dying: bool = self._initialization_apoptosis()
        # Time table defines END times for each phase (cumulative)
        # G1: 0-240, S: 240-360, G2: 360-480, M: 480-550
        self.time_table_cycle: dict[str, float] = {'G1': 240.0, 'S': 360.0, 'G2': 480.0} # in seconds
        # Mitosis phases: cumulative times from start of M phase (480s)
        # Prophase: 480-495, Metaphase: 495-510, Anaphase: 510-525, Telophase: 525-540, Cytokinesis: 540-550
        self.time_table_mitosis: dict[str, float] = {'Prophase': 495.0, 'Metaphase': 510.0, 'Anaphase': 525.0, 'Telophase': 540.0, 'Cytokinesis': 550.0}
        # add random start time point for each cell
        if initial_time is not None:
            self.current_time_life: float = float(initial_time)
        else:
            self.current_time_life: float = float(self._initial_random_time_life())
        # chromatin pts - store as offsets from center
        if copy_chromatin_from is not None:
            self.chromatin_offset = copy_chromatin_from.chromatin_offset.copy()
        else:
            self.chromatin_offset = self._initial_chromatin_offsets()
        # Physics state for cell constriction (anaphase/telophase)
        self.constriction_progress: float = 0.0  # 0.0 = no constriction, 1.0 = complete    
        self.chromatin_pts = self._update_chromatin_pts()
        self.original_r = self.r.copy()  # Store original radii for constriction calculations
        
        # Death tracking
        if death_time is not None:
            self.death_timer: float = death_time
        else:
            self.death_timer: float = 0.0
            
        if initial_apoptosis is not None:
            self.apoptosis_death_phase: Literal['Shrinkage', 'Blebbing', 'Apoptotic bodies', 'Phagocytosis'] = initial_apoptosis
        else:
            self.apoptosis_death_phase: Literal['Shrinkage', 'Blebbing', 'Apoptotic bodies', 'Phagocytosis'] = 'Shrinkage'

        self.time_table_apoptois: dict[str, float] = {'Shrinkage': 20.0, 'Blebbing': 40.0, 'Apoptotic bodies': 50.0, 'Phagocytosis': 60.0}
        self.max_death_timer: float = 60.0
        self.remove_this_cell = False
        # Transition guards to prevent multiple transitions at same threshold
        self._last_transitioned_cycle_state: Optional[str] = None
        self._last_transitioned_mitosis_state: Optional[str] = None
        self._last_transitioned_apoptosis_phase: Optional[str] = None
        self._division_occurred_this_cycle: bool = False  # Prevent re-division in same cycle
        self._ready_to_separate: bool = False  # Flag set when bridge disappears during cytokinesis


    def _initial_chromatin_offsets(self) -> list[tuple[float, float]]:
        """Creates chromatin control points as offsets from center"""
        # Create 3 random control points relative to center
        # Nucleus radius = 0.4 * base_r, so keep chromatin within ~0.3 * base_r for safety
        offsets = []
        for _ in range(3):
            # Random angle and distance for circular distribution
            angle = random.uniform(0, 2 * np.pi)
            # Constrain chromatin to inside nucleus: max radius is 0.3 * base_r
            r = self.base_r * 0.3 * np.sqrt(random.uniform(0.2, 1.0))  # Range: 0.06-0.3 × base_r
            x = r * np.cos(angle)  # Offset, not absolute
            y = r * np.sin(angle)  # Offset, not absolute
            offsets.append((x, y))

        return offsets
    
    def _update_chromatin_pts(self) -> list[tuple[float, float]]:
        """Update chromatin points based on current center position and cell size
        
        During telophase/cytokinesis, chromatin moves towards the poles
        and is constrained to stay within the cell bounds.
        """
        # Scale chromatin offsets proportionally with cell size changes
        size_ratio = self.base_r / self.original_base_r if self.original_base_r > 0 else 1.0
        
        # During heavy pinching (constriction > 0.3), move chromatin towards poles (Y-axis)
        # This keeps it inside the narrowing cell
        if self.constriction_progress > 0.3:
            # Move chromatin towards top and bottom poles as pinching increases
            pole_shift = self.center[1] * 0.3 * self.constriction_progress  # Shift along Y-axis
            
            chromatin_pts = []
            for i, offset in enumerate(self.chromatin_offset):
                # Alternate between top and bottom poles for each chromatin point
                direction = 1 if i % 2 == 0 else -1
                
                # Scale offset and shift towards pole
                new_x = self.center[0] + offset[0] * size_ratio
                new_y = self.center[1] + offset[1] * size_ratio + (pole_shift * direction)
                chromatin_pts.append((new_x, new_y))
            return chromatin_pts
        else:
            return [
                (self.center[0] + offset[0] * size_ratio, self.center[1] + offset[1] * size_ratio)
                for offset in self.chromatin_offset
            ]
    
    def _initial_cycle_state(self) -> Literal['M', 'G1', 'G2', 'S']:
        """Randomly selecte a state for a cell."""
        cell_cycle_dict = {0: 'M', 1: 'G1', 2:'G2', 3: 'S'}
        int_rand = random.randint(0,3)

        return cell_cycle_dict[int_rand]  # type: ignore
    
    def _initial_mitosis_state(self) -> Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']:
        """Randomly select a state from the Mitosis state"""
        if self.cell_cycle_state in ('G1', 'G2', 'S'):
            return 'Interphase'
        else:
            mitosis_dict = {0:'Cytokinesis', 1:'Prophase', 2:'Metaphase', 3:'Anaphase', 4:'Telophase'}

            int_rand = random.randint(0, 4)

            
            return mitosis_dict[int_rand]  # type: ignore
        
    def _initial_number_division(self) -> int:
        """Randomly select number of division for a cell."""
        int_rand = random.randint(0,self.max_nb_div) # at initialization cell can be in apoptosis state

        return int_rand
    
    def _initialization_apoptosis(self) -> bool:
        """Flag all dying cell at start"""
        if self.n_div == 10:
            return True
        else:
            return False
        
    def _initial_random_time_life(self) -> int:
        """Randomly select time life of the cell (in seconds)"""
        cell_cycle_state = self.cell_cycle_state
        cell_mitosis_state = self.cell_mitosis_state # not in M, then this is None

        # time range of each state (in seconds)
        # G1(240): 0 -> 240
        # S(360): 240 -> 360
        # G2(480): 360 -> 480
        # M(660): 480 -> 550 (with phases)
        match cell_cycle_state:
            case 'G1':
                return random.randint(0, 240)
            case 'S':
                return random.randint(240, 360)
            case 'G2':
                return random.randint(360, 480)
            case 'M': # M
                match cell_mitosis_state:
                    case 'Prophase':
                        return random.randint(480, 495)
                    case 'Metaphase':
                        return random.randint(495, 510)
                    case 'Anaphase':
                        return random.randint(510, 525)
                    case 'Telophase':
                        return random.randint(525, 540)
                    case 'Cytokinesis':  # Cytokinesis or Interphase
                        return random.randint(540, 550)
                    case _:
                        return -1 # undefined
            case _:
                return -1 # undefined

    def _change_state(self) -> None:
        """Change the state of the cell. Only transitions once per state."""
        if self.cell_mitosis_state == 'Interphase':
            # Only transition if we haven't already transitioned from this cycle state
            if (self.current_time_life >= self.time_table_cycle[self.cell_cycle_state] and 
                self._last_transitioned_cycle_state != self.cell_cycle_state):
                transiction_dict = {'G1': 'S', 'S': 'G2', 'G2': 'M'}
                # update state
                self._last_transitioned_cycle_state = self.cell_cycle_state  # Mark this state as transitioned
                self.cell_cycle_state = transiction_dict[self.cell_cycle_state]  # type: ignore
                # update immediately from G2  to M
                if self.cell_cycle_state == 'M':
                    self.cell_mitosis_state = 'Prophase'

        else: # M
            # Only transition if we haven't already transitioned from this mitosis state
            if (self.current_time_life >= self.time_table_mitosis[self.cell_mitosis_state] and 
                self._last_transitioned_mitosis_state != self.cell_mitosis_state):
                transiction_dict = {'Prophase':'Metaphase', 'Metaphase':'Anaphase', 'Anaphase':'Telophase', 'Telophase':'Cytokinesis', 'Cytokinesis':'Interphase'}

                # update mitotic state
                self._last_transitioned_mitosis_state = self.cell_mitosis_state  # Mark this state as transitioned
                self.cell_mitosis_state = transiction_dict[self.cell_mitosis_state]  # type: ignore
                # Reset telophase tracker when entering telophase
                if self.cell_mitosis_state == 'Telophase':
                    self.base_r_at_telophase = self.base_r
                # update time for new starting cycle
                if self.cell_mitosis_state == 'Interphase':
                    self.current_time_life = 0
                    self.cell_cycle_state = 'G1'  # Reset to G1 for new cycle
                    # Reset constriction and radii for fresh new cycle
                    self.constriction_progress = 0.0
                    self.r = self.original_r.copy()  # Reset to baseline (once per cycle at G1 entry)
                    # Update original_r to capture any permanent changes to cell shape
                    self.original_r = self.r.copy()
                    self._last_transitioned_cycle_state = None  # Reset cycle state guard for new cycle
                    self._last_transitioned_mitosis_state = None  # Reset mitosis guard for next M phase cycle
                    self._division_occurred_this_cycle = False  # Reset division flag for new cycle
                    self._ready_to_separate = False  # Reset separation flag for new cycle
                    self._update_cell_div_count_and_flag_apoptotic_cell() # update cell count

    def _update_cell_div_count_and_flag_apoptotic_cell(self) -> None:
        """Check the division count for the cell"""
        # Only increment if division hasn't already occurred in this cycle
        if not self._division_occurred_this_cycle:
            # Division complete - increment division count
            self.n_div += 1
            # Mark that division occurred to prevent re-division
            self._division_occurred_this_cycle = True
            # Flag as apoptotic if reached max divisions
            if self.n_div >= self.max_nb_div:
                self.is_dying = True

    def update_behavior(self, dt: float) -> None:
        """Update cell cycle state and chromatin positions."""
        # Update death timer if dying
        if self.is_dying:
            self.death_timer += dt
            self._update_apoptosis_phase()
            self._update_apoptotic_physics()
            # Apoptotic cells do NOT move - they're anchored in place during cell death
            # Do not call update_physics() to prevent migration
        else:
            # Normal cell physics for non-dying cells
            super().update_behavior(dt)
            # Update state cycle (G1 -> S -> G2 -> M)
            self._change_state()
            self.current_time_life += dt
            # Update cell cycle physics (constriction during mitosis)
            self._physic_cell_cycle()

        # Update chromatin positions to follow cell center
        self.chromatin_pts = self._update_chromatin_pts()

    def _apply_constriction_to_radius(self) -> None:
        """Apply constriction to cell membrane based on constriction progress.
        
        Equatorial vertices pinch inward while polar vertices expand strongly.
        Creates dumbbell/hourglass shape during anaphase/telophase/cytokinesis.
        Poles maintain circular shape with larger expansion.
        
        Constriction progress interpretation:
        - 0.0 to 0.3: Anaphase (gentle pinching, 30%)
        - 0.3 to 0.8: Telophase (strong pinching, 80%)
        - 0.8 to 1.0: Cytokinesis (completion, 100%)
        - > 1.0: Complete separation (bridge disappears)
        """
        if self.constriction_progress <= 0.0:
            self.r = self.original_r.copy()
            return
        
        # Get angles relative to Y-axis (vertical division axis)
        # Poles are at 90° and 270° (top and bottom)
        # Equator is at 0° and 180° (sides)
        angles = self.angles
        
        # Store pole radii for circular shape maintenance
        pole_radii = []
        pole_indices = []
        
        for i, angle in enumerate(angles):
            # Normalize angle to [0, 180] to handle symmetry
            norm_angle = angle % np.pi
            
            # Distance from equator (0 = equator, 90° = pole)
            equator_distance = np.abs(norm_angle - np.pi/2)
            
            # Calculate position factor (0 = equator, 1 = pole)
            # FIXED: was inverted - equator at 0°/180° should be 0, poles at 90°/270° should be 1
            position_factor = 1.0 - (equator_distance / (np.pi/2))
            
            # Apply constriction: equatorial vertices shrink, polar vertices expand
            # Increased pinch strength with higher constriction_progress
            # At 0.3: equator to ~0.7x, poles to ~1.15x
            # At 0.8: equator to ~0.2x, poles to ~1.4x
            # At 1.0+: equator to near 0, poles expand more
            
            equator_reduction = 0.9 * self.constriction_progress  # Equator shrinks more
            pole_expansion_factor = 0.95 * self.constriction_progress  # Poles expand even more strongly
            
            constriction_factor = 1.0 - (equator_reduction * (1.0 - position_factor))
            pole_expansion = 1.0 + (pole_expansion_factor * position_factor)
            
            # Combine factors
            total_factor = constriction_factor * pole_expansion
            
            # Clamp to prevent negative radii at extreme constriction levels
            # Minimum radius is 1% of original to maintain visible bridge
            total_factor = max(total_factor, 0.01)
            
            self.r[i] = self.original_r[i] * total_factor
            
            # Track pole vertices for circular shape maintenance
            # Lower threshold to 0.5 to include more vertices and create rounder bulges
            if position_factor > 0.5:  # Include more vertices for rounder shape
                pole_radii.append(self.r[i])
                pole_indices.append(i)
        
        # Maintain circular shape at poles by averaging pole vertices
        # This ensures poles don't become distorted and stay round
        if pole_radii and len(pole_radii) >= 2:
            avg_pole_radius = np.mean(pole_radii)
            for idx in pole_indices:
                self.r[idx] = avg_pole_radius
        
        # Area compensation: as bridge narrows, increase base_r to maintain bulge volume
        # This ensures the two daughter cell bulges stay roughly the same size
        # as the bridge decreases
        if self.constriction_progress > 0.5:
            # After midway through pinching, start compensating for area loss
            # Scale up the overall cell to compensate for bridge volume loss
            # Increased coefficient from 0.65 to 0.95 for even larger bulges
            area_compensation = 1.0 + (0.95 * (self.constriction_progress - 0.5))
            self.base_r = self.original_base_r * area_compensation
        else:
            self.base_r = self.original_base_r
    
    def _physic_cell_cycle(self) -> None:
        """Update physics based on cell cycle state.
        
        Progressive constriction across three mitotic phases:
        - Anaphase (36s):   0.0 → 0.3 (gentle pinching, 30%)
        - Telophase (36s):  0.3 → 0.8 (strong pinching, 80%)
        - Cytokinesis (10s): 0.8 → 1.2+ (completion and separation)
        """
        # Continuous constriction from anaphase through cytokinesis
        anaphase_start = self.time_table_mitosis['Anaphase']  # 552s
        cytokinesis_end = self.time_table_mitosis['Cytokinesis']  # 634s
        time_since_anaphase = self.current_time_life - anaphase_start
        total_division_duration = cytokinesis_end - anaphase_start  # 82 seconds total
        
        # Global progress through all three phases
        global_progress = time_since_anaphase / total_division_duration  # 0 to 1
        
        if self.cell_mitosis_state == 'Anaphase':
            # Anaphase: first 36s, constriction 0.0 → 0.3 (30%)
            # Gentle pinching while chromosomes move
            local_progress = (self.current_time_life - anaphase_start) / (self.time_table_mitosis['Telophase'] - anaphase_start)
            self.constriction_progress = local_progress * 0.3  # 0 to 0.3
            self._apply_constriction_to_radius()
            
        elif self.cell_mitosis_state == 'Telophase':
            # Telophase: next 36s, constriction 0.3 → 0.7 (strong pinching)
            # Continue from where anaphase left off
            anaphase_duration = self.time_table_mitosis['Telophase'] - self.time_table_mitosis['Anaphase']
            telophase_start = self.time_table_mitosis['Telophase']
            telophase_duration = self.time_table_mitosis['Cytokinesis'] - telophase_start
            time_in_telophase = self.current_time_life - telophase_start
            local_progress = time_in_telophase / telophase_duration
            
            # Ramp from 0.3 to 0.7 during telophase
            self.constriction_progress = 0.3 + (local_progress * 0.5)  # 0.3 to 0.7
            self.constriction_progress = min(self.constriction_progress, 0.7)
            self._apply_constriction_to_radius()
            
        elif self.cell_mitosis_state == 'Cytokinesis':
            # Cytokinesis: final 10s, constriction 0.8 → 1.2+ (completion)
            cytokinesis_start = self.time_table_mitosis['Telophase']  # 624s
            cytokinesis_end = self.time_table_mitosis['Cytokinesis']  # 634s
            time_in_cytokinesis = self.current_time_life - cytokinesis_start
            cytokinesis_duration = cytokinesis_end - cytokinesis_start
            
            local_progress = min(time_in_cytokinesis / cytokinesis_duration, 1.0)  # 0 to 1
            
            # Ramp from 0.8 to 1.3 during cytokinesis
            self.constriction_progress = 0.7 + (local_progress * 0.5)  # 0.7 to 1.2
            self._apply_constriction_to_radius()
            
            # Mark ready to separate when bridge width reaches ~10% of original cell width
            # Check equatorial vertices (at angles 0° and π) where constriction is strongest
            if not self._ready_to_separate:
                # Find vertices near equator (angles close to 0 or π)
                equator_indices = [i for i, angle in enumerate(self.angles) 
                                 if abs(angle) < 0.3 or abs(angle - np.pi) < 0.3]
                if equator_indices:
                    equator_radii = [self.r[i] for i in equator_indices]
                    avg_equator_radius = np.mean(equator_radii)
                    bridge_ratio = avg_equator_radius / np.mean(self.original_r[equator_indices])
                    
                    # Flag when bridge is narrower than 10% of original width
                    if bridge_ratio < 0.1:
                        self._ready_to_separate = True
        else:
            # No constriction in other phases - just reset progress
            # DO NOT reset r here - let physics handle deformations
            self.constriction_progress = 0.0

    def copy_with_reset(self) -> 'CellCycleNormal':
        """Create a sister cell with reset cycle state but copied chromatin.
        
        Used during cell division to create a daughter cell that:
        - Starts in G1 phase (timer=0)
        - Has 0 divisions completed
        - Inherits mother's chromatin pattern
        - Inherits velocity but position wraps naturally
        - Positioned offset from mother cell to simulate cytokinesis separation
        """
        sister = CellCycleNormal(
            width=self.width,
            height=self.height,
            base_radius=self.original_base_r,  # Use original radius, not compensated base_r
            vertices=self.vertices,
            seed=self.seed + 1000,  # Different seed for variation
            initial_state='G1',
            initial_mitosis='Interphase',
            initial_time=0,
            initial_divisions=0,
            copy_chromatin_from=self
        )
        # Copy position and velocity from mother, then offset sister cell
        sister.center = self.center.copy()
        sister.vel = self.vel.copy()
        
        # Separate sister and mother cells: move sister away from mother
        # Use velocity direction or a random direction if velocity is near zero
        separation_distance = self.base_r * 0.8  # Separate by ~80% of cell radius
        if np.linalg.norm(self.vel) > 0.1:
            # Move along velocity direction
            direction = self.vel / np.linalg.norm(self.vel)
        else:
            # Random direction if not moving
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        
        sister.center = (sister.center + direction * separation_distance) % np.array([self.width, self.height])
        
        return sister
    

    def _update_apoptosis_phase(self) -> None:
        """Update the current apoptotic phase. Only transitions once per phase."""
        
        transiction_dict = {'Shrinkage': 'Blebbing', 'Blebbing': 'Apoptotic bodies', 'Apoptotic bodies': 'Phagocytosis'}

        # Only transition if we haven't already transitioned from this apoptosis phase
        if (self.death_timer >= self.time_table_apoptois[self.apoptosis_death_phase] and
            self._last_transitioned_apoptosis_phase != self.apoptosis_death_phase):

            self._last_transitioned_apoptosis_phase = self.apoptosis_death_phase  # Mark this phase as transitioned
            
            if self.apoptosis_death_phase == 'Phagocytosis':
                # signal cell remove.
                self.remove_this_cell = True
            else:
                self.apoptosis_death_phase = transiction_dict[self.apoptosis_death_phase] # type: ignore


    def _get_shrinkage_progess(self) -> float:
        """Returns the progress through 'Shrinkage' phase (0-1)."""
        shrinkage_duration = self.time_table_apoptois['Shrinkage']
        progress = min(self.death_timer / shrinkage_duration, 1.0)
        return progress

    def _get_current_shrinkage_factor(self) -> float:
        """Returns radius multiplier of cell shrinkage (1.0 -> 0.70)."""
        progress = self._get_shrinkage_progess()
        # Linear shrinkage: start at 1.0, end at 0.70 (30% shrinkage)
        shrinkage_factor = 1.0 - (0.30 * progress)
        return shrinkage_factor

    def _generate_bleb_vertex_indices(self) -> tuple[list[int], list[int]]:
        """Select 2-3 primary and 2-3 secondary vertices for progressive blebs.

        Primary blebs start immediately, secondary blebs appear halfway through blebbing.
        This creates a progressive increase in membrane protrusions.

        Returns:
            tuple: (primary_bleb_indices, secondary_bleb_indices)
        """
        # Primary blebs: appear immediately at start of blebbing
        num_primary = random.randint(2, 3)
        primary_indices = random.sample(range(len(self.angles)), num_primary)

        # Secondary blebs: different vertices, appear later in blebbing phase
        remaining_indices = [i for i in range(len(self.angles)) if i not in primary_indices]
        num_secondary = random.randint(2, 3)
        secondary_indices = random.sample(remaining_indices, num_secondary)

        return sorted(primary_indices), sorted(secondary_indices)

    def _generate_chromatin_blebs(self) -> list[tuple[float, float]]:
        """Generate chromatin fragments scattered within the cell during blebbing.

        These represent fragmented nuclear material, positioned near the chromatin offsets
        but scattered to show fragmentation.
        """
        num_fragments = len(self.chromatin_offset)
        chromatin_blebs = []

        # Create fragments based on chromatin offsets, with some scattering
        for offset in self.chromatin_offset:
            # Add random displacement to show fragmentation
            scatter_radius = self.base_r * 0.1
            angle = random.uniform(0, 2 * np.pi)
            dx = scatter_radius * np.cos(angle)
            dy = scatter_radius * np.sin(angle)

            # Position relative to center
            bleb_x = self.center[0] + offset[0] + dx
            bleb_y = self.center[1] + offset[1] + dy
            chromatin_blebs.append((bleb_x, bleb_y))

        return chromatin_blebs

    def _get_apoptotic_body_positions(self) -> tuple[list[tuple[float, float]], list[int]]:
        """Generate 3-5 body positions and their chromatin fragment assignments.

        Bodies scatter within 0.5 of the original cell radius from center.
        Each body contains a chromatin fragment.

        Returns:
            tuple: (body_positions, chromatin_indices) where chromatin_indices
                   maps each body to which chromatin fragment it contains
        """
        num_bodies = random.randint(3, 5)
        original_radius = self.base_r
        max_scatter_radius = original_radius * 0.5

        positions = []
        chromatin_indices = []
        for i in range(num_bodies):
            # Spread linearly around center
            angle = (i / num_bodies) * 2 * np.pi + random.uniform(-0.2, 0.2)
            radius = random.uniform(0, max_scatter_radius)

            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            positions.append((x, y))

            # Assign chromatin fragment (cycle through available chromatin)
            chromatin_indices.append(i % min(len(self.chromatin_offset), num_bodies))

        return positions, chromatin_indices

    def _initialize_apoptotic_bodies(self) -> None:
        """Initialize 'Apoptotic Bodies' phase to set up nucleus fragments with Brownian motion.

        Store original radius, generate body positions, assign chromatin fragments,
        and initialize velocities for Brownian motion.
        """
        if not hasattr(self, 'apoptotic_body_positions'):
            body_pos, chromatin_assign = self._get_apoptotic_body_positions()
            self.apoptotic_body_positions = body_pos
            self.apoptotic_body_chromatin = chromatin_assign
        if not hasattr(self, 'original_base_r_for_apoptosis'):
            self.original_base_r_for_apoptosis = self.base_r
        if not hasattr(self, 'apoptotic_body_velocities'):
            # Initialize random velocities for Brownian motion
            # Each body gets a small random velocity in 2D space
            num_bodies = len(self.apoptotic_body_positions)
            self.apoptotic_body_velocities = []
            for _ in range(num_bodies):
                # Random direction and speed (small for realistic drift)
                angle = random.uniform(0, 2 * np.pi)
                speed = random.uniform(0.5, 2.0)  # Pixels per frame
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                self.apoptotic_body_velocities.append(np.array([vx, vy]))

    def _update_apoptotic_bodies_brownian_motion(self) -> None:
        """Update apoptotic body positions with Brownian motion.

        Bodies drift randomly with periodic direction changes,
        simulating diffusion through tissue during phagocytosis.
        """
        if not hasattr(self, 'apoptotic_body_positions') or not hasattr(self, 'apoptotic_body_velocities'):
            return

        # Update each body position with Brownian motion
        for i, pos in enumerate(self.apoptotic_body_positions):
            # Update position
            new_x = pos[0] + self.apoptotic_body_velocities[i][0]
            new_y = pos[1] + self.apoptotic_body_velocities[i][1]

            # Wrap position at world boundaries (toroidal space)
            new_x = new_x % self.width
            new_y = new_y % self.height

            self.apoptotic_body_positions[i] = (new_x, new_y)

            # Occasionally change direction (at random intervals)
            if random.random() < 0.1:  # 10% chance each update
                angle = random.uniform(0, 2 * np.pi)
                speed = random.uniform(0.5, 2.0)
                self.apoptotic_body_velocities[i] = np.array([
                    speed * np.cos(angle),
                    speed * np.sin(angle)
                ])
    
    def _update_apoptotic_physics(self) -> None:
        """Disable physics for apoptotic cells.
        
        - Stop velocity
        - Disable brownian motion
        - Disable shape relaxation to preserve irregular membrane
        - Increase membrane perturbations for irregular appearance
        - Initialize apoptotic bodies when entering that phase
        """
        # Stop all movement
        self.vel[:] = 0.0
        
        # Apply shrinkage factor during Shrinkage phase and maintain through Blebbing
        if self.apoptosis_death_phase in ('Shrinkage', 'Blebbing'):
            if self.apoptosis_death_phase == 'Shrinkage':
                shrinkage_factor = self._get_current_shrinkage_factor()
            else:
                # During blebbing, maintain the final shrinkage factor (0.70 = 30% shrinkage)
                shrinkage_factor = 0.70
            
            # Scale base_r for shrinkage (will affect nucleus size proportionally)
            if not hasattr(self, 'original_base_r_for_apoptosis'):
                self.original_base_r_for_apoptosis = self.base_r
            self.base_r = self.original_base_r_for_apoptosis * shrinkage_factor
        
        # Disable shape relaxation during all apoptotic phases to preserve irregular membrane
        if not hasattr(self, 'original_curvature_relax'):
            # Store original values on first entry to apoptosis
            self.original_curvature_relax = self.curvature_relax
            self.original_radial_relax = self.radial_relax
            self.original_ruffle_std = self.ruffle_std
        
        # During apoptosis, disable shape relaxation entirely to preserve blebs
        # Set to near-zero to stop any recovery to round shape
        self.curvature_relax = 0.001  # Almost no smoothing
        self.radial_relax = 0.0  # No pulling back to base_r
        
        # Increase ruffle std for more irregular membrane, especially during blebbing
        if self.apoptosis_death_phase == 'Blebbing':
            # Much more irregular membrane during blebbing
            self.ruffle_std = self.original_ruffle_std * 5.0  # 0.04 * 5 = 0.2
            # Initialize progressive bleb vertices and chromatin when entering blebbing phase
            if not hasattr(self, 'bleb_vertex_indices_primary'):
                primary_blebs, secondary_blebs = self._generate_bleb_vertex_indices()
                self.bleb_vertex_indices_primary = primary_blebs
                self.bleb_vertex_indices_secondary = secondary_blebs
            if not hasattr(self, 'chromatin_bleb_positions'):
                self.chromatin_bleb_positions = self._generate_chromatin_blebs()
        else:
            # Less dramatic during shrinkage and later phases
            self.ruffle_std = self.original_ruffle_std * 2.0  # 0.04 * 2 = 0.08

        # Initialize apoptotic bodies when entering that phase
        if self.apoptosis_death_phase == 'Apoptotic bodies':
            self._initialize_apoptotic_bodies()
            # Update bodies with Brownian motion
            self._update_apoptotic_bodies_brownian_motion()
        elif self.apoptosis_death_phase == 'Phagocytosis':
            # Continue Brownian motion during phagocytosis
            self._update_apoptotic_bodies_brownian_motion()