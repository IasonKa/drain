# Drain + sponge tissue: pictures and videos

**Real**: the drain geometry is your CAD (dome-cap and standoff drains). The organ is a shape made here and placed on the drain in its real coordinates, 50 um above the skin (current design) or above the ring tops (standoff).

**Model results**: the time histories, pressures and tissue movement come from the reduced-order model (axisymmetric, small deformation) with placeholder values: tissue stiffness 10 kPa, permeability 1e-14 m^4/(N s), suction 10 kPa. They show trends and design differences; the full 3D OpenFOAM + FEBio runs (hpc/) give the final numbers.


## Real geometry (your CAD) with the sponge organ

- `01_drain_with_sponge_organ_standoff.png`: New standoff-ring drain with the sponge organ lying on it.
- `video_turntable_drain_organ_standoff.mp4`: Turntable: standoff drain with the organ (real drain CAD).
- `render_cut_open_standoff.png`: Organ cut in half through the modelled hole (circled): it rests on the ring tops.
- `render_closeup_organ_seethrough_standoff.png`: Organ made see-through: the ringed holes it lies on.
- `render_drain_surface_standoff.png`: Standoff drain surface: rings with 4 slots around every hole; modelled hole circled.
- `04_section_at_hole_standoff.png`: Exact cross-section at the hole (45 deg cut through two ring arcs), organ 50 um above the rings.
- `standoff_design.png`: Your standoff design (from new_offset_desing.zip).
- `standoff_section.png`: Your standoff section (from new_offset_desing.zip).
- `01_drain_with_sponge_organ_current.png`: Current dome-cap drain with the organ.
- `video_turntable_drain_organ_current.mp4`: Turntable: current drain with the organ.
- `render_cut_open_current.png`: Current drain: organ cut in half through the modelled hole.
- `render_closeup_organ_seethrough_current.png`: Current drain under the see-through organ.
- `render_drain_surface_current.png`: Current drain surface: plain holes.
- `04_section_at_hole_current.png`: Exact cross-section at the hole, current design.

## The modelled single hole

- `02_hole_window_cutaway_standoff.png`: Single-hole window, standoff design: tissue, liquid gap, ring, hole, channel.
- `02_hole_window_cutaway_current.png`: Single-hole window, current design.
- `02_hole_window_cutaway_standoff_opt.png`: Single-hole window, optimized ring (0.50 mm high, wider, wider slots).
- `03_mesh_section_standoff.png`: Computational mesh shared by OpenFOAM (liquid) and FEBio (tissue), standoff.
- `03_mesh_section_current.png`: Computational mesh, current design.

## What happens over time (model results)

- `video_cross_section_closed.mp4`: Animation, no free liquid: current design vs the two ring designs side by side.
- `video_cross_section_open.mp4`: Animation, free wound liquid around the drain.
- `video_3d_current_closed.mp4`: 3D animation: current design, the tissue is sucked into the hole.
- `video_3d_standoff_closed.mp4`: 3D animation: standoff ring, the tissue rests on the ring.
- `video_3d_standoff_opt_closed.mp4`: 3D animation: optimized ring.
- `video_cross_section_closed_last_frame.png`: After 6 h, no free liquid (last frame of the animation).
- `render_3d_current_closed_end.png`: 3D after 6 h: current design.
- `render_3d_standoff_closed_end.png`: 3D after 6 h: standoff ring.
- `render_3d_standoff_opt_closed_end.png`: 3D after 6 h: optimized ring.
- `09_design_comparison.png`: Comparison over 6 h: tissue position and flow into the hole.
- `06_fields_current_closed.png`: Current design: pressure and liquid flow inside the tissue (4 moments).
- `06_fields_standoff_closed.png`: Standoff ring: pressure and liquid flow inside the tissue.
- `06_fields_standoff_opt_closed.png`: Optimized ring: pressure and liquid flow inside the tissue.
- `05_timeline_current_closed.png`: Current design: time history (no free liquid).
- `05_timeline_standoff_closed.png`: Standoff ring: time history (no free liquid).
- `05_timeline_standoff_opt_closed.png`: Optimized ring: time history (no free liquid).
- `05_timeline_standoff_open.png`: Standoff ring: time history with free liquid.
- `05_timeline_current_open.png`: Current design: time history with free liquid.

## Design study

- `08_standoff_screening.png`: 45 ring variants: height, width, slot width.
- `07_parameter_map.png`: Current design: tissue stiffness x suction.
