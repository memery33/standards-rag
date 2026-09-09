# ITCP Fundamentals

A paraphrased summary of the FHWA / ARTBA guidance: *Developing Internal Traffic Control Plans (ITCPs) for Work Zones* (2016). This is the canonical reference for what an Internal Traffic Control Plan is and how it's supposed to be built.

**Source:** https://workzonesafety-media.s3.amazonaws.com/workzonesafety/files/documents/training/courses_programs/rsa_program/RSP_Guidance_Documents_Download/RSP_ITCP_Guidance_Download.pdf

This summary is paraphrased for use as internal product reference and as ground truth for AI-assisted features. Read the original document for authoritative detail before making safety claims.

---

## What an ITCP is

An ITCP is a method for coordinating the movement of workers, work vehicles, and equipment within the activity area of a roadway work zone. Its core purpose: separate workers on foot from vehicles and equipment to the extent possible, and especially reduce long backing maneuvers by trucks.

Distinct from a **Temporary Traffic Control Plan (TTCP)**, which governs how the motoring public moves *around* a work zone. The ITCP governs what happens *inside* the work space.

The two plans share principles (clear travel paths, separation of vehicles from people on foot, smooth flow) but address different audiences.

## Why ITCPs exist

Roughly 50–60 highway workers are killed each year and thousands injured when struck by vehicles or equipment at road construction sites (per FHWA data summarized in the source document). Roughly half of those fatalities are caused by work equipment striking workers on foot — not by external motorists.

The compounding hazard is **blind spots**. Every piece of construction equipment has unique blind areas. A worker who steps into one is effectively invisible to the operator. NIOSH publishes blind-area diagrams for common equipment (see `README.md`).

## The three components of every ITCP

1. **Diagrams** — the layout of the work space showing personnel and equipment movement. Need not be to scale, but must be clear enough to communicate how safety features function.
2. **Legend** — explanation of symbols. Standard MUTCD symbols where applicable; custom symbols where MUTCD doesn't cover (personnel roles, specific vehicle types).
3. **Notes** — safety points, injury-reduction measures, site-specific provisions, and duties of contractor personnel.

## Standard symbology (per the source's example legend)

| Symbol category | Examples |
|---|---|
| Equipment | Dozer (D), Paving machine (P), Roller, Bottom dump full/empty, Dump truck full/empty, Water truck, Backhoe |
| Movement | Truck movement arrows, traffic direction |
| Personnel | Foreman, Inspector (I), Spotter (S), Flagger (F), Surveyor, Worker on foot, Other type of worker |
| Devices | Channelizing device, Barrier |
| Surface | New pavement, Existing pavement |
| Zones | Prohibited area for workers on foot |

Spotter's symbol library is built to align with these conventions.

## The 8-step development process

1. **Identify project and ITCP scope.** Note hazardous areas (overhead lines, underground utilities, drop-offs, bridges, areas of close traffic). Negotiate who owns what — roadway owner, project engineer, superintendent, foremen.
2. **Determine the construction sequence.** What operations happen? At what frequency do they move? What sub-operations occur (sampling, watering, loading)?
3. **Determine vehicle/equipment/worker movements within each operation.** For each: layout, typical location, range of movement, equipment swing radii, pinch points, and blind areas. Use NIOSH blind-area diagrams. Identify worker-free ("no-go") zones.
4. **Determine vehicle/equipment movements to and from each operation.** Establish queuing locations for continuous deliveries. Designate travel lanes where multiple simultaneous operations require it.
5. **Determine safe movements for workers on foot.** Worker parking, paths from parking to work, location of portable toilets, break areas, staging. Workers should never have to cross active equipment paths during normal duties.
6. **Assess and resolve potential internal traffic conflicts.** Look for points where work operations, vehicle paths, and worker paths intersect. Adjust schedule, sequencing, or layout to eliminate.
7. **Identify who needs to understand and use the ITCP.** Safety professionals, inspectors, superintendents, workers on foot, truck supervisors, drivers (including independents), equipment operators, spotters. Each role has specific duties.
8. **Develop the communication, monitoring, and enforcement plan.** How violations are reported. Safe-backing protocols (spotters, hand signals). What workers do if they must cross vehicle paths. Method for assessing the plan's effectiveness over time. Daily updates as conditions change.

## Personnel responsibility matrix (summarized)

| Role | Key ITCP duties |
|---|---|
| Safety professionals | Create the ITCP. Train on-site personnel. Ensure supervisors are familiar with the plan. |
| Inspectors / QC | Assist in plan execution. Communicate ITCP importance to the road owner. |
| Superintendent / PM | Oversee implementation. Ensure subcontractor coordination. Update the plan as conditions change. |
| Workers on foot | Learn ITCP elements. Understand hazards. Apply training. |
| Truck supervisors | Communicate ITCP instructions to all drivers, especially independents. |
| Truck drivers | Receive ITCP instruction. Use spotters. Never operate into a blind area without checking. |
| Equipment operators | Know blind areas. Do a 360 walk-around before moving. Make eye contact with spotter. |
| Spotters | Receive training in safe spotting. Stay in continuous communication. Remain visible. |

## Operation templates (example diagrams in the source)

The guidance includes example ITCP diagrams for these common operations. Each is a candidate template for Spotter's library.

### Dirt spreading (limited backing distance)
Trucks come from cut, dump dirt, return to cut. A spotter controls vehicle backing. Trucks waiting are queued behind currently backing vehicles to limit overall backing distance.

Key safety patterns:
- Workers prohibited from crossing behind or immediately in front of trucks
- Spotter maintains line-of-sight with drivers
- Crossing prohibited in trucking area without supervisor approval

### Asphalt milling
A milling machine and supporting equipment (broom, rubber tire or bobcat). Multiple equipment pieces moving back-and-forth simultaneously.

Key safety patterns:
- Water truck only present when milling with water
- Brooms and bobcats have large rear blind spots and move unpredictably; cones may restrict worker access
- Spotter spots from the driver's side only
- Flashlight + cone required at night

### Asphalt paving
Paving machine, roller, and asphalt delivery trucks. Heavy material delivery operation. Trucks queue behind the paving operation.

Key safety patterns:
- Asphalt delivery trucks queue in the opposite-direction closed lane and behind the paving operation
- Only one backing maneuver is required in the optimized layout
- Temporary traffic control creates the queuing lane

### Asphalt patching (with breaker or milling machine)
Removal-and-replacement operations. Excavator/breaker, multiple trucks, broom, roller.

Key safety patterns:
- Do not stand between truck and excavator during loading
- Do not stand on either side of a truck while it is dumping
- Cross behind equipment only when ≥ 50 feet away (roughly 3 roller lengths)
- Eye contact with operator required before crossing
- Rollers change direction frequently — stay clear of direction changes

## Other considerations

### Worker visibility
Per MUTCD Section 6D.03.04, all workers exposed to traffic or work vehicles in a TTC zone must wear ANSI Class II or III high-visibility apparel.

- Class I / unrated: not appropriate for roadway work
- Class II: standard for road construction. Fluorescent background, retroreflective tape on front/back/sides, closed sides, front fastener
- Class III: night work or maximum-visibility daytime. Class II vest + Class E pants, or full-torso jacket with sleeves

Color: yellow-green when working around orange equipment/cones; orange-red when working around green foliage. Defer to state-specific regulations where they exist.

### Human factors
The ITCP should consider where workers naturally congregate (shade, shelter, toilets), where they walk during breaks, and how cell phones and radios affect attention. People talking on phones in noisy environments often look down and plug their other ear — missing alarms and visual hazards.

## Communication plan elements

Every ITCP should include:

- Safe zones and unsafe (no-go) zones from the diagrams
- Consequences for violating zones or site speed limits
- How violations are identified and reported
- Safe-backing requirements (spotter assignment, hand signals)
- Procedures for entering normally-restricted areas (e.g., QA testing)
- Eye contact and 360-awareness rules for crossing vehicle paths
- Chain of command for plan changes
- Method for assessing plan effectiveness

Common communication tools: air horns (for emergency stop signals), radios, hand signals, daily pre-shift huddles.

## Daily use

The ITCP must be reviewed and modified each day before the shift starts. Workers receive that day's specific instruction during the pre-shift safety briefing. Modifications during the day are permitted as conditions change, but must be communicated to all affected personnel.

---

## What this means for Spotter

The implications for Spotter's product design, distilled from the above:

- **Phases are the unit.** Daily review + frequent change = phases, exactly as we have them. ✓
- **Templates matter.** Real practitioners reuse layouts across projects. Spotter's equipment library + plan duplication features serve this directly. ✓
- **Symbol library must match.** Spotter's equipment and personnel symbols should align with MUTCD where applicable and the example legends here where it doesn't.
- **Backing minimization is the primary safety metric.** The Plan Health score should weight backing events heavily.
- **Spotter assignment is a first-class concept.** Future versions should support attaching a designated spotter to a piece of equipment, not just placing them as separate workers.
- **Pre-shift briefing is the workflow.** Huddle mode and the worker phone view are exactly the delivery channels the guidance assumes.
- **No-go zones exist in two directions.** Workers have zones they cannot enter (around backing equipment); equipment has zones it cannot enter (around breaks, parking, hazards). Spotter's zone kinds should reflect both directions.
- **Communication plan is part of the plan.** Eventually Spotter should let safety reps attach communication-plan notes (radio channels, hand signals, air horn protocol) to each phase.
