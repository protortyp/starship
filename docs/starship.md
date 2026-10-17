# Starship reference values

Upper stage (Ship), V2 geometry, Raptor 2 unless noted. x = station from the engine plane. Basis: **O** official (SpaceX, FAA), **S** study or flight data, **E** estimate.

| Parameter | Value | Range | Basis | Sources |
|---|---|---|---|---|
| Length / diameter | 52.1 m / 9 m (V1 50.3 m, V3 ~52–53 m) | | O | [1] [2] [3] |
| Nose cone (ogive) length | 14 m | 13–15 m | S | [4] |
| Engine skirt height | 7.3 m | | S | [5] |
| Common dome / header tanks CH4, LOX | x ≈ 23.5 m / ≈ 46 m, ≈ 49.5 m (LOX header in the nose tip) | ±1.5 m | E | [4] [6] |
| Dry mass | 160 t (V2, telemetry 161 ± 4 t; V1 149 t); design studies 118–120 t | 120–165 t | S/E | [8] [9] [34] |
| Propellant capacity | 1,500 t (1,170 LOX + 330 CH4); V1 1,200 t, V3 ~1,600 t | | O | [2] [6] [10] |
| Mixture ratio O/F | 3.6 | | O | [2] |
| Header (landing) propellant | 23 t (18 LOX + 5 CH4) | 15–30 t | E | [6] [11] |
| Propellant used in flip + landing | 10–20 t | | E | [8] |
| Landing-start mass | ~183 t (dry 160 t + 23 t header) | 140–195 t | E | derived |
| CoM from engine plane, dry / landing | 21.4 m (0.41 L) / 24.9 m (0.48 L) | 19–23 / 22–27 m | E | `starship.sim.mass` with 30 t aft structure; studies assume 0.58–0.60 L from nose [12] [13] |
| Inertia pitch = yaw, dry / landing | 3.7e7 / 5.2e7 kg·m² | 2.5e7–7e7 | E | `starship.sim.mass`, [13] |
| Inertia roll | 2.9e6 kg·m² | 1.8e6–3e6 | E | `starship.sim.mass` |
| Sea-level Raptors (landing) | 3, gimballed, radius ≈ 1.15 m, pivot x ≈ 3.1 m | radius 1.0–1.3 m | E | [4] [5] |
| Thrust per engine, sea level | 2.26 MN (230 tf); R3 2.75 MN (280 tf, Wikipedia 250 tf) | ±5 % | O | [14] [15] [16] |
| Thrust vs ambient pressure | F = F_vac − p_a·A_e; F_vac 2.39 MN (R3 2.88 MN) | | S | [15] [16] |
| Nozzle exit area | 1.33 m² (D ≈ 1.3 m) | | S | [16] |
| Isp sea level / vacuum | 327 / 347 s (R3 ~333 / 350 s) | ±1 % | O/S | [14] [15] |
| Mass flow at full thrust | 703 kg/s (R3 840 kg/s) | | S | derived from thrust, Isp |
| Chamber pressure | 300 bar (R3 330 bar) | | S | [16] |
| Throttle range | 40–100 % | min 35–50 % | S | [15] [17] |
| Gimbal limit | ±15°, two axes | ±10–15° | S | [15] |
| Gimbal rate / lag | 20 °/s, 0.1 s | 10–30 °/s | E | [18] |
| Ignition to full thrust | 2 s | 1–3 s | S | [19] [20] |
| Engine sequence (landing) | light 3 → 2 at +9–11 s → 1 for the last ~2–5 s | | O | [33] [19] [20] |
| Engine dry mass | 1,630 kg (R3 1,525 kg) | | O | [14] |
| Flaps | 4: aft pair 180° apart; forward pair 140° apart, shifted leeward (V1 180°) | | S | [4] [21] |
| Aft flap area / outboard extent | 40 m² each / 4.5 m (tip-to-tip span 18 m) | 35–45 m² | E | [12] [22] |
| Aft flap hinge | parallel to body axis, x ≈ 1–12 m | ±1 m | S/E | [12] |
| Forward flap area | 18 m² each (V2 16 m²) | 14–25 m² | E | [12] [22] |
| Forward flap hinge | along the nose generatrix (≈ 10–15° to the axis), x ≈ 39–47 m | ±1.5 m | S/E | [4] [12] |
| Flap angle range | 0° (flat) to 85° (stowed) | max 80–90° | S | [12] |
| Flap rate / lag | 25 °/s, 0.15 s | 15–40 °/s, 0.1–0.3 s | E | |
| RCS | V2 has no dedicated thrusters; vents act as thrusters, incl. 6 roll; 2 kN each | 0.5–5 kN, Isp 60–110 s | E | [6] [23] |
| Belly-flop angle of attack | 90° | trim 65–120° | S | [12] [13] |
| Terminal velocity, sea level | 85 m/s at SN-era ~120 t; ~100 m/s at 183 t (∝ √m) | 75–105 m/s | S | [11] [24] |
| Drag area C_D·A at α = 90°, sea level | 280 m² | 230–340 m² | S | [11] derived |
| Drag area C_D·A at α = 90°, M 0.8 | 600 m² | 550–700 m² | S | [12] |
| Crossflow C_dn at Re ≈ 5e7 (tiles) | 0.7 | 0.5–0.9 | S | [25] [26] |
| Finite-length factor η | 0.63 subsonic → 1 supersonic | | S | [25] [26] |
| CoP from nose, body / flaps flat | 0.54 L / 0.60 L | 0.52–0.56 / 0.57–0.62 L | S | [12] |
| Pitch stability | Cmα < 0 for α < 92–95°; damping stabilizing for α < 135° | | S | [27] |
| Lateral-directional | Dutch roll unstable open-loop | | S | [27] |
| Belly-flop duration (flight tests) | 110 s subsonic | 90–126 s | S | [28] [19] [20] |
| Flip / relight altitude | 600 m | 450 m–1.5 km | S | [24] [11] |
| Speed at flip start | 80–110 m/s | | S | [11] [13] |
| Flip rotation time | 5 s | 4–7 s | E | [11] |
| Landing burn, relight to touchdown | 20 s | 16–28 s | O/S | [33] [19] [20] |
| Landing accuracy | ~3 m (Flight 10) | | O | [30] |
| Touchdown that destroyed the legs | 10 m/s (SN10, landed on 1 of 3 engines) | | O | [7] [29] [31] |

## Models

| Component | Model | Sources |
|---|---|---|
| Atmosphere | U.S. Standard Atmosphere 1976, 7 layers to 86 km | NASA-TM-X-74335 |
| Hull normal force | strip-wise viscous crossflow: dN/dx = η·C_dn(M_n, Re_n)·½ρV_n²·w(x), plus slender-body term (A_b/A)·sin2α·cos(α/2) | [25] [26] |
| Hull axial force | C_A = C_A0·cos²α, C_A0 ≈ 0.4 nose-first, ≈ 0.9 tail-first | [25] |
| Flap normal force | flat plate, subsonic C_N = 1.6·sinα·cosα + 1.17·sin²α; Newtonian for M ≥ 2 | [32] |
| Terminal velocity check | V_t = sqrt(2mg / (ρ·C_D·A)) | |

## Sources

1. https://www.spacex.com/vehicles/starship/design
2. https://www.faa.gov/media/94346
3. https://en.wikipedia.org/wiki/SpaceX_Starship_(spacecraft)
4. https://ringwatchers.com/article/s33-nose
5. https://ringwatchers.com/article/s33-aft
6. https://ringwatchers.com/article/s33-tanks
7. https://twitter.com/elonmusk/status/1369379914139451406 (SN10 touchdown)
8. https://www.emmastrong.net/wp-content/uploads/2022/01/WStrong_CulminatingExperience.pdf
9. https://www.reddit.com/r/SpaceXLounge/comments/1oe4c1v/block_2_starship_dry_mass_estimate_from_ift/ (telemetry dry mass)
10. https://www.nasaspaceflight.com/2026/03/ship-39-preflight-test-objectives/
11. https://everydayastronaut.com/starships-belly-flop-maneuver/
12. https://html.rhhz.net/KQDLXXB/2022-05-1.htm (Li et al. 2022, CFD)
13. https://arxiv.org/abs/2508.06520
14. https://www.spacex.com/updates/reusability
15. https://everydayastronaut.com/spacex-raptor-engine-comparison/
16. https://en.wikipedia.org/wiki/SpaceX_Raptor
17. https://twitter.com/elonmusk/status/1295553672454311941
18. https://www.mdpi.com/2227-7390/11/2/288
19. https://en.wikipedia.org/wiki/Starship_flight_test_11
20. https://en.wikipedia.org/wiki/Starship_flight_test_12
21. https://www.teslarati.com/spacex-starship-forward-flap-tweaks-elon-musk/
22. https://forum.nasaspaceflight.com/index.php?topic=50049.100
23. https://www.teslarati.com/spacex-starship-hot-gas-thruster-photos/
24. https://everydayastronaut.com/starship-sn15-medium-altitude-test-flight/
25. https://ntrs.nasa.gov/api/citations/19730006261/downloads/19730006261.pdf (Jorgensen, NASA TN D-6996)
26. https://ntrs.nasa.gov/citations/19770026166 (Jorgensen, NASA TR R-474)
27. https://www.mdpi.com/2073-8994/14/9/1862
28. https://en.wikipedia.org/wiki/List_of_Starship_upper_stage_flight_tests
29. https://www.universetoday.com/150477/fantastic-analysis-of-sn-10-landing-and-explosion-by-djsnm-1/
30. https://www.digitaltrends.com/space/starship-precise-ocean-landing-10th-flight/
31. https://www.engadget.com/sn-10-starship-explosion-elon-musk-070217155.html
32. https://www.pdas.com/refs/rep1048.pdf (Allen & Perkins, NACA Report 1048)
33. https://www.spacex.com/launches/starship-flight-13
34. https://doi.org/10.1007/s12567-025-00625-8 (DLR 2025, V1 mass model)
