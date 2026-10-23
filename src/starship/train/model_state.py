"""Full float-state capture for the opt-in world-model experiment.

Random streams stay in the simulator; finite-difference replay is restricted to calm
physics. The first 43 channels are the Markov physical state used by the model.
"""

import warp as wp

from starship.sim.physics import State

PHYSICAL_DIM = 43
STATE_NAMES = (
    "body.pos.0",
    "body.pos.1",
    "body.pos.2",
    "body.vel.0",
    "body.vel.1",
    "body.vel.2",
    "body.rot.0",
    "body.rot.1",
    "body.rot.2",
    "body.rot.3",
    "body.omega.0",
    "body.omega.1",
    "body.omega.2",
    "engine.spool.0",
    "engine.spool.1",
    "engine.spool.2",
    "engine.throttle.0",
    "engine.throttle.1",
    "engine.throttle.2",
    "engine.gimbal2.0",
    "engine.gimbal2.1",
    "engine.gimbal2.2",
    "engine.gimbal3.0",
    "engine.gimbal3.1",
    "engine.gimbal3.2",
    "engine.gimbal2_filtered.0",
    "engine.gimbal2_filtered.1",
    "engine.gimbal2_filtered.2",
    "engine.gimbal3_filtered.0",
    "engine.gimbal3_filtered.1",
    "engine.gimbal3_filtered.2",
    "flaps.angle.0",
    "flaps.angle.1",
    "flaps.angle.2",
    "flaps.angle.3",
    "flaps.filtered.0",
    "flaps.filtered.1",
    "flaps.filtered.2",
    "flaps.filtered.3",
    "tanks.0",
    "tanks.1",
    "tanks.2",
    "tanks.3",
    "attitude.roll_travel",
    "attitude.horizontal_travel",
    "attitude.peak_roll_rate",
    "attitude.peak_horizontal_rate",
    "attitude.flips",
    "attitude.reflips",
    "attitude.sideslip",
    "attitude.phase",
    "attitude.peak_dynamic_pressure",
    "attitude.peak_mach",
    "attitude.peak_load",
    "attitude.flip_altitude",
    "attitude.ignition_altitude",
    "attitude.start_offset",
    "wind.surface.0",
    "wind.surface.1",
    "wind.jet.0",
    "wind.jet.1",
    "wind.shear",
    "wind.turbulence",
    "wind.gust.0",
    "wind.gust.1",
    "wind.gust.2",
    "wind.gust_progress",
    "wind.dryden1.0",
    "wind.dryden1.1",
    "wind.dryden1.2",
    "wind.dryden2.0",
    "wind.dryden2.1",
    "wind.dryden2.2",
    "wind.shed_phase",
    "wind.flap_noise.0",
    "wind.flap_noise.1",
    "wind.flap_noise.2",
    "wind.flap_noise.3",
    "wind.wind.0",
    "wind.wind.1",
    "wind.wind.2",
    "time",
    "steps",
)
STATE_DIM = len(STATE_NAMES)


@wp.func
def pack_state(s: State, i: int, out: wp.array2d(dtype=float)):
    body = s.body[i]
    out[i, 0] = float(body.pos[0])
    out[i, 1] = float(body.pos[1])
    out[i, 2] = float(body.pos[2])
    out[i, 3] = float(body.vel[0])
    out[i, 4] = float(body.vel[1])
    out[i, 5] = float(body.vel[2])
    out[i, 6] = float(body.rot[0])
    out[i, 7] = float(body.rot[1])
    out[i, 8] = float(body.rot[2])
    out[i, 9] = float(body.rot[3])
    out[i, 10] = float(body.omega[0])
    out[i, 11] = float(body.omega[1])
    out[i, 12] = float(body.omega[2])
    engine = s.engine[i]
    out[i, 13] = float(engine.spool[0])
    out[i, 14] = float(engine.spool[1])
    out[i, 15] = float(engine.spool[2])
    out[i, 16] = float(engine.throttle[0])
    out[i, 17] = float(engine.throttle[1])
    out[i, 18] = float(engine.throttle[2])
    out[i, 19] = float(engine.gimbal2[0])
    out[i, 20] = float(engine.gimbal2[1])
    out[i, 21] = float(engine.gimbal2[2])
    out[i, 22] = float(engine.gimbal3[0])
    out[i, 23] = float(engine.gimbal3[1])
    out[i, 24] = float(engine.gimbal3[2])
    out[i, 25] = float(engine.gimbal2_filtered[0])
    out[i, 26] = float(engine.gimbal2_filtered[1])
    out[i, 27] = float(engine.gimbal2_filtered[2])
    out[i, 28] = float(engine.gimbal3_filtered[0])
    out[i, 29] = float(engine.gimbal3_filtered[1])
    out[i, 30] = float(engine.gimbal3_filtered[2])
    flaps = s.flaps[i]
    out[i, 31] = float(flaps.angle[0])
    out[i, 32] = float(flaps.angle[1])
    out[i, 33] = float(flaps.angle[2])
    out[i, 34] = float(flaps.angle[3])
    out[i, 35] = float(flaps.filtered[0])
    out[i, 36] = float(flaps.filtered[1])
    out[i, 37] = float(flaps.filtered[2])
    out[i, 38] = float(flaps.filtered[3])
    out[i, 39] = float(s.tanks[i][0])
    out[i, 40] = float(s.tanks[i][1])
    out[i, 41] = float(s.tanks[i][2])
    out[i, 42] = float(s.tanks[i][3])
    attitude = s.attitude[i]
    out[i, 43] = float(attitude.roll_travel)
    out[i, 44] = float(attitude.horizontal_travel)
    out[i, 45] = float(attitude.peak_roll_rate)
    out[i, 46] = float(attitude.peak_horizontal_rate)
    out[i, 47] = float(attitude.flips)
    out[i, 48] = float(attitude.reflips)
    out[i, 49] = float(attitude.sideslip)
    out[i, 50] = float(attitude.phase)
    out[i, 51] = float(attitude.peak_dynamic_pressure)
    out[i, 52] = float(attitude.peak_mach)
    out[i, 53] = float(attitude.peak_load)
    out[i, 54] = float(attitude.flip_altitude)
    out[i, 55] = float(attitude.ignition_altitude)
    out[i, 56] = float(attitude.start_offset)
    wind = s.wind[i]
    out[i, 57] = float(wind.surface[0])
    out[i, 58] = float(wind.surface[1])
    out[i, 59] = float(wind.jet[0])
    out[i, 60] = float(wind.jet[1])
    out[i, 61] = float(wind.shear)
    out[i, 62] = float(wind.turbulence)
    out[i, 63] = float(wind.gust[0])
    out[i, 64] = float(wind.gust[1])
    out[i, 65] = float(wind.gust[2])
    out[i, 66] = float(wind.gust_progress)
    out[i, 67] = float(wind.dryden1[0])
    out[i, 68] = float(wind.dryden1[1])
    out[i, 69] = float(wind.dryden1[2])
    out[i, 70] = float(wind.dryden2[0])
    out[i, 71] = float(wind.dryden2[1])
    out[i, 72] = float(wind.dryden2[2])
    out[i, 73] = float(wind.shed_phase)
    out[i, 74] = float(wind.flap_noise[0])
    out[i, 75] = float(wind.flap_noise[1])
    out[i, 76] = float(wind.flap_noise[2])
    out[i, 77] = float(wind.flap_noise[3])
    out[i, 78] = float(wind.wind[0])
    out[i, 79] = float(wind.wind[1])
    out[i, 80] = float(wind.wind[2])
    out[i, 81] = float(s.time[i])
    out[i, 82] = float(s.steps[i])


@wp.kernel
def capture_state(s: State, out: wp.array2d(dtype=float)):
    i = wp.tid()
    pack_state(s, i, out)


@wp.kernel
def restore_state(x: wp.array2d(dtype=float), s: State):
    i = wp.tid()
    body = s.body[i]
    body.pos[0] = float(x[i, 0])
    body.pos[1] = float(x[i, 1])
    body.pos[2] = float(x[i, 2])
    body.vel[0] = float(x[i, 3])
    body.vel[1] = float(x[i, 4])
    body.vel[2] = float(x[i, 5])
    body.rot[0] = float(x[i, 6])
    body.rot[1] = float(x[i, 7])
    body.rot[2] = float(x[i, 8])
    body.rot[3] = float(x[i, 9])
    body.omega[0] = float(x[i, 10])
    body.omega[1] = float(x[i, 11])
    body.omega[2] = float(x[i, 12])
    s.body[i] = body
    engine = s.engine[i]
    engine.spool[0] = float(x[i, 13])
    engine.spool[1] = float(x[i, 14])
    engine.spool[2] = float(x[i, 15])
    engine.throttle[0] = float(x[i, 16])
    engine.throttle[1] = float(x[i, 17])
    engine.throttle[2] = float(x[i, 18])
    engine.gimbal2[0] = float(x[i, 19])
    engine.gimbal2[1] = float(x[i, 20])
    engine.gimbal2[2] = float(x[i, 21])
    engine.gimbal3[0] = float(x[i, 22])
    engine.gimbal3[1] = float(x[i, 23])
    engine.gimbal3[2] = float(x[i, 24])
    engine.gimbal2_filtered[0] = float(x[i, 25])
    engine.gimbal2_filtered[1] = float(x[i, 26])
    engine.gimbal2_filtered[2] = float(x[i, 27])
    engine.gimbal3_filtered[0] = float(x[i, 28])
    engine.gimbal3_filtered[1] = float(x[i, 29])
    engine.gimbal3_filtered[2] = float(x[i, 30])
    s.engine[i] = engine
    flaps = s.flaps[i]
    flaps.angle[0] = float(x[i, 31])
    flaps.angle[1] = float(x[i, 32])
    flaps.angle[2] = float(x[i, 33])
    flaps.angle[3] = float(x[i, 34])
    flaps.filtered[0] = float(x[i, 35])
    flaps.filtered[1] = float(x[i, 36])
    flaps.filtered[2] = float(x[i, 37])
    flaps.filtered[3] = float(x[i, 38])
    s.flaps[i] = flaps
    s.tanks[i][0] = float(x[i, 39])
    s.tanks[i][1] = float(x[i, 40])
    s.tanks[i][2] = float(x[i, 41])
    s.tanks[i][3] = float(x[i, 42])
    attitude = s.attitude[i]
    attitude.roll_travel = float(x[i, 43])
    attitude.horizontal_travel = float(x[i, 44])
    attitude.peak_roll_rate = float(x[i, 45])
    attitude.peak_horizontal_rate = float(x[i, 46])
    attitude.flips = int(x[i, 47])
    attitude.reflips = int(x[i, 48])
    attitude.sideslip = float(x[i, 49])
    attitude.phase = int(x[i, 50])
    attitude.peak_dynamic_pressure = float(x[i, 51])
    attitude.peak_mach = float(x[i, 52])
    attitude.peak_load = float(x[i, 53])
    attitude.flip_altitude = float(x[i, 54])
    attitude.ignition_altitude = float(x[i, 55])
    attitude.start_offset = float(x[i, 56])
    s.attitude[i] = attitude
    wind = s.wind[i]
    wind.surface[0] = float(x[i, 57])
    wind.surface[1] = float(x[i, 58])
    wind.jet[0] = float(x[i, 59])
    wind.jet[1] = float(x[i, 60])
    wind.shear = float(x[i, 61])
    wind.turbulence = float(x[i, 62])
    wind.gust[0] = float(x[i, 63])
    wind.gust[1] = float(x[i, 64])
    wind.gust[2] = float(x[i, 65])
    wind.gust_progress = float(x[i, 66])
    wind.dryden1[0] = float(x[i, 67])
    wind.dryden1[1] = float(x[i, 68])
    wind.dryden1[2] = float(x[i, 69])
    wind.dryden2[0] = float(x[i, 70])
    wind.dryden2[1] = float(x[i, 71])
    wind.dryden2[2] = float(x[i, 72])
    wind.shed_phase = float(x[i, 73])
    wind.flap_noise[0] = float(x[i, 74])
    wind.flap_noise[1] = float(x[i, 75])
    wind.flap_noise[2] = float(x[i, 76])
    wind.flap_noise[3] = float(x[i, 77])
    wind.wind[0] = float(x[i, 78])
    wind.wind[1] = float(x[i, 79])
    wind.wind[2] = float(x[i, 80])
    s.wind[i] = wind
    s.time[i] = float(x[i, 81])
    s.steps[i] = int(x[i, 82])
    s.status[i] = 0
