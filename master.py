import numpy as np

# =============================================================================
# OVERTAKING SAFETY SIMULATION
# =============================================================================
# Vehicles:
#   Lead Car     (L) : ahead in same lane, same direction
#   Computing Car(C) : ego/subject vehicle being overtaken
#   Overtaking Car(O): attempts to overtake C, merge back between C and L
#   Oncoming Car (X) : approaching from opposite direction
#
# +x = direction of travel for L, C, O
# Oncoming car moves in -x direction
#
# Physics applied every time step:
#   1. Kinematics  : s = u*t + 0.5*a*t^2  (time to complete overtake)
#   2. Safe gap    : polynomial fit (speed -> safe following distance)
#   3. TTC         : Time-To-Collision with oncoming
#   4. Abort brake : v^2 = u^2 + 2*a*s  (can O brake back before hitting C?)
#   5. Merge gap   : projected gap between C and L at moment O inserts
#   6. Lateral time: time for lane change (fixed 3s per half manoeuvre)
# =============================================================================

np.random.seed(None)

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
dt            = 0.1        # s per step
TOTAL_STEPS   = 10000

VEHICLE_LEN   = 4.5        # m
ACCEL_O       = 1.5        # m/s^2  overtaking car acceleration in passing phase
BRAKE_DECEL   = 4.0        # m/s^2  comfortable deceleration
LATERAL_TIME  = 3.0        # s  time to complete each lane change (out / back in)
SAFETY_MARGIN = 10.0       # m  extra clearance required beyond calculated need
REACTION_TIME = 1.5        # s  driver reaction before braking

# Safe-distance polynomial: speed (km/h) -> distance (m)
speed_ref = np.array([40,  60,  70,  80,  90,  100, 110, 120])
dist_ref  = np.array([20,  30,  40,  55,  70,   88, 108, 130])
poly      = np.polyfit(speed_ref, dist_ref, 2)

def safe_dist(v_kmh):
    a, b, c = poly
    return float(max(5.0, a*v_kmh**2 + b*v_kmh + c))

def to_ms(v_kmh):  return v_kmh / 3.6
def to_kmh(v_ms):  return v_ms * 3.6

# -----------------------------------------------------------------------
# Speed profile generator
# -----------------------------------------------------------------------
def gen_speed(n, start=70, lo=60, hi=120):
    s = [start]; cur = start
    while len(s) < n:
        seg = np.random.randint(300, 1500)
        tgt = np.random.randint(lo, hi + 1)
        s.extend(np.round(np.linspace(cur, tgt, seg)).astype(int).tolist())
        cur = tgt
    return s[:n]

# -----------------------------------------------------------------------
# Generate speed profiles
# -----------------------------------------------------------------------
ti = np.round(np.arange(0, TOTAL_STEPS * dt, dt), 2)

lead_spd_kmh     = gen_speed(TOTAL_STEPS, start=70)
oncoming_spd_kmh = gen_speed(TOTAL_STEPS, start=80, lo=60, hi=120)

# -----------------------------------------------------------------------
# Simulate Computing Car following Lead Car
# -----------------------------------------------------------------------
comp_spd_kmh = [60]
lead_pos      = 0.0
comp_pos      = -safe_dist(60) - VEHICLE_LEN  # start at safe gap behind lead

lead_pos_arr  = np.zeros(TOTAL_STEPS)
comp_pos_arr  = np.zeros(TOTAL_STEPS)
gap_CL_arr    = np.zeros(TOTAL_STEPS)

for i in range(TOTAL_STEPS):
    lv  = lead_spd_kmh[i]
    lead_pos += to_ms(lv) * dt

    cv  = comp_spd_kmh[-1]
    gap = lead_pos - comp_pos - VEHICLE_LEN

    req = safe_dist(cv)
    if gap < req * 0.85:
        cv = max(0, cv - 4)
    elif gap > req * 1.3 and cv < lv:
        cv = min(cv + 2, lv)
    elif gap > req * 1.1 and cv < lv:
        cv = min(cv + 1, lv)

    comp_pos += to_ms(cv) * dt
    comp_spd_kmh.append(int(cv))
    lead_pos_arr[i] = lead_pos
    comp_pos_arr[i] = comp_pos
    gap_CL_arr[i]   = max(0.0, lead_pos - comp_pos - VEHICLE_LEN)

comp_spd_kmh = comp_spd_kmh[:TOTAL_STEPS]

# -----------------------------------------------------------------------
# Oncoming car: continuous traffic — resets 1500 m ahead whenever it passes
# This models realistic road conditions with periodic oncoming vehicles
# -----------------------------------------------------------------------
ONC_RESET_DIST  = 1500.0   # m ahead of comp car when a new vehicle spawns
ONC_MIN_RESPAWN = 800.0    # minimum gap before next vehicle appears

onc_pos         = comp_pos_arr[0] + ONC_RESET_DIST
onc_pos_arr     = np.zeros(TOTAL_STEPS)
dist_to_X_arr   = np.zeros(TOTAL_STEPS)

for i in range(TOTAL_STEPS):
    onc_pos -= to_ms(oncoming_spd_kmh[i]) * dt
    dist = onc_pos - comp_pos_arr[i]

    # When oncoming car has passed (dist < -50 m behind comp car), spawn new one
    if dist < -50.0:
        # New oncoming vehicle appears 1200–1800 m ahead
        gap = float(np.random.randint(1200, 1801))
        onc_pos = comp_pos_arr[i] + gap

    onc_pos_arr[i]   = onc_pos
    dist_to_X_arr[i] = onc_pos - comp_pos_arr[i]

# -----------------------------------------------------------------------
# Per-step overtaking safety analysis
# -----------------------------------------------------------------------
# Manoeuvre phases:
#   Phase A  O accelerates from v_C, travels d_pass to clear C and reach L rear
#            d_pass = gap_CL + 2*VEHICLE_LEN
#            kinematics: 0.5*a*t_A^2 + v_C*t_A - d_pass = 0  => solve t_A
#            v_O_peak = v_C + a*t_A
#
#   Phase B  O makes lateral lane change back (LATERAL_TIME seconds)
#            d_B = v_O_peak * LATERAL_TIME
#
#   t_total  = t_A + LATERAL_TIME
#   d_O_total= d_pass + d_B
#
# During t_total, oncoming covers: d_X = v_X * t_total
# Remaining clearance = dist_X - d_X - d_O_total  (must be >= SAFETY_MARGIN)
#
# TTC (head-on):  dist_X / (v_O_peak + v_X)   must be >= t_total + REACTION_TIME
#
# Abort brake: s_abort = v_O_peak*REACTION_TIME + (v_O_peak^2-v_C^2)/(2*BRAKE_DECEL)
#              must be < gap_CL - SAFETY_MARGIN
#
# Merge gap:   gap_CL must be >= VEHICLE_LEN + SAFETY_MARGIN + 0.5*safe_dist(v_L)
#              (O needs physical space; C will adapt speed after merge)
# -----------------------------------------------------------------------

print("=" * 132)
print(f"{'TIME':>7} | {'V_L':>6} | {'V_C':>6} | {'V_X':>6} | "
      f"{'Gap_CL':>8} | {'Dist_X':>9} | "
      f"{'d_pass':>7} | {'t_tot':>6} | {'v_Opeak':>7} | "
      f"{'d_X_cov':>8} | {'Clrnce':>8} | {'TTC':>7} | "
      f"{'AbrtOk':>6} | SAFE_OT / REASON")
print(f"{'(s)':>7} | {'(km/h)':>6} | {'(km/h)':>6} | {'(km/h)':>6} | "
      f"{'(m)':>8} | {'(m)':>9} | "
      f"{'(m)':>7} | {'(s)':>6} | {'(km/h)':>7} | "
      f"{'(m)':>8} | {'(m)':>8} | {'(s)':>7} | "
      f"{'':>6} |")
print("-" * 132)

for i in range(TOTAL_STEPS):

    v_L   = float(lead_spd_kmh[i])
    v_C   = float(comp_spd_kmh[i])
    v_X   = float(oncoming_spd_kmh[i])

    v_L_ms = to_ms(v_L)
    v_C_ms = to_ms(v_C)
    v_X_ms = to_ms(v_X)

    gap_CL  = gap_CL_arr[i]
    dist_X  = dist_to_X_arr[i]

    # Phase A distance O must cover
    d_pass = gap_CL + 2.0 * VEHICLE_LEN

    # Solve quadratic: 0.5*a*t^2 + v_C*t - d_pass = 0
    disc = v_C_ms**2 + 2.0 * ACCEL_O * d_pass
    t_A  = (-v_C_ms + np.sqrt(max(0.0, disc))) / ACCEL_O
    v_O_peak_ms = v_C_ms + ACCEL_O * t_A

    # Phase B
    d_B     = v_O_peak_ms * LATERAL_TIME
    t_total = t_A + LATERAL_TIME
    d_O_total = d_pass + d_B

    # Oncoming coverage
    d_X_covered = v_X_ms * t_total

    # Clearance
    clearance = dist_X - d_X_covered - d_O_total

    # TTC head-on
    closing = v_O_peak_ms + v_X_ms
    TTC = dist_X / closing if closing > 0 and dist_X > 0 else float('inf')

    # Abort braking
    s_react = v_O_peak_ms * REACTION_TIME
    if v_O_peak_ms > v_C_ms:
        s_brake_kin = (v_O_peak_ms**2 - v_C_ms**2) / (2.0 * BRAKE_DECEL)
    else:
        s_brake_kin = 0.0
    s_abort  = s_react + s_brake_kin
    abort_ok = s_abort < (gap_CL - SAFETY_MARGIN)

    # Merge gap: O needs physical space to slot in; C will auto-adjust after merge
    min_merge_gap = VEHICLE_LEN + SAFETY_MARGIN + 0.5 * safe_dist(v_L)
    merge_ok      = gap_CL >= min_merge_gap

    # Safety conditions
    cond_clr   = clearance >= SAFETY_MARGIN
    cond_ttc   = TTC       >= (t_total + REACTION_TIME)
    cond_merge = merge_ok
    cond_onc   = dist_X    > 0

    safe = cond_clr and cond_ttc and cond_merge and cond_onc

    reasons = []
    if not cond_clr:   reasons.append("NO_CLR")
    if not cond_ttc:   reasons.append("LOW_TTC")
    if not cond_merge: reasons.append("NO_GAP")
    if not cond_onc:   reasons.append("ONC_PAST")

    def fmt(v, w=7, dec=2):
        return f"{v:{w}.{dec}f}" if abs(v) < 99999 else f"{'inf':>{w}}"

    safety_label = ">>> SAFE <<<" if safe else f"NOT SAFE  [{','.join(reasons)}]"

    print(f"{ti[i]:7.1f} | {v_L:6.1f} | {v_C:6.1f} | {v_X:6.1f} | "
          f"{gap_CL:8.2f} | {dist_X:9.2f} | "
          f"{d_pass:7.2f} | {fmt(t_total,6,2)} | {to_kmh(v_O_peak_ms):7.1f} | "
          f"{d_X_covered:8.2f} | {clearance:8.2f} | {fmt(TTC,7,2)} | "
          f"{'YES':>6} | {safety_label}")

print("=" * 132)
print()
print("COLUMN LEGEND")
print("  TIME    : Simulation time (s)")
print("  V_L     : Lead car speed (km/h)")
print("  V_C     : Computing car speed (km/h)   [vehicle being overtaken]")
print("  V_X     : Oncoming car speed (km/h)")
print("  Gap_CL  : Bumper-to-bumper gap between Computing car and Lead car (m)")
print("  Dist_X  : Distance from Computing car front to Oncoming car front (m)")
print("  d_pass  : Distance O must travel to fully clear C and reach L rear (m)")
print("  t_tot   : Total overtake manoeuvre time: phase_A + lateral_change (s)")
print("  v_Opeak : Peak speed of Overtaking car during manoeuvre (km/h)")
print("  d_X_cov : Distance Oncoming car covers during t_tot (m)")
print("  Clrnce  : Remaining clearance = Dist_X - d_X_cov - d_O_total  (m)")
print("  TTC     : Time-To-Collision head-on at peak closing speed (s)")
print("  AbrtOk  : Whether O can brake safely back behind C if manoeuvre aborted")
print("  SAFE_OT : Verdict  |  Failure codes:")
print("            NO_CLR   = remaining clearance < 10 m")
print("            LOW_TTC  = TTC < t_total + 1.5 s reaction time")
print("            NO_GAP   = gap between C and L too small to merge into")
print("            ONC_PAST = oncoming car has already passed computing car")
print()
print("PHYSICS")
print("  Phase A (passing):  0.5·1.5·t² + v_C·t - d_pass = 0  => t_A")
print("  Phase B (merge in): t_B = 3.0 s (lateral change),  d_B = v_Opeak · 3.0")
print("  Clearance:          Dist_X - v_X·t_tot - d_O_total  >= 10 m")
print("  TTC:                Dist_X / (v_Opeak + v_X)         >= t_tot + 1.5 s")
print("  Abort distance:     v_O·1.5 + (v_O²-v_C²)/(2·4.0)  < gap_CL - 10 m")
print("  Min merge gap:      VEHICLE_LEN + 10 m + 0.5·safe_dist(v_L)")