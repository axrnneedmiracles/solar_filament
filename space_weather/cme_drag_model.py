"""
CME Hydrodynamic Drag-Based Model (DBM) & Geomagnetic Storm Severity Forecaster
==============================================================================
Solves the analytical hydrodynamic drag kinematics for Coronal Mass Ejections (CMEs)
traveling through the ambient solar wind from the Sun to Earth (1.0 AU).

References:
- Vršnak, B. et al. (2013). Propagation of Coronal Mass Ejections: The Drag-Based Model. Solar Physics, 285(1-2), 295-315.
- Žic, T. et al. (2015). Heliospheric Propagation of Coronal Mass Ejections: Comparison of Analytical and Numerical Solutions.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

AU_KM = 1.495978707e8  # 1 Astronomical Unit in kilometers
DEFAULT_GAMMA_KM = 0.20e-7  # Aerodynamic drag parameter (km^-1)


class CMEDragModel:
    """
    Kinematic solver predicting CME 1-AU Transit Time, Arrival Speed, and Kp Storm Scale.
    """

    def __init__(self, gamma: float = DEFAULT_GAMMA_KM):
        self.gamma = gamma

    def calculate_cme_transit(
        self,
        v0_kms: float = 1200.0,
        v_sw_kms: float = 400.0,
        target_dist_au: float = 1.0,
        start_time_utc: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Solves analytical DBM equation:
        r(t) = v_sw * t + (1 / gamma) * ln(1 + gamma * (v0 - v_sw) * t)
        """
        start_time = start_time_utc or datetime.utcnow()
        target_dist_km = target_dist_au * AU_KM
        gamma = self.gamma
        delta_v = v0_kms - v_sw_kms

        # Fast Newton-Raphson solver to find exact transit time t (seconds)
        t_sec = target_dist_km / max(v0_kms, 300.0)  # Initial guess
        for _ in range(50):
            if abs(delta_v) < 1.0:
                f_val = v_sw_kms * t_sec - target_dist_km
                f_prime = v_sw_kms
            else:
                arg = max(1e-6, 1.0 + gamma * delta_v * t_sec)
                f_val = v_sw_kms * t_sec + (1.0 / gamma) * math.log(arg) - target_dist_km
                f_prime = v_sw_kms + delta_v / arg

            dt = f_val / max(f_prime, 1.0)
            t_sec -= dt
            if abs(dt) < 0.1:
                break

        transit_hours = max(12.0, t_sec / 3600.0)
        arrival_time = start_time + timedelta(seconds=t_sec)

        # Arrival velocity at 1 AU
        if abs(delta_v) < 1.0:
            v_arrival = v_sw_kms
        else:
            arg = max(1e-6, 1.0 + gamma * delta_v * t_sec)
            v_arrival = delta_v / arg + v_sw_kms

        # Estimate Geomagnetic Storm Index (Kp Index 0-9 & NOAA G1-G5 Scale)
        kp_index, storm_scale, storm_severity, power_grid_impact = self._estimate_geomagnetic_storm(v_arrival, v0_kms)

        return {
            'initial_cme_speed_kms': round(v0_kms, 1),
            'solar_wind_speed_kms': round(v_sw_kms, 1),
            'target_distance_au': round(target_dist_au, 2),
            'transit_time_hours': round(transit_hours, 1),
            'transit_time_days': round(transit_hours / 24.0, 2),
            'arrival_speed_kms': round(v_arrival, 1),
            'arrival_time_utc': arrival_time.strftime("%Y-%m-%d %H:%M UTC"),
            'kp_index': kp_index,
            'storm_scale': storm_scale,
            'storm_severity': storm_severity,
            'power_grid_impact': power_grid_impact,
            'drag_parameter_gamma': gamma
        }

    def _estimate_geomagnetic_storm(self, v_arrival: float, v0: float) -> tuple:
        """
        Empirically maps CME arrival velocity & kinetic energy to NOAA Geomagnetic Storm Scales.
        """
        if v_arrival >= 900.0 or v0 >= 2000.0:
            return 9.0, "G5 (Extreme)", "Critical Geomagnetic Superstorm", "Widespread voltage regulation issues; protective grid tripping; satellite surface charging & orbital decay."
        elif v_arrival >= 750.0 or v0 >= 1500.0:
            return 8.0, "G4 (Severe)", "Severe Geomagnetic Storm", "Possible grid voltage alarms; satellite tracking issues; aurora visible down to mid-latitudes."
        elif v_arrival >= 600.0 or v0 >= 1000.0:
            return 7.0, "G3 (Strong)", "Strong Geomagnetic Storm", "Intermittent satellite navigation (GPS) degradation; false alarms on power grid relays."
        elif v_arrival >= 500.0 or v0 >= 700.0:
            return 6.0, "G2 (Moderate)", "Moderate Geomagnetic Storm", "High-latitude power systems may experience voltage fluctuations."
        elif v_arrival >= 430.0:
            return 5.0, "G1 (Minor)", "Minor Geomagnetic Storm", "Weak power grid fluctuations; minor impact on satellite operations."
        else:
            return 3.0, "G0 (Nominal)", "Quiet / Below Storm Threshold", "Standard space environment conditions maintained."


if __name__ == '__main__':
    dbm = CMEDragModel()
    res = dbm.calculate_cme_transit(v0_kms=1400.0, v_sw_kms=420.0)
    print("[*] CME Drag-Based Model Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
