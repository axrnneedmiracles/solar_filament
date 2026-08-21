"""
Space Weather & Parker Spiral Visualizer
========================================
Generates high-resolution publication-quality orbital diagrams of:
1. Parker Spiral Interplanetary Magnetic Field (IMF) lines
2. Solar flare eruption site & magnetic footpoint connection cone
3. Satellite asset positions with real-time radiation exposure coloring
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional

from space_weather.parker_spiral import ParkerSpiralConnectivityEngine, DEFAULT_SATELLITE_FLEET, SOLAR_ROTATION_RAD_S, AU_KM


def generate_parker_spiral_plot(
    flare_lon_deg: float = 68.0,
    flare_lat_deg: float = 18.0,
    flare_class: str = "X2.5",
    v_sw: float = 400.0,
    save_path: Optional[str] = "outputs/space_weather/parker_spiral_connectivity.png"
) -> np.ndarray:
    """
    Renders top-down 2D ecliptic view of Parker Spiral magnetic connectivity and satellite fleet.
    Returns RGB image array suitable for Gradio dashboard.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    engine = ParkerSpiralConnectivityEngine(default_v_sw=v_sw)
    sat_risks = engine.evaluate_satellite_risk(flare_lon_deg, flare_lat_deg, flare_class, v_sw)

    # Dark Astronomical Theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=120, subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0F172A')

    # Grid & Axis setup
    ax.set_theta_zero_location("N")  # 0° pointing North (towards Earth along Sun-Earth line)
    ax.set_theta_direction(-1)       # Clockwise (West is right/clockwise)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0.25 AU', '0.50 AU', '0.75 AU', '1.00 AU (Earth Orbit)'], color='#94A3B8', fontsize=8)
    ax.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)

    # 1. Plot Earth Orbit (1.0 AU Circle)
    theta_orbit = np.linspace(0, 2*np.pi, 200)
    ax.plot(theta_orbit, np.ones_like(theta_orbit)*1.0, color='#38BDF8', linestyle=':', linewidth=1.2, alpha=0.5, label='Earth Orbit (1.0 AU)')

    # 2. Plot Parker Spiral Magnetic Field Lines (Archimedean Spirals)
    # Winding equation: r = (v_sw / Ω) * (φ - φ0)
    a_au = (v_sw * 1e3 / (SOLAR_ROTATION_RAD_S * AU_KM * 1e3))  # a in AU/rad
    
    r_spiral = np.linspace(0.08, 1.25, 150)
    
    # Plot background field lines from 8 sectors
    for phi0_deg in range(0, 360, 45):
        phi0_rad = math.radians(phi0_deg)
        theta_spiral = phi0_rad + (r_spiral / a_au)
        ax.plot(theta_spiral, r_spiral, color='#1E293B', linestyle='-', linewidth=0.8, alpha=0.6)

    # 3. Plot the CONNECTED Field Line from Earth (starts at φ=0 at r=1.0 AU, winds back to ~W60°)
    earth_footpoint_deg = engine.calculate_nominal_footpoint(1.0, 0.0, v_sw)
    earth_footpoint_rad = math.radians(earth_footpoint_deg)
    
    theta_earth_spiral = earth_footpoint_rad - (r_spiral / a_au)
    ax.plot(theta_earth_spiral, r_spiral, color='#38BDF8', linestyle='-', linewidth=2.5, alpha=0.9, label=f'Earth Connected Field Line (Footpoint: W{earth_footpoint_deg:.0f}°)')

    # 4. Highlight the High-Connectivity Magnetic Cone (±20° around Earth footpoint)
    cone_min_rad = math.radians(earth_footpoint_deg - 22.0)
    cone_max_rad = math.radians(earth_footpoint_deg + 22.0)
    r_cone = np.linspace(0.08, 1.20, 50)
    theta_cone_1 = cone_min_rad - (r_cone / a_au)
    theta_cone_2 = cone_max_rad - (r_cone / a_au)
    ax.fill_between(r_cone, theta_cone_1, theta_cone_2, color='#EA580C', alpha=0.15, label='High-Risk SEP Connection Zone (W40°–W85°)')

    # 5. Plot the Solar Flare / Filament Eruption Site
    flare_rad = math.radians(flare_lon_deg)
    ax.scatter([flare_rad], [0.08], color='#EF4444', s=220, marker='*', edgecolor='#FEF08A', linewidth=1.5, zorder=10, label=f'Active Flare: {flare_class} at W{flare_lon_deg:.0f}°')

    # Draw Spiral Field Line EMANATING FROM FLARE SITE
    theta_flare_spiral = flare_rad - (r_spiral / a_au)
    flare_line_color = '#EF4444' if abs(flare_lon_deg - earth_footpoint_deg) <= 30.0 else '#F59E0B'
    ax.plot(theta_flare_spiral, r_spiral, color=flare_line_color, linestyle='--', linewidth=2.0, alpha=0.85, label='Flare Magnetic Emission Path')

    # 6. Plot the Sun at Center
    ax.scatter([0], [0], color='#FBBF24', s=350, marker='o', edgecolor='#F59E0B', linewidth=2, zorder=12, label='The Sun (Origin)')

    # 7. Plot Satellite Fleet Assets (Grouped with orbital offsets for clean visual hierarchy)
    plotted_positions = {}
    
    # Priority order: Show high-risk and distinct orbital representatives first
    for sat in sat_risks:
        sat_name = sat['satellite_name']
        asset = next((a for a in DEFAULT_SATELLITE_FLEET if a.name == sat_name), None)
        if not asset:
            continue
            
        sat_lon_rad = math.radians(asset.heliolongitude_deg)
        sat_r = asset.heliocentric_dist_au
        sat_color = sat['risk_color']
        
        # Slight radial jitter for Earth-orbit assets so they don't overlap into a single dot
        cat = getattr(asset, 'orbit_category', asset.orbit_type)
        if "L1" in cat:
            r_plot = sat_r - 0.02
        elif "L2" in cat:
            r_plot = sat_r + 0.03
        elif "MEO" in cat:
            r_plot = sat_r - 0.04
        elif "LEO" in cat or "Human" in cat:
            r_plot = sat_r - 0.01
        elif "Lunar" in cat:
            r_plot = sat_r + 0.015
        else:
            r_plot = sat_r

        # Marker style based on category
        marker = 's' if "GEO" in cat else ('^' if "L1" in cat or "L2" in cat else ('D' if "Navigation" in cat else 'o'))
        ax.scatter([sat_lon_rad], [r_plot], color=sat_color, s=90, marker=marker, edgecolor='#FFFFFF', linewidth=1.0, zorder=15)

    # Annotate Top Representative Asset Classes
    key_annotations = [
        ("L1 Sentinels (DSCOVR/SOHO/Aditya)", 0.0, 0.965, "#38BDF8"),
        ("GEO Weather (GOES-16/18/Meteosat)", 0.0, 1.045, "#FBBF24"),
        ("LEO & Crewed (ISS/Starlink/OneWeb)", 0.0, 0.935, "#34D399"),
        ("GNSS Nav Fleet (GPS III/Galileo)", 0.0, 0.905, "#A78BFA"),
        ("L2 Deep Space (JWST/Euclid)", 0.0, 1.075, "#F472B6"),
        ("Lunar Gateway / Artemis", 0.0, 1.015, "#FB923C"),
    ]
    for label, lon_d, r_a, col in key_annotations:
        lon_r = math.radians(lon_d)
        ax.annotate(
            label,
            xy=(lon_r, r_a),
            xytext=(lon_r + 0.12, r_a + 0.025),
            color='#E2E8F0',
            fontsize=7.2,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#0B0F19", ec=col, lw=1.0, alpha=0.90)
        )

    # Plot Deep Space Spacecraft specifically
    for d_name, d_lon, d_r, d_col in [("STEREO-A", 25.0, 0.96, "#38BDF8"), ("Solar Orbiter", -35.0, 0.55, "#F59E0B"), ("Parker Solar Probe", 45.0, 0.15, "#EF4444")]:
        d_rad = math.radians(d_lon)
        ax.scatter([d_rad], [d_r], color=d_col, s=120, marker='*', edgecolor='#FFFFFF', linewidth=1.2, zorder=16)
        ax.annotate(
            f"{d_name} ({d_r} AU)",
            xy=(d_rad, d_r),
            xytext=(d_rad + 0.10, d_r + 0.04),
            color=d_col,
            fontsize=7.5,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#0B0F19", ec=d_col, lw=0.8, alpha=0.85)
        )

    # Title & Legend
    delta_e = abs(flare_lon_deg - earth_footpoint_deg)
    status_text = "CRITICAL / DIRECT CONNECTION" if delta_e <= 25 else ("ELEVATED RISK" if delta_e <= 45 else "POORLY CONNECTED")
    
    ax.set_title(
        f"PARKER SPIRAL MAGNETIC CONNECTIVITY & SATELLITE SEP RADIATION PATH\n"
        f"Flare: {flare_class} at W{flare_lon_deg:.1f}° | Earth Footpoint: W{earth_footpoint_deg:.1f}° (Δφ = {delta_e:.1f}°) | Status: {status_text}",
        color='#F8FAFC',
        fontsize=9.5,
        fontweight='bold',
        pad=18
    )

    legend = ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.18), fontsize=7.5, facecolor='#1E293B', edgecolor='#475569', ncol=2)
    plt.setp(legend.get_texts(), color='#E2E8F0')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=120)

    # Convert plot to RGB numpy array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)

    return rgb


if __name__ == '__main__':
    img = generate_parker_spiral_plot(flare_lon_deg=68.0, flare_lat_deg=18.0, flare_class="X2.5")
    print(f"[+] Successfully generated Parker Spiral Connectivity plot! Shape: {img.shape}")
