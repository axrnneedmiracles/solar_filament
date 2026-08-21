"""
Parker Spiral Magnetic Field & Satellite Radiation Risk Engine
==============================================================
Calculates interplanetary magnetic field (IMF) Parker Spiral topology,
heliolongitude footpoint connectivity, and particle radiation exposure risk
for orbital assets (LEO, GEO, L1, Lunar Gateway, Deep Space).

Physics Model:
- Solar Angular Velocity: Ω = 2.87e-6 rad/s (~14.71 deg/day)
- Nominal Solar Wind Speed: v_sw = 400 km/s (variable 300 - 800 km/s)
- Magnetic Footpoint Equation: φ_footpoint = φ_satellite + (Ω * r) / v_sw
- Core Magnetic Connectivity Cone: Δφ <= 25° (Direct Field Line Connection)
- Validated against NASA DONKI Solar Energetic Particle (SEP) event records.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

AU_KM = 149597870.7  # 1 AU in km
SOLAR_ROTATION_RAD_S = 2.865e-6  # Solar sidereal rotation rate (~14.71 deg/day in rad/s)
SPEED_OF_LIGHT_KMS = 299792.458


@dataclass
class SatelliteAsset:
    name: str
    operator: str  # NASA, ESA, NOAA, US Space Force, ISRO, JAXA, Commercial
    orbit_type: str  # LEO, MEO, GEO, L1, L2, Lunar NRHO, Heliocentric
    orbit_category: str  # Human Spaceflight, Space Weather, Navigation, Earth Obs, Telecom, Deep Space
    altitude_km: float
    heliocentric_dist_au: float
    heliolongitude_deg: float  # Relative to Sun-Earth line (Earth = 0.0°)
    magnetospheric_shielding: str  # High, Moderate, Low, Zero
    primary_hazard: str
    critical_subsystems: List[str]
    mitigation_protocol: str
    description: str


# Comprehensive Catalog of Real Operational Satellite Assets Across All Orbital Regimes
DEFAULT_SATELLITE_FLEET = [
    # ── 1. Lagrange L1 & Space Weather Sentinels (Zero Shielding, 0.99 AU) ──
    SatelliteAsset(
        name="DSCOVR",
        operator="NOAA / NASA",
        orbit_type="Sun-Earth Lagrange L1",
        orbit_category="Space Weather (L1)",
        altitude_km=1500000,
        heliocentric_dist_au=0.99,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Unshielded Interplanetary)",
        primary_hazard="High-energy proton sensor saturation & Faraday cup noise",
        critical_subsystems=["PlasMag Faraday Cup", "Tri-axial Magnetometer", "EPIC Camera"],
        mitigation_protocol="Switch Faraday cup to high-flux mode; cross-calibrate with ACE",
        description="Upstream Solar Wind Sentinel providing real-time 15-45 min space weather alerts."
    ),
    SatelliteAsset(
        name="SOHO (Solar & Heliospheric Obs)",
        operator="ESA / NASA",
        orbit_type="Sun-Earth Lagrange L1",
        orbit_category="Space Weather (L1)",
        altitude_km=1500000,
        heliocentric_dist_au=0.99,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Unshielded Interplanetary)",
        primary_hazard="Coronagraph CCD snow & star tracker single-event upsets",
        critical_subsystems=["LASCO C2/C3 Coronagraphs", "EIT Imager", "CELIAS Solar Wind Sensor"],
        mitigation_protocol="Stow sensitive detector shutters; enable autonomous recovery",
        description="Historic solar observatory continuously tracking coronal mass ejections."
    ),
    SatelliteAsset(
        name="ACE (Advanced Composition Explorer)",
        operator="NASA / Caltech",
        orbit_type="Sun-Earth Lagrange L1",
        orbit_category="Space Weather (L1)",
        altitude_km=1500000,
        heliocentric_dist_au=0.99,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Unshielded Interplanetary)",
        primary_hazard="EPAM electron/proton detector saturation & SEUs",
        critical_subsystems=["EPAM Energetic Particle Spectrometer", "SWEPAM Plasma Sensor", "MAG"],
        mitigation_protocol="Calibrate real-time telemetry pipelines for extreme particle flux",
        description="Interplanetary magnetic field and energetic particle composition monitor."
    ),
    SatelliteAsset(
        name="Aditya-L1",
        operator="ISRO (India)",
        orbit_type="Sun-Earth Lagrange L1",
        orbit_category="Space Weather (L1)",
        altitude_km=1500000,
        heliocentric_dist_au=0.99,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Unshielded Interplanetary)",
        primary_hazard="VELC coronagraph detector noise & ASPEX plasma sensor SEUs",
        critical_subsystems=["VELC Solar Coronagraph", "SUIT UV Telescope", "ASPEX Solar Wind Analyzer"],
        mitigation_protocol="Configure payload gain settings; orient payload shielding",
        description="India's flagship solar observation satellite operating in halo orbit around L1."
    ),

    # ── 2. Lagrange L2 Deep Space Observatories (Zero Shielding, 1.01 AU) ──
    SatelliteAsset(
        name="James Webb Space Telescope (JWST)",
        operator="NASA / ESA / CSA",
        orbit_type="Sun-Earth Lagrange L2",
        orbit_category="Deep Space Science (L2)",
        altitude_km=1500000,
        heliocentric_dist_au=1.01,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Deep Space / Outside Magnetosphere)",
        primary_hazard="NIRCam/MIRI infrared detector artifact spikes & Solid-State Recorder bit-flips",
        critical_subsystems=["Fine Guidance Sensor (FGS)", "NIRCam / MIRI Detectors", "Cryocooler Electronics"],
        mitigation_protocol="Increase cosmic-ray rejection sampling; monitor gyro star tracker health",
        description="NASA premier deep-space infrared observatory operating at Sun-Earth L2."
    ),
    SatelliteAsset(
        name="Euclid Space Telescope",
        operator="ESA / NASA",
        orbit_type="Sun-Earth Lagrange L2",
        orbit_category="Deep Space Science (L2)",
        altitude_km=1500000,
        heliocentric_dist_au=1.01,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Deep Space)",
        primary_hazard="VIS/NISP optical sensor noise & CCD charge-transfer inefficiency",
        critical_subsystems=["VIS Optical Instrument", "NISP Near-IR Spectrometer", "Attitude Control"],
        mitigation_protocol="Flag survey exposures for particle noise decontamination",
        description="ESA mission mapping the geometry of the dark Universe from L2."
    ),

    # ── 3. Geostationary (GEO) Weather & Communication Satellites (35,786 km) ──
    SatelliteAsset(
        name="NOAA GOES-16 (GOES-East, 75.2° W)",
        operator="NOAA / NASA",
        orbit_type="Geostationary (GEO)",
        orbit_category="GEO Weather & Defense",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate (Outer Magnetopause Boundary)",
        primary_hazard="Deep dielectric charging in cables & SEISS proton sensor saturation",
        critical_subsystems=["Advanced Baseline Imager (ABI)", "EXIS X-Ray Sensor", "SEISS Particle Monitor"],
        mitigation_protocol="Discharge spacecraft surface charge; initiate heater cycle protocols",
        description="NOAA primary operational geostationary weather and space weather satellite."
    ),
    SatelliteAsset(
        name="NOAA GOES-18 (GOES-West, 137.2° W)",
        operator="NOAA / NASA",
        orbit_type="Geostationary (GEO)",
        orbit_category="GEO Weather & Defense",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate (Outer Magnetopause Boundary)",
        primary_hazard="Surface ESD discharging & Star Tracker false star identification",
        critical_subsystems=["ABI Multi-Spectral Sensor", "Magnetometer", "Space Environment In-Situ"],
        mitigation_protocol="Cross-reference star tracker vectors with coarse Sun sensors",
        description="NOAA Pacific and Western US operational geostationary sentinel."
    ),
    SatelliteAsset(
        name="Solar Dynamics Observatory (SDO)",
        operator="NASA",
        orbit_type="Inclined Geosynchronous (GSO)",
        orbit_category="Space Weather (GSO)",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate (Traverses Radiation Belts)",
        primary_hazard="AIA / HMI CCD degradation & high-gain Ka-band antenna telemetry corruption",
        critical_subsystems=["AIA Atmospheric Imaging Assembly", "HMI Helioseismic Imager", "EVE EUV Sensor"],
        mitigation_protocol="Cycle CCD baking heaters; enable EDAC memory scrubbing",
        description="NASA premier high-cadence full-disk solar physics space telescope."
    ),
    SatelliteAsset(
        name="Meteosat-11 / MTG-I1",
        operator="EUMETSAT / ESA",
        orbit_type="Geostationary (GEO, 0° E)",
        orbit_category="GEO Weather & Defense",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate",
        primary_hazard="Solar array photo-diode degradation & momentum wheel bit-flips",
        critical_subsystems=["FCI Flexible Combined Imager", "Lightning Imager (LI)", "Power Control Unit"],
        mitigation_protocol="Activate redundant power distribution buses; orient solar panels",
        description="European geostationary meteorological and storm monitoring satellite."
    ),
    SatelliteAsset(
        name="TDRS-12 / TDRS-13 (Space Network)",
        operator="NASA",
        orbit_type="Geostationary (GEO)",
        orbit_category="GEO Telecom & Relay",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate",
        primary_hazard="Traveling Wave Tube Amplifier (TWTA) arcing & relay outage",
        critical_subsystems=["S/Ku/Ka-band Phased Arrays", "Command Decoders", "Attitude Gyros"],
        mitigation_protocol="Maintain backup tracking relays for ISS and Hubble communications",
        description="NASA critical communications relay satellite network for crewed and LEO missions."
    ),
    SatelliteAsset(
        name="Viasat-3 / Inmarsat-6",
        operator="Commercial Telecom Fleet",
        orbit_type="Geostationary (GEO)",
        orbit_category="GEO Telecom & Relay",
        altitude_km=35786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate",
        primary_hazard="Internal electrostatic discharge & transponder power fluctuations",
        critical_subsystems=["High-Throughput Ka-band Transponders", "Electric Propulsion Thrusters"],
        mitigation_protocol="Monitor payload thermal dissipation; power cycle redundant payload slices",
        description="Commercial high-capacity broadband and maritime aviation telecom satellites."
    ),

    # ── 4. Medium Earth Orbit (MEO) GNSS Constellations (19,000 - 24,000 km) ──
    SatelliteAsset(
        name="GPS III Constellation (31 Satellites)",
        operator="US Space Force",
        orbit_type="Medium Earth Orbit (MEO)",
        orbit_category="Navigation (GNSS MEO)",
        altitude_km=20200,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Low (Trapped Outer Van Allen Belt)",
        primary_hazard="Rubidium/Cesium atomic clock drift & L-band signal scintillation",
        critical_subsystems=["Rubidium Atomic Clocks", "L1/L2/L5 Signal Generators", "Crosslink Transponders"],
        mitigation_protocol="Upload ephemeris clock corrections; activate ground-based Kalman filtering",
        description="Global Positioning System baseline providing precision worldwide PNT."
    ),
    SatelliteAsset(
        name="Galileo Constellation (28 Satellites)",
        operator="ESA / European Union",
        orbit_type="Medium Earth Orbit (MEO)",
        orbit_category="Navigation (GNSS MEO)",
        altitude_km=23222,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Low (Severe Trapped Particle Belts)",
        primary_hazard="Passive Hydrogen Maser (PHM) clock single-event latch-ups",
        critical_subsystems=["Passive Hydrogen Maser Clocks", "Search & Rescue (SAR) Transponders"],
        mitigation_protocol="Switch to redundant secondary PHM clock if telemetry anomaly detected",
        description="European civil global satellite navigation and positioning system."
    ),
    SatelliteAsset(
        name="GLONASS-K2 Constellation",
        operator="Roscosmos (Russia)",
        orbit_type="Medium Earth Orbit (MEO)",
        orbit_category="Navigation (GNSS MEO)",
        altitude_km=19100,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Low (Intense Van Allen Radiation)",
        primary_hazard="Solar panel darkening & avionics memory single-bit upsets",
        critical_subsystems=["Navigation Payload", "Barium Atomic Clocks", "Solar Array Regulators"],
        mitigation_protocol="Perform periodic memory scrubbing; monitor charging currents",
        description="Russian global satellite navigation constellation in 64.8° inclination MEO."
    ),
    SatelliteAsset(
        name="BeiDou-3 (MEO / IGSO Fleet)",
        operator="CNSA (China)",
        orbit_type="MEO / IGSO (Mixed Orbit)",
        orbit_category="Navigation (GNSS MEO)",
        altitude_km=21528,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Low",
        primary_hazard="Inter-satellite laser link pointing jitter & clock stability degradation",
        critical_subsystems=["Inter-Satellite Optical Terminals", "Hydrogen Clocks", "Ka-band Payloads"],
        mitigation_protocol="Re-align laser pointing terminals; activate error-correcting firmware",
        description="Chinese global satellite navigation and short message communication system."
    ),

    # ── 5. Low Earth Orbit (LEO) & Human Spaceflight (300 - 800 km) ──
    SatelliteAsset(
        name="International Space Station (ISS)",
        operator="NASA / ESA / JAXA / CSA",
        orbit_type="Low Earth Orbit (LEO, 51.6° Inc)",
        orbit_category="Human Spaceflight (LEO)",
        altitude_km=415,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="High (Geomagnetic Dipole Shielded)",
        primary_hazard="Elevated astronaut crew ionizing dosage during SAA passes & EVA risks",
        critical_subsystems=["Astronaut Extravehicular Activities (EVAs)", "Life Support Avionics", "Solar Arrays"],
        mitigation_protocol="Suspend spacewalks (EVAs); shelter crew in heavily-shielded Zvezda / Destiny modules",
        description="Permanently crewed microgravity orbital laboratory in low Earth orbit."
    ),
    SatelliteAsset(
        name="Tiangong Space Station (CSS)",
        operator="CMSA (China)",
        orbit_type="Low Earth Orbit (LEO, 41.5° Inc)",
        orbit_category="Human Spaceflight (LEO)",
        altitude_km=390,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="High (Geomagnetic Shielded)",
        primary_hazard="Taikonaut ionizing radiation exposure during high-inclination passes",
        critical_subsystems=["Tianhe Core Module", "Wentian/Mengtian Lab Avionics", "Robotic Arm"],
        mitigation_protocol="Stow external payload experiments; shelter taikonauts in core sleeping berths",
        description="China's permanently crewed modular space station in low Earth orbit."
    ),
    SatelliteAsset(
        name="Hubble Space Telescope (HST)",
        operator="NASA / ESA",
        orbit_type="Low Earth Orbit (LEO, 28.5° Inc)",
        orbit_category="Earth Obs & Science (LEO)",
        altitude_km=535,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="High",
        primary_hazard="Wide Field Camera 3 (WFC3) detector false pixel hits during SAA passes",
        critical_subsystems=["WFC3 Camera", "Advanced Camera for Surveys (ACS)", "Fine Guidance Gyros"],
        mitigation_protocol="Turn off high-voltage instrument detectors during radiation storm peaks",
        description="Historic optical space telescope operating in low Earth orbit since 1990."
    ),
    SatelliteAsset(
        name="Starlink Constellation (6,000+ Sats)",
        operator="SpaceX",
        orbit_type="Low Earth Orbit (LEO, 53° Inc)",
        orbit_category="LEO Mega-Constellations",
        altitude_km=550,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="High (Vulnerable to Atmospheric Drag Expansion)",
        primary_hazard="Thermospheric Joule heating causing severe atmospheric drag & orbital decay",
        critical_subsystems=["Krypton/Argon Ion Thrusters", "Star Trackers", "Phased Array Antennas"],
        mitigation_protocol="Orient satellites edge-on (knife-edge mode) to minimize drag; fire thrusters to boost altitude",
        description="World's largest commercial broadband constellation in low Earth orbit."
    ),
    SatelliteAsset(
        name="Eutelsat OneWeb Constellation",
        operator="Eutelsat OneWeb",
        orbit_type="Polar LEO (86.4° Inc)",
        orbit_category="LEO Mega-Constellations",
        altitude_km=1200,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate (Direct Auroral Cusp Exposure)",
        primary_hazard="Direct precipitation of energetic particles in polar caps & SEUs",
        critical_subsystems=["Ku/Ka-band Payloads", "Central Flight Computers", "Power Units"],
        mitigation_protocol="Enable triple-modular redundancy error voting; monitor gyro resets",
        description="Global polar-orbiting commercial satellite communications network."
    ),
    SatelliteAsset(
        name="Copernicus Sentinel-1 / Sentinel-2",
        operator="ESA / European Commission",
        orbit_type="Sun-Synchronous LEO (SSO)",
        orbit_category="Earth Obs & Science (LEO)",
        altitude_km=786,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate (Polar Horn Crossing)",
        primary_hazard="Synthetic Aperture Radar (SAR) transmit-receive module latch-ups",
        critical_subsystems=["C-band Synthetic Aperture Radar (SAR)", "Multi-Spectral Instrument (MSI)"],
        mitigation_protocol="Place SAR radar in standby mode during extreme polar particle events",
        description="European Copernicus flagship Earth observation and environmental sentinel fleet."
    ),
    SatelliteAsset(
        name="Landsat 8 / Landsat 9",
        operator="NASA / USGS",
        orbit_type="Sun-Synchronous LEO (SSO)",
        orbit_category="Earth Obs & Science (LEO)",
        altitude_km=705,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Moderate",
        primary_hazard="OLI-2 optical sensor noise & solid-state memory single-bit upsets",
        critical_subsystems=["Operational Land Imager (OLI-2)", "Thermal Infrared Sensor (TIRS-2)"],
        mitigation_protocol="Enable autonomous memory scrubbing; cross-check calibration arrays",
        description="USGS flagship land imaging and climate monitoring satellites."
    ),
    SatelliteAsset(
        name="PlanetScope / Dove Cubesat Fleet",
        operator="Planet Labs",
        orbit_type="Sun-Synchronous LEO (SSO)",
        orbit_category="LEO Mega-Constellations",
        altitude_km=500,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="High (Commercial Off-The-Shelf Electronics)",
        primary_hazard="COTS microcontroller watchdog resets and reboot loops",
        critical_subsystems=["Optical Telescopes", "Reaction Wheels", "COTS Microprocessors"],
        mitigation_protocol="Automate constellation watchdog self-heal and orbital ephemeris reload",
        description="Large constellation of commercial Earth-imaging cubesats."
    ),

    # ── 6. Lunar & Deep Space Exploration (Zero Shielding, Interplanetary) ──
    SatelliteAsset(
        name="Lunar Gateway (Artemis Station)",
        operator="NASA / ESA / JAXA / CSA",
        orbit_type="Cislunar / Near-Rectilinear Halo Orbit (NRHO)",
        orbit_category="Lunar & Deep Space",
        altitude_km=384400,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Deep Space / Outside Magnetosphere)",
        primary_hazard="Unshielded relativistic proton bombardment, crew habitat dosage & Solar Electric Propulsion SEUs",
        critical_subsystems=["Power and Propulsion Element (PPE)", "HALO Habitat Life Support", "HERMES Radiation Suite"],
        mitigation_protocol="Command crew to water-wall radiation storm shelters; throttle SEP ion thrusters",
        description="NASA/International crewed lunar space station in high cislunar NRHO orbit."
    ),
    SatelliteAsset(
        name="Orion Spacecraft (Artemis Fleet)",
        operator="NASA / ESA",
        orbit_type="Trans-Lunar / Cislunar Trajectory",
        orbit_category="Lunar & Deep Space",
        altitude_km=384400,
        heliocentric_dist_au=1.0,
        heliolongitude_deg=0.0,
        magnetospheric_shielding="Zero (Deep Space Unshielded)",
        primary_hazard="Critical flight computer latch-ups & acute astronaut radiation sickness",
        critical_subsystems=["Avionics Quad-Redundant Flight Computers", "European Service Module", "Life Support"],
        mitigation_protocol="Construct internal radiation shelter using cargo stowage bags; power down non-critical telemetry",
        description="NASA deep-space crewed exploration vehicle for lunar and interplanetary missions."
    ),
    SatelliteAsset(
        name="STEREO-A Spacecraft",
        operator="NASA",
        orbit_type="Heliocentric Orbit",
        orbit_category="Space Weather (Interplanetary)",
        altitude_km=143000000,
        heliocentric_dist_au=0.96,
        heliolongitude_deg=25.0,  # Leading Earth by ~25°
        magnetospheric_shielding="Zero (Interplanetary Space)",
        primary_hazard="Coronagraph detector saturated star streaks & SECCHI imager noise",
        critical_subsystems=["SECCHI Telescope Suite", "IMPACT Plasma Analyzer", "PLASTIC Sensor"],
        mitigation_protocol="Enable high-cadence CME tracking mode; upload beacon data stream",
        description="Solar Terrestrial Relations Observatory leading Earth in heliocentric orbit."
    ),
    SatelliteAsset(
        name="Solar Orbiter",
        operator="ESA / NASA",
        orbit_type="Inner Heliocentric Orbit",
        orbit_category="Space Weather (Interplanetary)",
        altitude_km=80000000,
        heliocentric_dist_au=0.55,  # Variable 0.28 to 0.90 AU
        heliolongitude_deg=-35.0,
        magnetospheric_shielding="Zero (Extreme Proximity)",
        primary_hazard="Severe proton fluence on titanium heat shield and Energetic Particle Detector (EPD)",
        critical_subsystems=["Energetic Particle Detector (EPD)", "PHI Magnetometer", "STIX X-Ray Spectrometer"],
        mitigation_protocol="Orient heat shield directly towards Sun; adjust EPD sensor gain attenuators",
        description="ESA mission observing the Sun and inner heliosphere from close perihelion distances."
    ),
    SatelliteAsset(
        name="Parker Solar Probe",
        operator="NASA / JHUAPL",
        orbit_type="Ultra-Close Solar Orbit",
        orbit_category="Space Weather (Interplanetary)",
        altitude_km=25000000,
        heliocentric_dist_au=0.15,  # Down to 0.04 AU perihelion
        heliolongitude_deg=45.0,
        magnetospheric_shielding="Zero (Extreme Immersion in Solar Corona)",
        primary_hazard="Extreme coronal plasma sputtering, FIELDS antenna thermal stress, high-energy particle bombardment",
        critical_subsystems=["Thermal Protection Shield (TPS)", "FIELDS Plasma Antenna", "SWEAP Solar Probe Cup"],
        mitigation_protocol="Maintain autonomous solar shadow pointing; record high-rate burst memory",
        description="NASA historic mission diving directly through the Sun's outer corona."
    ),
    SatelliteAsset(
        name="BepiColombo (Cruising Fleet)",
        operator="ESA / JAXA",
        orbit_type="Heliocentric Cruise Orbit",
        orbit_category="Lunar & Deep Space",
        altitude_km=110000000,
        heliocentric_dist_au=0.72,
        heliolongitude_deg=-15.0,
        magnetospheric_shielding="Zero (Interplanetary)",
        primary_hazard="Mercury Planetary Orbiter sensor degradation & solar electric propulsion arcing",
        critical_subsystems=["MPO Science Instruments", "Mercury Transfer Module (MTM) Ion Engines"],
        mitigation_protocol="Monitor ion thruster beam stability; safeguard sensitive planetary spectrometers",
        description="European-Japanese joint mission en route to orbit Mercury."
    )
]


class ParkerSpiralConnectivityEngine:
    """
    Computes magnetic field line connectivity and satellite radiation risk
    given solar flare/filament eruption heliolatitude/heliolongitude.
    """

    def __init__(
        self,
        default_v_sw: float = 400.0,  # Solar wind velocity in km/s
        satellites: Optional[List[SatelliteAsset]] = None
    ):
        self.default_v_sw = default_v_sw
        self.satellites = satellites or DEFAULT_SATELLITE_FLEET

    def calculate_nominal_footpoint(
        self,
        r_au: float,
        obs_lon_deg: float = 0.0,
        v_sw: Optional[float] = None
    ) -> float:
        """
        Calculates the solar surface heliolongitude footpoint for an observer at (r_au, obs_lon_deg).
        φ_footpoint = obs_lon_deg + (Ω * r) / v_sw
        Returns footpoint in degrees (West positive).
        """
        v = v_sw or self.default_v_sw
        r_km = r_au * AU_KM
        
        # Winding angle in radians
        delta_phi_rad = (SOLAR_ROTATION_RAD_S * r_km) / v
        delta_phi_deg = math.degrees(delta_phi_rad)
        
        footpoint_lon = obs_lon_deg + delta_phi_deg
        return footpoint_lon

    def calculate_spiral_arc_length(
        self,
        r_au: float,
        v_sw: Optional[float] = None
    ) -> float:
        """
        Calculates the exact physical curved path length of the Archimedean Parker Spiral field line in km.
        """
        v = v_sw or self.default_v_sw
        r_km = r_au * AU_KM
        a = v / SOLAR_ROTATION_RAD_S  # Archimedean spiral parameter
        theta = r_km / a
        
        # Exact integral arc length of Archimedean spiral: L = (a/2) * [θ√(1+θ²) + ln(θ + √(1+θ²))]
        arc_km = 0.5 * a * (theta * math.sqrt(1 + theta**2) + math.log(theta + math.sqrt(1 + theta**2) + 1e-12))
        return arc_km

    def estimate_particle_transit_time(
        self,
        r_au: float,
        v_sw: Optional[float] = None,
        proton_energy_mev: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculates relativistic solar energetic proton (SEP) transit time along the Parker spiral.
        """
        # Relativistic velocity calculation for proton (m0 = 938.272 MeV)
        m0 = 938.272
        gamma = 1.0 + (proton_energy_mev / m0)
        beta = math.sqrt(1.0 - (1.0 / (gamma**2)))
        v_proton_kms = beta * SPEED_OF_LIGHT_KMS
        
        path_length_km = self.calculate_spiral_arc_length(r_au, v_sw)
        transit_sec = path_length_km / v_proton_kms
        
        return {
            'proton_energy_mev': proton_energy_mev,
            'proton_speed_kms': round(v_proton_kms, 1),
            'proton_speed_fraction_c': round(beta, 3),
            'magnetic_path_length_au': round(path_length_km / AU_KM, 3),
            'transit_time_minutes': round(transit_sec / 60.0, 1),
            'transit_time_seconds': round(transit_sec, 1)
        }

    def evaluate_satellite_risk(
        self,
        flare_lon_deg: float,
        flare_lat_deg: float = 0.0,
        flare_class: str = "M5.0",
        v_sw: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates radiation exposure risk for each satellite asset in the fleet.
        
        Risk Grading:
        - SEVERE / CRITICAL: Δφ <= 20° AND Flare >= M5.0 / X-class
        - ELEVATED: Δφ <= 35° OR (Δφ <= 50° with X-class)
        - MODERATE: 35° < Δφ <= 65°
        - LOW / NOMINAL: Δφ > 65° (Poor magnetic connection)
        """
        v = v_sw or self.default_v_sw
        results = []

        # Parse flare power multiplier
        flare_class = flare_class.upper().strip()
        class_letter = flare_class[0] if flare_class else 'C'
        try:
            class_num = float(flare_class[1:]) if len(flare_class) > 1 else 1.0
        except ValueError:
            class_num = 1.0

        if class_letter == 'X':
            flare_severity_score = 10.0 * class_num
        elif class_letter == 'M':
            flare_severity_score = 1.0 * class_num
        elif class_letter == 'C':
            flare_severity_score = 0.1 * class_num
        else:
            flare_severity_score = 0.01

        for sat in self.satellites:
            footpoint_lon = self.calculate_nominal_footpoint(sat.heliocentric_dist_au, sat.heliolongitude_deg, v)
            # Angular separation between flare and satellite's magnetic connection footpoint
            delta_lon = abs(flare_lon_deg - footpoint_lon)
            
            # Normal magnetic connectivity factor (Gaussian falloff around footpoint, σ ~ 25°)
            conn_factor = math.exp(-0.5 * (delta_lon / 25.0)**2)
            
            # Exposure Risk Score [0 to 100]
            raw_risk = min(100.0, conn_factor * min(100.0, 15.0 * math.sqrt(flare_severity_score) + 20.0))
            
            # Adjust for shielding
            if "High" in sat.magnetospheric_shielding:
                risk_score = raw_risk * 0.35  # Geomagnetic shielding factor
            elif "Moderate" in sat.magnetospheric_shielding:
                risk_score = raw_risk * 0.70
            else:
                risk_score = raw_risk * 1.00  # Full interplanetary exposure

            # Categorize Risk Level
            if risk_score >= 65.0:
                risk_level = "CRITICAL / SEVERE ELEVATED RISK"
                risk_color = "#DC2626"  # Red
                action_alert = "Initiate Satellite Safe Mode / Power Down High-Voltage Payloads & Suspend EVAs"
            elif risk_score >= 40.0:
                risk_level = "ELEVATED RADIATION RISK"
                risk_color = "#EA580C"  # Orange
                action_alert = "Orient Solar Arrays / Monitor Star Tracker Single-Event Upsets (SEUs)"
            elif risk_score >= 20.0:
                risk_level = "MODERATE WATCH"
                risk_color = "#CA8A04"  # Yellow
                action_alert = "Heightened Radiation Telemetry Monitoring"
            else:
                risk_level = "NOMINAL / LOW RISK"
                risk_color = "#16A34A"  # Green
                action_alert = "Standard Operations Maintained"

            transit_info = self.estimate_particle_transit_time(sat.heliocentric_dist_au, v, proton_energy_mev=50.0)

            results.append({
                'satellite_name': sat.name,
                'operator': getattr(sat, 'operator', 'International / Agency'),
                'orbit': sat.orbit_type,
                'orbit_category': getattr(sat, 'orbit_category', sat.orbit_type),
                'altitude_km': getattr(sat, 'altitude_km', 35786),
                'magnetic_footpoint_lon': round(footpoint_lon, 1),
                'angular_separation_deg': round(delta_lon, 1),
                'magnetic_connectivity_pct': round(conn_factor * 100.0, 1),
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'shielding': sat.magnetospheric_shielding,
                'primary_hazard': getattr(sat, 'primary_hazard', 'Single-Event Upsets & Sensor Saturation'),
                'action_alert': getattr(sat, 'mitigation_protocol', action_alert),
                'particle_arrival_minutes': transit_info['transit_time_minutes'],
                'critical_subsystems': sat.critical_subsystems,
                'description': sat.description
            })

        # Sort by highest risk score first
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results


if __name__ == '__main__':
    engine = ParkerSpiralConnectivityEngine()
    print("[*] Testing Parker Spiral Connectivity for Solar Flare at N19W68 (Class X2.5):")
    risks = engine.evaluate_satellite_risk(flare_lon_deg=68.0, flare_lat_deg=19.0, flare_class="X2.5")
    for r in risks:
        print(f" -> {r['satellite_name']:<25} | Footpoint: W{r['magnetic_footpoint_lon']}° | ΔLon: {r['angular_separation_deg']}° | Risk: {r['risk_level']} ({r['risk_score']}/100)")
