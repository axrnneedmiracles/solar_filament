"""
NASA DONKI Space Weather Client
===============================
Official client for NASA's Space Weather Database Of Notifications, Knowledge, Information (DONKI)
run by the Space Weather Research Center (SWRC) at NASA GSFC.

Endpoints Supported:
- Solar Flares: /DONKI/FLR
- Coronal Mass Ejections: /DONKI/CME
- Solar Energetic Particle Events: /DONKI/SEP
- Space Weather Forecaster Notifications: /DONKI/notifications
- WSA-ENLIL Solar Wind Simulations: /DONKI/WSAEnlilSimulations
"""

import os
import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_NASA_API_KEY = os.environ.get("NASA_API_KEY", "g10YORHBV05NHqE45OgWoZGyqK9AENqwarAjeNGy")


def parse_heliographic_location(loc_str: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Parses standard heliographic coordinate string like 'N19W68' or 'S12E45'.
    Returns signed degrees (Lat: North > 0, South < 0; Lon: West > 0, East < 0 for Parker spiral / Stonyhurst).
    """
    if not loc_str or not isinstance(loc_str, str):
        return None
    
    loc_str = loc_str.strip().upper()
    match = re.search(r'([NS])(\d+)\s*([EW])(\d+)', loc_str)
    if not match:
        return None
    
    ns, lat_deg, ew, lon_deg = match.groups()
    lat = float(lat_deg) * (1.0 if ns == 'N' else -1.0)
    # Standard heliographic western longitude (West is positive towards Earth magnetic footpoint connectivity)
    lon_signed = float(lon_deg) * (1.0 if ew == 'W' else -1.0)
    
    return {
        'latitude': lat,
        'longitude': lon_signed,
        'is_western': (ew == 'W'),
        'raw': loc_str
    }


class NASA_DONKI_Client:
    """
    NASA Space Weather Database Client with local caching and coordinate parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "cache_donki",
        timeout: int = 15
    ):
        self.api_key = api_key or os.environ.get("NASA_API_KEY", DEFAULT_NASA_API_KEY)
        self.base_url = "https://api.nasa.gov/DONKI"
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(self.cache_dir, exist_ok=True)

    def _fetch_cached_endpoint(
        self,
        endpoint: str,
        params: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Queries DONKI API with disk caching to respect NASA rate limits."""
        params_with_key = dict(params)
        params_with_key['api_key'] = self.api_key

        # Build clean cache key from endpoint and sorted params (without API key)
        param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        cache_filename = f"{endpoint.replace('/', '_')}_{param_str}.json"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        # Perform HTTPS Request
        query_string = urllib.parse.urlencode(params_with_key)
        url = f"{self.base_url}/{endpoint}?{query_string}"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'SolarFilamentPlatform/2.0 (NASA Space Weather Research Extension)'}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_data = resp.read().decode('utf-8')
                    data = json.loads(raw_data)
                    # Cache successful query
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    return data
                else:
                    return []
        except Exception as e:
            # If rate limit or offline, return empty list or fallback to cache
            print(f"[!] Warning: DONKI API query failed for {endpoint}: {e}")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
            return []

    def get_flares(
        self,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves solar flare event records from NASA DONKI /FLR.
        """
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {'startDate': start_date, 'endDate': end_date}
        raw_flares = self._fetch_cached_endpoint("FLR", params)

        enriched = []
        for f in raw_flares:
            item = dict(f)
            # Parse heliographic coordinates
            coords = parse_heliographic_location(f.get('sourceLocation'))
            item['heliographic_coords'] = coords
            item['class_type'] = f.get('classType', 'Unknown')
            item['active_region'] = f.get('activeRegionNum')
            enriched.append(item)
        return enriched

    def get_cmes(
        self,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Coronal Mass Ejections (CMEs) from NASA DONKI /CME.
        """
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {'startDate': start_date, 'endDate': end_date}
        return self._fetch_cached_endpoint("CME", params)

    def get_seps(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Solar Energetic Particle (SEP) events from NASA DONKI /SEP.
        Crucial ground-truth record for satellite radiation impact validation.
        """
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {'startDate': start_date, 'endDate': end_date}
        return self._fetch_cached_endpoint("SEP", params)

    def get_notifications(
        self,
        start_date: str = "2024-05-01",
        end_date: Optional[str] = None,
        notif_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves human-written space weather forecaster notifications from NASA CCMC / SWRC.
        """
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {'startDate': start_date, 'endDate': end_date, 'type': notif_type}
        return self._fetch_cached_endpoint("notifications", params)

    def get_wsa_enlil_simulations(
        self,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves WSA-ENLIL solar wind and CME propagation simulations.
        """
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")

        params = {'startDate': start_date, 'endDate': end_date}
        return self._fetch_cached_endpoint("WSAEnlilSimulations", params)


if __name__ == '__main__':
    client = NASA_DONKI_Client()
    flares = client.get_flares("2024-05-01", "2024-05-15")
    seps = client.get_seps("2024-01-01", "2024-06-01")
    print(f"[+] Loaded {len(flares)} flares and {len(seps)} SEP radiation events.")
