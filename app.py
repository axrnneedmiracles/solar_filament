"""
Root Entrypoint for Gradio Web Application
===========================================
Allows direct launching from root level: python app.py
"""

import sys
import socket
from dashboard.app import create_dashboard

demo = create_dashboard()
app = getattr(demo, 'app', None)


def find_free_port(start_port: int = 7860, max_tries: int = 20) -> int:
    """Finds an available local TCP port cleanly without socket collisions."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


if __name__ == '__main__':
    share_flag = "--share" in sys.argv
    port = find_free_port(7860)

    print(f"[*] Starting Solar Filament AI Research Platform on http://127.0.0.1:{port} ...")
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=share_flag,
        )
    except Exception as e:
        print(f"[!] Warning on launch ({e}). Retrying standard local launch on port {port}...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
        )
