"""
Root Entrypoint for Gradio Web Application
===========================================
Allows direct launching from root level: python app.py
"""

from dashboard.app import create_dashboard

demo = create_dashboard()
app = demo.app

import sys

if __name__ == '__main__':
    share_flag = "--share" in sys.argv or True
    try:
        demo.launch(share=share_flag, server_name="0.0.0.0", server_port=7860)
    except OSError:
        demo.launch(share=share_flag, server_name="0.0.0.0")
