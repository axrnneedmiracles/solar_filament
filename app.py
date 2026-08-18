"""
Root Entrypoint for Gradio Web Application
===========================================
Allows direct launching from root level: python app.py
"""

from dashboard.app import create_dashboard

demo = create_dashboard()
app = demo.app

if __name__ == '__main__':
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
