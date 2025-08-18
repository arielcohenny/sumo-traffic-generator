"""
DBPS GUI Entry Point.

This module provides the command-line entry point for launching the DBPS GUI
as a desktop application experience.
"""

import subprocess
import sys
import os
import platform
import shutil
import time
import requests
from pathlib import Path


def launch_chrome_app_mode(url):
    """Launch Chrome in app mode."""
    system = platform.system()

    if system == "Darwin":  # macOS
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(chrome_path):
            return subprocess.Popen([
                chrome_path,
                f"--app={url}",
                "--no-first-run",
                "--no-default-browser-check"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elif system == "Windows":
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ]
        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                return subprocess.Popen([
                    chrome_path,
                    f"--app={url}",
                    "--no-first-run",
                    "--no-default-browser-check"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    else:  # Linux
        if shutil.which("google-chrome"):
            return subprocess.Popen([
                "google-chrome",
                f"--app={url}",
                "--no-first-run",
                "--no-default-browser-check"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return None


def wait_for_streamlit_ready(url, max_attempts=30, delay=1):
    """Wait for Streamlit server to be ready."""
    print("🔄 Waiting for DBPS server to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("✅ DBPS server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        if attempt < max_attempts - 1:
            time.sleep(delay)
            print(
                f"🔄 Checking server readiness... ({attempt + 1}/{max_attempts})")

    return False








def main():
    """Launch the DBPS GUI as a desktop application."""
    print("🚀 Starting DBPS - Decentralised Bottleneck Prioritization Simulation...")

    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    streamlit_process = None

    try:
        # Start Streamlit server in background
        print("🔄 Starting DBPS server...")
        streamlit_process = subprocess.Popen([
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.port=8501"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for Streamlit to be ready
        url = "http://localhost:8501"
        if not wait_for_streamlit_ready(url):
            print("❌ DBPS server failed to start within 30 seconds")
            if streamlit_process:
                streamlit_process.terminate()
            return

        # Launch Chrome in app mode
        print("🖥️  Launching DBPS application...")
        browser_process = launch_chrome_app_mode(url)

        if browser_process:
            print("✅ DBPS GUI launched successfully!")
            print("💡 Use Ctrl+C to stop DBPS")

            # Keep the process running indefinitely
            # User will use Ctrl+C to stop the server
            try:
                streamlit_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 DBPS stopped")
                if streamlit_process:
                    streamlit_process.terminate()
                    
        else:
            print("❌ Could not launch Chrome app mode")
            if streamlit_process:
                streamlit_process.terminate()

    except FileNotFoundError:
        print("❌ Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        if streamlit_process:
            streamlit_process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching DBPS: {e}")
        if streamlit_process:
            streamlit_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
