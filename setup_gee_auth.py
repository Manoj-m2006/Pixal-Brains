"""
Simple Earth Engine Authentication Script
Run this once to set up Earth Engine access
"""

print("=" * 70)
print("🌍 GOOGLE EARTH ENGINE - ONE-TIME SETUP")
print("=" * 70)
print()
print("This will open your browser to authenticate with Google.")
print("After signing in, you'll get a code to paste here.")
print()

input("Press ENTER to continue...")

import subprocess
import sys
import os

# Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run earthengine authenticate
print("\n🌐 Opening browser for authentication...")
print("(If browser doesn't open, copy the URL shown below)\n")

try:
    # Get the path to earthengine command
    ee_cmd = os.path.join("astrava_env", "Scripts", "earthengine.exe")
    
    # Run authentication
    result = subprocess.run([ee_cmd, "authenticate"], check=True)
    
    print("\n" + "=" * 70)
    print("✅ AUTHENTICATION SUCCESSFUL!")
    print("=" * 70)
    print()
    print("You can now use Google Earth Engine!")
    print("Your satellite images will be 5-10x faster with no rate limits.")
    print()
    print("Next step: Start your Django server")
    print()
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Authentication failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're connected to internet")
    print("2. Try closing your browser and trying again")
    print("3. Check if you have a Google account")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
