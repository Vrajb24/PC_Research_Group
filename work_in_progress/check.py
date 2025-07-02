import sys
import os

def check_python_version():
    python_version = sys.version
    version_info = sys.version_info
    
    print(f"Python Version: {python_version}")
    print(f"Major: {version_info.major}")
    print(f"Minor: {version_info.minor}")
    print(f"Micro: {version_info.micro}")
    
    # Check if Python 3.6+
    if version_info >= (3, 6):
        print("You're running Python 3.6 or newer.")
    else:
        print("Warning: You're running an older version of Python. Some features may not work.")

if __name__ == "__main__"
    check_python_version()