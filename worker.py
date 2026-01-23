"""
Worker entry point - redirects to src module for backward compatibility.
This allows existing systemd services to continue working.
"""

# Redirect to the refactored worker in src/app
from src.app.worker import main

if __name__ == '__main__':
    main()
