"""
Upload report to PythonAnywhere via API

Required environment variables:
  PA_USERNAME: PythonAnywhere username
  PA_API_TOKEN: PythonAnywhere API token

Get your API token from:
  https://www.pythonanywhere.com/account/#api_token
"""
import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# PythonAnywhere API base URL
PA_API_BASE = "https://www.pythonanywhere.com/api/v0/user/{username}/files/path{path}"


def upload_file(username: str, api_token: str, local_path: Path, remote_path: str) -> bool:
    """Upload a file to PythonAnywhere using their API"""
    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Run: pip install requests")
        return False

    url = PA_API_BASE.format(username=username, path=remote_path)

    with open(local_path, 'rb') as f:
        content = f.read()

    response = requests.post(
        url,
        headers={"Authorization": f"Token {api_token}"},
        files={"content": content}
    )

    if response.status_code in [200, 201]:
        logger.info(f"Successfully uploaded {local_path} to {remote_path}")
        return True
    else:
        logger.error(f"Failed to upload: {response.status_code} - {response.text}")
        return False


def reload_webapp(username: str, api_token: str, domain: str) -> bool:
    """Reload the PythonAnywhere web app"""
    try:
        import requests
    except ImportError:
        return False

    url = f"https://www.pythonanywhere.com/api/v0/user/{username}/webapps/{domain}/reload/"

    response = requests.post(
        url,
        headers={"Authorization": f"Token {api_token}"}
    )

    if response.status_code == 200:
        logger.info(f"Successfully reloaded webapp {domain}")
        return True
    else:
        logger.warning(f"Failed to reload webapp: {response.status_code}")
        return False


def main():
    # Get credentials from environment
    username = os.environ.get("PA_USERNAME")
    api_token = os.environ.get("PA_API_TOKEN")

    if not username or not api_token:
        logger.error("Missing PA_USERNAME or PA_API_TOKEN environment variables")
        logger.info("Set these in GitHub repository secrets")
        sys.exit(1)

    # Paths
    project_root = Path(__file__).parent.parent
    local_report = project_root / "data" / "reports" / "latest_report.json"
    remote_report = f"/home/{username}/finance/data/reports/latest_report.json"

    # Check if report exists
    if not local_report.exists():
        logger.error(f"Report not found: {local_report}")
        sys.exit(1)

    # Upload report
    logger.info("Uploading report to PythonAnywhere...")
    success = upload_file(username, api_token, local_report, remote_report)

    if not success:
        sys.exit(1)

    # Optionally reload webapp (not strictly necessary for static JSON)
    domain = f"{username}.pythonanywhere.com"
    reload_webapp(username, api_token, domain)

    logger.info("Upload complete!")


if __name__ == "__main__":
    main()
