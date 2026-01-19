"""
Mobile-friendly Web Dashboard for Stock Recommendations
Run with: python -m dashboard.app
"""
from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import REPORTS_DIR

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_latest_report():
    """Load the latest analysis report"""
    report_file = REPORTS_DIR / "latest_report.json"

    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Return sample data if no report exists
    return {
        "generated_at": datetime.now().isoformat(),
        "market_regime": "sideways",
        "recommendations": [],
        "positions": [],
        "performance": {
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }
    }


@app.route('/')
def index():
    """Main dashboard page"""
    report = load_latest_report()
    return render_template('dashboard.html', report=report)


@app.route('/api/report')
def api_report():
    """API endpoint for latest report"""
    report = load_latest_report()
    return jsonify(report)


@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for recommendations only"""
    report = load_latest_report()
    return jsonify(report.get('recommendations', []))


@app.route('/api/positions')
def api_positions():
    """API endpoint for current positions"""
    report = load_latest_report()
    return jsonify(report.get('positions', []))


@app.route('/api/performance')
def api_performance():
    """API endpoint for performance metrics"""
    report = load_latest_report()
    return jsonify(report.get('performance', {}))


if __name__ == '__main__':
    # Create templates and static directories
    templates_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("Stock Recommendation Dashboard")
    print("=" * 50)
    print("Access from your phone:")
    print("  http://<your-pc-ip>:5000")
    print("")
    print("Find your PC's IP with:")
    print("  Windows: ipconfig")
    print("  Mac/Linux: ifconfig or ip addr")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
