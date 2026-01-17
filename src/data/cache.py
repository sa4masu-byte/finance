"""
Data cache management utilities
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)


class DataCache:
    """
    Manages caching of stock data and calculated indicators
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache files

        Args:
            older_than_days: If specified, only clear files older than this many days
        """
        count = 0
        cutoff_date = None

        if older_than_days is not None:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)

        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                if cutoff_date is not None:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time > cutoff_date:
                        continue

                cache_file.unlink()
                count += 1
                logger.debug(f"Deleted cache file: {cache_file}")

            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")

        logger.info(f"Cleared {count} cache files")

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached data

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        if cache_files:
            oldest = min(f.stat().st_mtime for f in cache_files)
            newest = max(f.stat().st_mtime for f in cache_files)
            oldest_date = datetime.fromtimestamp(oldest)
            newest_date = datetime.fromtimestamp(newest)
        else:
            oldest_date = None
            newest_date = None

        return {
            "num_files": len(cache_files),
            "total_size_mb": round(total_size_mb, 2),
            "oldest_file": oldest_date,
            "newest_file": newest_date,
            "cache_dir": str(self.cache_dir),
        }
