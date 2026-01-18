"""
Data cache management utilities
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)


class DataCache:
    """
    Manages caching of stock data and calculated indicators
    """

    def __init__(self, cache_dir: Union[Path, str] = CACHE_DIR) -> None:
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache files

        Args:
            older_than_days: If specified, only clear files older than this many days

        Returns:
            Number of files deleted
        """
        count: int = 0
        cutoff_date: Optional[datetime] = None

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

            except OSError as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
            except PermissionError as e:
                logger.warning(f"Permission denied deleting {cache_file}: {e}")

        logger.info(f"Cleared {count} cache files")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached data

        Returns:
            Dictionary with cache statistics:
            - num_files: Number of cached files
            - total_size_mb: Total size in megabytes
            - oldest_file: Timestamp of oldest file
            - newest_file: Timestamp of newest file
            - cache_dir: Path to cache directory
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))

        total_size: int = sum(f.stat().st_size for f in cache_files)
        total_size_mb: float = total_size / (1024 * 1024)

        oldest_date: Optional[datetime] = None
        newest_date: Optional[datetime] = None

        if cache_files:
            oldest = min(f.stat().st_mtime for f in cache_files)
            newest = max(f.stat().st_mtime for f in cache_files)
            oldest_date = datetime.fromtimestamp(oldest)
            newest_date = datetime.fromtimestamp(newest)

        return {
            "num_files": len(cache_files),
            "total_size_mb": round(total_size_mb, 2),
            "oldest_file": oldest_date,
            "newest_file": newest_date,
            "cache_dir": str(self.cache_dir),
        }

    def is_cache_valid(self, symbol: str, start_date: str, end_date: str, max_age_hours: int = 24) -> bool:
        """
        Check if cache for a given symbol and date range is still valid

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_age_hours: Maximum age in hours for cache to be valid

        Returns:
            True if valid cache exists, False otherwise
        """
        safe_symbol = symbol.replace(".", "_")
        filename = f"{safe_symbol}_{start_date}_{end_date}.parquet"
        cache_path = self.cache_dir / filename

        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = timedelta(hours=max_age_hours)

        return cache_age <= max_age
