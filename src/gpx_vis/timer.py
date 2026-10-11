"""Tiny wall-clock timer used to report how long loading and mapping took."""
import logging
from datetime import timedelta
from time import time

import humanfriendly

logger = logging.getLogger(__name__)


class InvalidTimerCall(Exception):
    """Raised when ``Timer.end()`` is called before ``Timer.begin()``."""


class Timer:
    """Measure elapsed wall-clock time and log it in a human-friendly way.

    >>> timer = Timer()
    >>> timer.begin()
    >>> ...            # do work
    >>> timer.end()    # logs "Time elapsed: 1.2 seconds." at DEBUG level
    """

    def __init__(self):
        self.start = None
        self.stop = None

    def secs2str(self):
        """Elapsed time between ``begin()`` and ``end()`` as readable text."""
        elapsed = timedelta(seconds=self.stop - self.start)
        return humanfriendly.format_timespan(elapsed)

    def begin(self):
        """Record the start time."""
        self.start = time()

    def end(self):
        """Record the end time, log the elapsed time and reset the timer."""
        if self.start is None:
            raise InvalidTimerCall('Timer.end() called before Timer.begin().')
        self.stop = time()
        time_str = self.secs2str()
        separator = '~' * (len(time_str) + 15)
        logger.debug(f'{separator}\nTime elapsed: {time_str}.\n{separator}')
        self.start = None
        self.stop = None
