#!/usr/bin/env python3

"""Time stamps.
"""

import datetime


def stamp(today=True, now=False, format=None):
    """..."""
    if today is not True:
        raise NotImplementedError("TODO")

    def lead_zero(number):
        """..."""
        return "0" if number < 10 else ""

    today = datetime.date.today()
    daystamp = (f"{today.year}"
                f"{lead_zero(today.month)}{today.month}"
                f"{lead_zero(today.day)}{today.day}")

    if not now:
        return daystamp

    now = datetime.datetime.now()
    timestamp = (f"{lead_zero(now.hour)}{now.hour}"
                 f"{lead_zero(now.minute)}{now.minute}")

    return (daystamp + "-" + timestamp if not format else format(daystamp, timestamp))
