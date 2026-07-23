"""Event scheduler for async, multi-rate sensor simulation.

Replaces the old lockstep `for frame in range(...): step everything`
loop with a time-ordered stream of (time, stream_name) events, each
stream ticking at its own nominal period with optional Gaussian timing
jitter and independent random dropout.
"""
import heapq
import numpy as np


class SensorStream:
    """One periodic, possibly-jittered, possibly-lossy event source.

    period: nominal seconds between events.
    jitter_std: stddev (seconds) of Gaussian timing jitter added to each
        scheduled event time. Purely a timing perturbation -- does not
        add or remove events, just moves them in time. 0 disables it.
    dropout_prob: probability in [0, 1] that a scheduled event is
        silently skipped (dropped frame / lost packet / missed
        detection). Applied independently per tick -- NOT autocorrelated,
        so this does not model bursty/correlated dropouts, only isolated
        random loss. If you need burst dropouts, this needs extending.
    min_gap: floor on the gap between consecutive fired events of this
        stream, guarding against jitter occasionally scheduling a "next"
        event at or before the previous one when jitter_std is large
        relative to period. Defaults to 10% of period.
    """

    def __init__(self, name, period, jitter_std=0.0, dropout_prob=0.0,
                 min_gap=None, rng=None):
        self.name = name
        self.period = period
        self.jitter_std = jitter_std
        self.dropout_prob = dropout_prob
        self.min_gap = min_gap if min_gap is not None else 0.1 * period
        self.rng = rng if rng is not None else np.random.default_rng()
        self._last_fire_time = None

    def next_time(self, after):
        nominal = after + self.period
        jittered = nominal + (self.rng.normal(0, self.jitter_std) if self.jitter_std else 0.0)
        if self._last_fire_time is not None:
            jittered = max(jittered, self._last_fire_time + self.min_gap)
        return jittered

    def should_drop(self):
        return self.dropout_prob > 0 and self.rng.random() < self.dropout_prob


class EventScheduler:
    """Merges several SensorStreams into one time-ordered event sequence.

    The primary timer stream is just another SensorStream by convention
    (jitter_std=0, dropout_prob=0) -- nothing here enforces that; it's
    on the caller to configure it that way if a reliable backbone is
    wanted.
    """

    def __init__(self, streams, t0=0.0):
        self.streams = {s.name: s for s in streams}
        self._heap = []
        for s in self.streams.values():
            heapq.heappush(self._heap, (s.next_time(t0), s.name))

    def pop_next(self, horizon=None):
        """Return (time, stream_name) of the next surviving event, or
        None if the heap's next event is past `horizon` (event is put
        back so a later call with a larger horizon still sees it) or
        the schedule is otherwise exhausted (never happens here, since
        streams reschedule themselves forever).
        """
        while self._heap:
            t, name = heapq.heappop(self._heap)
            if horizon is not None and t > horizon:
                heapq.heappush(self._heap, (t, name))
                return None
            stream = self.streams[name]
            stream._last_fire_time = t
            heapq.heappush(self._heap, (stream.next_time(t), name))
            if stream.should_drop():
                continue
            return (t, name)
        return None