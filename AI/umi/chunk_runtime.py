"""Bounded N-step runner with injected clock, planner and driver.

Snapshot timestamps MUST already be translated into clock()'s monotonic domain.
Driver.send returns True on acceptance (not physical arrival). Calls must be
bounded by the driver: this synchronous runner cannot interrupt a blocked call.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Snapshot:
    timestamp: float
    payload: object


def run_chunks(acquire, plan, driver, *, cycles, n_steps, period_s,
               max_age_s, max_lateness_s, clock, sleep):
    """Reacquire/replan each cycle; discard unused targets. Stop on any failure.

    First target is scheduled at snapshot time, subsequent targets at +period.
    Late/stale targets abort rather than producing a catch-up command burst.
    plan must validate the whole chunk before returning. No automatic retries.
    """
    if (type(cycles) is not int or cycles <= 0 or type(n_steps) is not int
            or n_steps <= 0 or not all(math.isfinite(v) and v > 0
                for v in (period_s, max_age_s, max_lateness_s))):
        raise ValueError('invalid execution settings')
    sent = 0
    def check(snapshot, deadline=None):
        now = clock()
        age = now - snapshot.timestamp
        if not math.isfinite(now) or not math.isfinite(age) or age < 0 or age > max_age_s:
            raise ValueError('stale or invalid snapshot timestamp')
        if deadline is not None and now - deadline > max_lateness_s:
            raise ValueError('command deadline missed')
    try:
        for _ in range(cycles):
            snapshot = acquire()
            check(snapshot)
            commands = tuple(plan(snapshot.payload))
            if len(commands) < n_steps:
                raise ValueError('chunk shorter than n_steps')
            for i in range(n_steps):
                deadline = snapshot.timestamp + i * period_s
                while clock() < deadline:
                    sleep(deadline - clock())
                check(snapshot, deadline)
                cmd = commands[i]
                if driver.send(cmd.arm_positions_rad.copy(), cmd.gripper_width_m) is not True:
                    raise RuntimeError('driver did not accept command')
                sent += 1
                check(snapshot, deadline)
            # Next observation is acquired at the next tick, not immediately
            # after the last command of this chunk.
            next_tick = snapshot.timestamp + n_steps * period_s
            while clock() < next_tick:
                sleep(next_tick - clock())
        return sent
    except Exception as error:
        try:
            driver.stop()
        except Exception as stop_error:
            raise RuntimeError(f'execution failed: {error}; stop failed: {stop_error}') from error
        raise
