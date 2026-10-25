"""Finite queues and fixed offered load; every offered task receives an outcome."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
import time


def utility_function(settings):
    scales = settings['family_scales']
    penalties = settings['status_values']
    def score(episode):
        if episode.status in penalties:
            return float(penalties[episode.status])
        return episode.reward * float(scales[episode.family])
    return score


def run_arrivals(runtime, tasks, environment_factory, policy_factory, settings):
    concurrency, queued = int(settings['concurrency']), int(settings['queue_capacity'])
    interval, timeout = float(settings['interval_seconds']), float(settings['task_timeout_seconds'])
    if concurrency <= 0 or queued < 0 or interval < 0 or timeout <= 0:
        raise ValueError('invalid arrival or queue limits')
    if concurrency > runtime.config.max_sessions or concurrency + queued >= runtime.config.stream_window:
        raise ValueError('arrival limits exceed session or recovery-window capacity')
    origin, pending, counts = time.monotonic(), {}, Counter()
    buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, math.inf]
    histogram, elapsed_sum = [0] * len(buckets), 0.0

    def record(episode, scheduled, wait_seconds=0, finished=None):
        nonlocal elapsed_sum
        if episode is None:
            return
        latency = max(0.0, (time.monotonic() if finished is None else finished) - scheduled)
        counts[episode.status] += 1
        elapsed_sum += latency
        histogram[next(i for i, boundary in enumerate(buckets) if latency <= boundary)] += 1
        runtime.event('service_outcome', task_id=episode.task_id, status=episode.status,
                      latency_seconds=latency, queue_seconds=wait_seconds,
                      utility=runtime.utility(episode) if runtime.utility else episode.reward)

    def execute(task, sequence, scheduled):
        waiting = max(0.0, time.monotonic() - scheduled)
        if time.monotonic() >= scheduled + timeout:
            return runtime.reject_task(task, sequence, 'deadline'), waiting, time.monotonic()
        episode = runtime.run_task(task, environment_factory(task), policy_factory(task), sequence,
                                   deadline=scheduled + timeout)
        return episode, waiting, time.monotonic()

    def collect(block=False):
        if not pending:
            return
        finished, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in finished:
            scheduled = pending.pop(future)
            episode, waiting, finished = future.result()
            record(episode, scheduled, waiting, finished)

    stream = runtime.store.stats()['stream']
    # Resume starts a new measured service segment; never pretend downtime was served.
    first_sequence = stream['high_watermark'] + 1
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='memturn-task') as pool:
        for sequence, task in enumerate(tasks):
            if sequence < first_sequence:
                continue
            scheduled = origin + (sequence - first_sequence) * interval
            while time.monotonic() < scheduled:
                collect()
                time.sleep(min(0.05, max(0, scheduled - time.monotonic())))
            collect()
            while sequence > runtime.store.stats()['stream']['high_watermark'] + runtime.config.stream_window:
                if not pending:
                    raise RuntimeError('recovery window has an unresolved task without an active worker')
                collect(block=True)
            if time.monotonic() >= scheduled + timeout:
                record(runtime.reject_task(task, sequence, 'deadline'), scheduled)
            elif len(pending) >= concurrency + queued:
                record(runtime.reject_task(task, sequence, 'rejected'), scheduled)
            else:
                pending[pool.submit(execute, task, sequence, scheduled)] = scheduled
        while pending:
            collect(block=True)
    total = sum(counts.values())
    threshold, cumulative, p95 = math.ceil(total * 0.95), 0, None
    for bound, count in zip(buckets, histogram):
        cumulative += count
        if total and cumulative >= threshold:
            p95 = bound if math.isfinite(bound) else None
            break
    return {'scope':'current_service_segment', 'offered_outcomes':total, 'statuses':dict(counts),
            'seconds':time.monotonic()-origin, 'latency_mean_seconds':elapsed_sum/total if total else None,
            'latency_p95_bucket_upper_seconds':p95, 'histogram_counts':histogram,
            'histogram_upper_seconds':[b if math.isfinite(b) else None for b in buckets],
            'concurrency':concurrency, 'queue_capacity':queued}
