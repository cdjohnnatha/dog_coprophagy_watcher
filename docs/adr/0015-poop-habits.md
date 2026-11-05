---
status: accepted (2026-11-05)
date: 2026-11-05
---

# Poop habits: usual time, interval and hours since the last poop

## Context and problem

Ellie has a lazy bowel and does not always eat on time. With the poop counter running (ADR-0014) the owner asked for her usual poop times and for a warning when she goes 12 h or more without one. A link with the feeder was considered too.

## Decision

1. **Statistics over the last 14 local days** of counted poops (`final_poop = poop`, manual ones included), in a pure module `poop_habits.py`: usual time, 25-75 % window, median interval between poops, average per day, counts per period of the day (madrugada 00-06, manhã 06-12, tarde 12-18, noite 18-24). 14 days is long enough for about 20 poops (she goes about 1.4 times a day) and short enough to follow a change of habit. The typical time and window need at least 5 poops in the window (`ready`); the interval needs 3. Until then the sensor says "coletando dados" instead of showing a number that means nothing.
2. **The day for the time-of-day statistics starts at 03:00 local.** A poop at 02:43 and one at 06:13 belong to the same part of the day; with a midnight start they would sit at both ends of the scale and the median would land at midday for nobody. Her poops cluster at 06:00-10:00 and again at 16:00-20:00, with rare ones in the night, so 03:00 is the quietest cut.
3. **Hours since the last poop** looks at all counted poops, not only the window, so it keeps counting through a quiet fortnight. It is recomputed in every state publish, which the minute task runs, so it grows with the clock without a new event. A poop in the future is ignored.
4. **The alert lives in Home Assistant** (a helper with the number of hours, an automation like the pee one: waking hours only, once per gap, cleared when a poop is counted). The app only provides the numbers.
5. **No link with the feeder and no prediction.** The owner often takes food out of the feeder to put it in toys, so "eaten" does not mean "ate" and a correlation would be noise. Statistics describe; nothing says "she will go at X". Review after about 3 weeks of data.

## Consequences

- The numbers are only as good as the poop counting: a missed poop lengthens the interval and the hours since. The owner can add one with `manual_poop` or answer a visit.
- A poop is counted for its `started_at`, in local time, so a poop added by hand for the right hour lands in the right period.
- The first week has 11 poops, so the usual time is a rough guide (its window is wide); it tightens as data accumulates.
- Summer-time changes need no special case: the local clock is derived from UTC with the Lisbon zone.
