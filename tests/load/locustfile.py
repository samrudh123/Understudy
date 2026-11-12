"""Load test for the edge (Locust).

Two user types:

* ``FrequentAsker`` (90%): asks questions the student can answer instantly (non-streaming).
* ``NewTopicAsker`` (10%): asks unusual questions, which are routed to the teacher and streamed.

Headless run, writing CSV stats into reports/load/::

    locust -f tests/load/locustfile.py --host http://localhost:8001 --headless \
        -u 20 -r 5 -t 2m --csv reports/load/edge --only-summary

Simulated users send ``"simulated"`` metadata so their questions are logged as test traffic,
not live traffic.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

FREQUENT = [
    "what is today's date",
    "can you roll a dice for me",
    "which timezone is new york in",
    "are you a real person",
    "how do i change my pin",
    "what plug do i need in germany",
    "tell me something fun to do in paris",
    "how small a payment can i make on my electric bill",
    "do you have any pets",
    "what's the date today",
]
NEW_TOPICS = [
    "how do i fix a squeaky door hinge",
    "what is the best way to learn the guitar",
    "how long should i let bread dough rise",
    "why is the sky blue",
]
STATUS = {"instant": 0, "teacher": 0, "unavailable": 0, "other": 0}


class FrequentAsker(HttpUser):
    """Asks frequent questions (instant answers expected)."""

    weight = 9
    wait_time = between(0.2, 1.0)

    @task
    def ask(self) -> None:
        """Ask one frequent question."""
        body = {
            "question": random.choice(FREQUENT),
            "stream": False,
            "use_teacher": False,
            "simulated": {"test": True, "label": "load-test"},
        }
        with self.client.post("/ask", json=body, name="/ask (frequent)", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            route = r.json().get("route")
            STATUS["instant" if route == "student" else "other"] += 1
            r.success()


class NewTopicAsker(HttpUser):
    """Asks new questions (streamed from the teacher)."""

    weight = 1
    wait_time = between(2, 5)

    @task
    def ask(self) -> None:
        """Ask one unusual question and read the whole stream."""
        body = {
            "question": random.choice(NEW_TOPICS),
            "stream": True,
            "simulated": {"test": True, "label": "load-test"},
        }
        with self.client.post(
            "/ask",
            json=body,
            name="/ask (teacher, streamed)",
            stream=True,
            catch_response=True,
            timeout=120,
        ) as r:
            text = r.content.decode(errors="replace")
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "event: done" in text:
                STATUS["teacher"] += 1
                r.success()
            elif "event: error" in text:
                STATUS["unavailable"] += 1
                r.success()  # a graceful "server busy" answer is correct behaviour under overload
            else:
                r.failure("stream ended without done/error")


@events.quitting.add_listener
def _summary(environment, **_kw) -> None:
    print("routes:", STATUS)
