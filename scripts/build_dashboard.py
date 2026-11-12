"""Generate the provisioned Grafana dashboard (``monitoring/grafana/dashboards/understudy.json``).

Dashboard as code: edit this file and re-run it, rather than editing the JSON in the Grafana UI.
Colours follow the project palette: route identity uses fixed categorical hues (student blue,
teacher orange); status colours (red/amber) are reserved for failures and thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
BLUE, ORANGE, GREEN, AMBER, RED, GREY = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#b42318",
    "#8a8984",
)
ROUTE_COLOURS = {"student": BLUE, "teacher": ORANGE, "unavailable": RED, "rejected": GREY}

_id = 0


def nid() -> int:
    """Return the next panel id."""
    global _id
    _id += 1
    return _id


def target(expr: str, legend: str = "", ref: str = "A", instant: bool = False) -> dict:
    """Return a Prometheus query target."""
    return {
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "refId": ref,
        "instant": instant,
        "range": not instant,
    }


def colour_override(name: str, colour: str) -> dict:
    """Return a fixed-colour override for a series."""
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": colour}}],
    }


def stat(
    title: str,
    expr: str,
    unit: str,
    x: int,
    y: int,
    w: int = 4,
    h: int = 4,
    thresholds: list | None = None,
    legend: str = "",
    text_mode: str = "auto",
    description: str = "",
    decimals: int | None = None,
) -> dict:
    """Return a stat panel."""
    steps = thresholds or [{"color": "text", "value": None}]
    return {
        "id": nid(),
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend, instant=True)],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": decimals,
                "thresholds": {"mode": "absolute", "steps": steps},
                "color": {"mode": "thresholds"},
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
            "textMode": text_mode,
            "justifyMode": "auto",
            "orientation": "auto",
        },
    }


def timeseries(
    title: str,
    targets: list[dict],
    unit: str,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    stack: bool = False,
    overrides: list | None = None,
    threshold: float | None = None,
    description: str = "",
    minimum: float | None = 0,
    maximum: float | None = None,
) -> dict:
    """Return a time-series panel (thin lines, optional dashed threshold)."""
    custom = {
        "lineWidth": 2,
        "fillOpacity": 25 if stack else 0,
        "showPoints": "never",
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        "axisSoftMin": minimum,
        "gradientMode": "none",
    }
    defaults = {
        "unit": unit,
        "custom": custom,
        "color": {"mode": "palette-classic"},
        "min": minimum,
        "max": maximum,
    }
    if threshold is not None:
        custom["thresholdsStyle"] = {"mode": "dashed"}
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [{"color": "transparent", "value": None}, {"color": RED, "value": threshold}],
        }
    return {
        "id": nid(),
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def row(title: str, y: int) -> dict:
    """Return a collapsed-false row header."""
    return {
        "id": nid(),
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def build() -> dict:
    """Assemble the dashboard."""
    good_low = [{"color": GREEN, "value": None}]
    panels = [row("Users and routing", 0)]
    y = 1
    panels += [
        stat(
            "Serving student",
            "max by (version) (edge_model_info)",
            "none",
            0,
            y,
            legend="v{{version}}",
            text_mode="name",
            description="Registry version behind the champion alias",
        ),
        stat(
            "Fallback ratio (2m)",
            "understudy:fallback_ratio:rate2m",
            "percentunit",
            4,
            y,
            thresholds=[{"color": "text", "value": None}, {"color": AMBER, "value": 0.55}],
            decimals=1,
            description="Share of questions the student deferred to the teacher. Alert above 55%.",
        ),
        stat(
            "Instant answer p95",
            "understudy:student_latency_p95:rate5m",
            "s",
            8,
            y,
            thresholds=good_low + [{"color": RED, "value": 0.1}],
            decimals=3,
        ),
        stat(
            "Teacher TTFT p95",
            "understudy:teacher_ttft_p95:rate5m",
            "s",
            12,
            y,
            thresholds=good_low + [{"color": AMBER, "value": 5}],
            decimals=2,
        ),
        stat(
            "Unanswered (2m)",
            "understudy:error_ratio:rate2m",
            "percentunit",
            16,
            y,
            thresholds=good_low + [{"color": RED, "value": 0.05}],
            decimals=1,
            description="Fallbacks that failed (server down / timeout). Alert above 5%.",
        ),
        stat(
            "Helpful feedback",
            'sum(edge_feedback_total{helpful="true"}) / sum(edge_feedback_total)',
            "percentunit",
            20,
            y,
            decimals=0,
        ),
    ]
    y += 4
    panels += [
        timeseries(
            "Questions per second by route",
            [target("sum by (route) (rate(edge_requests_total[1m]))", "{{route}}")],
            "reqps",
            0,
            y,
            stack=True,
            overrides=[colour_override(k, v) for k, v in ROUTE_COLOURS.items()],
        ),
        timeseries(
            "Fallback ratio (drift signal)",
            [target("understudy:fallback_ratio:rate2m", "fallback ratio")],
            "percentunit",
            12,
            y,
            threshold=0.55,
            maximum=1,
            overrides=[colour_override("fallback ratio", ORANGE)],
            description="Dashed line: FallbackRatioHigh threshold. Firing starts discovery.",
        ),
    ]
    y += 8
    panels.append(row("Latency", y))
    y += 1
    panels += [
        timeseries(
            "Instant-answer latency (edge, 1 CPU)",
            [
                target(
                    'histogram_quantile(0.5, sum by (le) (rate(edge_request_latency_seconds_bucket{route="student"}[2m])))',
                    "p50",
                ),
                target(
                    'histogram_quantile(0.95, sum by (le) (rate(edge_request_latency_seconds_bucket{route="student"}[2m])))',
                    "p95",
                    "B",
                ),
            ],
            "s",
            0,
            y,
            w=8,
            threshold=0.1,
        ),
        timeseries(
            "Teacher time to first token",
            [
                target(
                    "histogram_quantile(0.5, sum by (le) (rate(gateway_teacher_ttft_seconds_bucket[5m])))",
                    "p50",
                ),
                target(
                    "histogram_quantile(0.95, sum by (le) (rate(gateway_teacher_ttft_seconds_bucket[5m])))",
                    "p95",
                    "B",
                ),
            ],
            "s",
            8,
            y,
            w=8,
            threshold=5,
        ),
        timeseries(
            "Teacher decode speed",
            [
                target(
                    "histogram_quantile(0.5, sum by (le) (rate(gateway_teacher_tokens_per_second_bucket[5m])))",
                    "median tokens/s",
                ),
            ],
            "none",
            16,
            y,
            w=8,
        ),
    ]
    y += 8
    panels.append(row("Answer quality", y))
    y += 1
    panels += [
        timeseries(
            "Student confidence (top FAQ)",
            [
                target(
                    "histogram_quantile(0.5, sum by (le) (rate(edge_student_confidence_bucket[2m])))",
                    "median",
                ),
                target(
                    "histogram_quantile(0.25, sum by (le) (rate(edge_student_confidence_bucket[2m])))",
                    "25th percentile",
                    "B",
                ),
            ],
            "percentunit",
            0,
            y,
            w=8,
            maximum=1,
            description="Top-FAQ probability. A drop means more questions the student does not recognise (drift).",
        ),
        timeseries(
            "Shadow grading: teacher agrees with instant answer",
            [
                target(
                    'sum(increase(edge_shadow_grades_total{verdict="relevant"}[30m])) / sum(increase(edge_shadow_grades_total{verdict=~"relevant|irrelevant"}[30m]))',
                    "agreement",
                )
            ],
            "percentunit",
            8,
            y,
            w=8,
            maximum=1,
            overrides=[colour_override("agreement", GREEN)],
        ),
        timeseries(
            "Gateway outcomes",
            [target("sum by (outcome) (rate(gateway_requests_total[2m]))", "{{outcome}}")],
            "reqps",
            16,
            y,
            w=8,
        ),
    ]
    y += 8
    panels.append(row("Pipeline and model operations", y))
    y += 1
    panels += [
        {
            "id": nid(),
            "type": "bargauge",
            "title": "Last run duration by stage",
            "datasource": DS,
            "gridPos": {"x": 0, "y": y, "w": 8, "h": 9},
            "targets": [
                target(
                    "max by (stage) (pipeline_stage_duration_seconds)", "{{stage}}", instant=True
                )
            ],
            "fieldConfig": {
                "defaults": {"unit": "s", "color": {"mode": "fixed", "fixedColor": BLUE}, "min": 0},
                "overrides": [],
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "basic",
                "showUnfilled": True,
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            },
        },
        {
            "id": nid(),
            "type": "stat",
            "title": "Last run status by stage",
            "datasource": DS,
            "gridPos": {"x": 8, "y": y, "w": 8, "h": 9},
            "targets": [
                target("max by (stage) (pipeline_stage_success)", "{{stage}}", instant=True)
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "1": {"text": "✓ ok", "color": GREEN},
                                "0": {"text": "✕ failed", "color": RED},
                            },
                        }
                    ],
                    "color": {"mode": "fixed", "fixedColor": GREY},
                },
                "overrides": [],
            },
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "colorMode": "background",
                "textMode": "value_and_name",
                "graphMode": "none",
                "orientation": "horizontal",
            },
        },
        stat(
            "Drafts awaiting review",
            "console_pending_drafts",
            "none",
            16,
            y,
            w=4,
            thresholds=[{"color": "text", "value": None}, {"color": AMBER, "value": 1}],
        ),
        stat("Last rollback took", "console_last_rollback_seconds", "s", 20, y, w=4, decimals=1),
        stat(
            "Pipeline runs (since console start)",
            "sum(console_pipeline_runs_total)",
            "none",
            16,
            y + 4,
            w=4,
            h=5,
            decimals=0,
        ),
        stat(
            "Drift triggers (since console start)",
            'sum(console_alert_triggers_total{action="started"})',
            "none",
            20,
            y + 4,
            w=4,
            h=5,
            decimals=0,
        ),
    ]
    y += 9
    panels.append(row("Infrastructure", y))
    y += 1
    panels.append(
        {
            "id": nid(),
            "type": "state-timeline",
            "title": "Alerts firing",
            "datasource": DS,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 6},
            "targets": [
                target('max by (alertname) (ALERTS{alertstate="firing"})', "{{alertname}}")
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "fixed", "fixedColor": RED},
                    "custom": {"fillOpacity": 80, "lineWidth": 0},
                },
                "overrides": [],
            },
            "options": {
                "showValue": "never",
                "mergeValues": True,
                "rowHeight": 0.8,
                "legend": {"showLegend": False},
            },
        }
    )
    y += 6
    panels += [
        stat(
            "Up",
            'up{job=~"edge|gateway|teacher|console|pushgateway"}',
            "none",
            0,
            y,
            w=6,
            h=8,
            legend="{{job}}",
            thresholds=[{"color": RED, "value": None}, {"color": GREEN, "value": 1}],
        ),
        timeseries(
            "Service CPU (cores; edge limit = 1)",
            [
                target(
                    'rate(process_cpu_seconds_total{job=~"edge|gateway|console"}[1m])',
                    "{{job}}",
                )
            ],
            "none",
            6,
            y,
            w=9,
        ),
        timeseries(
            "Service resident memory (edge limit = 1 GiB)",
            [
                target(
                    'process_resident_memory_bytes{job=~"edge|gateway|console"}',
                    "{{job}}",
                )
            ],
            "bytes",
            15,
            y,
            w=9,
        ),
    ]
    return {
        "uid": "understudy",
        "title": "Understudy: edge, teacher and pipeline",
        "tags": ["understudy"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
        "editable": True,
        "annotations": {"list": []},
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "monitoring/grafana/dashboards/understudy.json"
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(out)
