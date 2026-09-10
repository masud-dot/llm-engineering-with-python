"""A scripted generator client.

Builds a valid batch from the plan it is given, so the
pipeline, validation, emission, and review are all real. The
wording of each case is authored, not generated.
"""
import json
import re

from llmapp.llm.base import Completion
from llmapp.projects.testgen.plan import CaseKind

FAIL_STATUS = {
    CaseKind.MISSING_REQUIRED: 422,
    CaseKind.WRONG_TYPE: 422,
    CaseKind.BOUNDARY: 422,
    CaseKind.EXTRA_FIELD: 422,
    CaseKind.UNAUTHENTICATED: 401,
}
SLOT = re.compile(r"^- (\S+) \[(\w+)\](.*)$", re.MULTILINE)


class ScriptedGenerator:
    """Answers the testgen prompt from its own plan block."""

    def __init__(self, *, duplicate: bool = False,
                 drop_kind: str = "", trivial: bool = False):
        self.duplicate = duplicate
        self.drop_kind = drop_kind
        self.trivial = trivial
        self.calls = 0

    def complete(self, messages, *, max_output_tokens=512):
        self.calls += 1
        body = messages[-1].content
        endpoint = json.loads(
            body.split("--- BEGIN ENDPOINT ---")[1]
                .split("--- END ENDPOINT ---")[0])
        plan = body.split("--- BEGIN PLAN ---")[1] \
                   .split("--- END PLAN ---")[0]
        cases = []
        for name, kind, note in SLOT.findall(plan):
            if kind == self.drop_kind:
                continue
            k = CaseKind(kind)
            status = _success(endpoint) if k is CaseKind.HAPPY \
                else FAIL_STATUS.get(k, 400)
            if self.trivial and k is not CaseKind.HAPPY:
                status = 200
            headers = {} if k is CaseKind.UNAUTHENTICATED \
                else {"Authorization": "Bearer TEST_TOKEN"}
            payload = {}
            if k is CaseKind.HAPPY:
                payload = {f["name"]: _value(f)
                           for f in endpoint["fields"]}
            elif k in (CaseKind.MISSING_REQUIRED,
                       CaseKind.WRONG_TYPE, CaseKind.BOUNDARY):
                payload = {f["name"]: _value(f)
                           for f in endpoint["fields"]}
                target = note.strip().split()[-1].rstrip(":,")
                if k is CaseKind.MISSING_REQUIRED:
                    payload.pop(target, None)
                elif k is CaseKind.WRONG_TYPE:
                    payload[target] = "not-a-number"
                else:
                    payload[target] = _out_of_bounds(
                        endpoint["fields"], target)
            elif k is CaseKind.EXTRA_FIELD:
                payload = {f["name"]: _value(f)
                           for f in endpoint["fields"]}
                payload["unexpected_field"] = "x"
            cases.append({
                "name": name, "kind": kind,
                "description": f"{kind} case for "
                               f"{endpoint['path']}",
                "method": endpoint["method"],
                "path": endpoint["path"].replace(
                    "{job_id}", "JOB_ID"),
                "headers": headers, "body": payload,
                "expected_status": status,
                "requirement_id": "REQ-" + kind.upper(),
            })
        if self.duplicate and cases:
            twin = dict(cases[0]); twin["name"] = \
                cases[0]["name"] + "_again"
            cases.append(twin)
        text = json.dumps({"cases": cases})
        return Completion(text, 800, 400, "stub", "stop")


def _success(endpoint):
    """Use a declared 2xx status, not an assumed 200."""
    for code in endpoint.get("statuses", []):
        if code.isdigit() and 200 <= int(code) < 300:
            return int(code)
    return 200


ALTERNATION = re.compile(r"\^\(([^)]+)\)\$")


def _value(field):
    if field["type"] in ("integer", "number"):
        return field.get("minimum") or 1
    if field["enum"]:
        return field["enum"][0]
    pattern = field.get("pattern") or ""
    match = ALTERNATION.match(pattern)
    if match:
        return match.group(1).split("|")[0]
    if field["type"] == "object":
        return {}
    return "sample text"


def _out_of_bounds(fields, name):
    for f in fields:
        if f["name"] != name:
            continue
        if f.get("maximum") is not None:
            return f["maximum"] + 1
        if f.get("minimum") is not None:
            return f["minimum"] - 1
        if f.get("max_length") is not None:
            return "x" * (int(f["max_length"]) + 1)
        if f["enum"] or f.get("pattern"):
            return "not-a-legal-value"
        return ""
    return ""
