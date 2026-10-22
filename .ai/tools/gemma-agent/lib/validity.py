"""Validity of an attempt record, decided by rule from the record's fields. One implementation shared by the dispatch
tool (which slots to re-issue), the attempt tool (what to tell the graph) and the regeneration script.

  infrastructure_invalid  the harness never reached the model: error set, no model calls (sandbox, Docker, adapter down)
  provider_interrupted    the model was reached, then the provider or transport failed and the harness gave up
  valid                   no error, or the terminal reason is a budget (the agent's call, turn or time cap)

Capped, empty-patch and failed attempts are valid outcomes and are never discarded. A lane time ceiling that binds
before the call cap is still valid; it is visible in the record's budget and tool_calls and is reported as such."""
import re

PROVIDER_ERROR = re.compile(
    r'(provider|upstream|connection|APIConnection|InternalServerError|ServiceUnavailable|BadGateway|RateLimit|'
    r'\b50[0-4]\b|\b429\b|litellm\.)', re.IGNORECASE)


def classify(record: dict) -> str:
    outcome = record.get('outcome') or {}
    error = outcome.get('error')
    calls = outcome.get('total_llm_calls') or 0
    if error and calls == 0:
        return 'infrastructure_invalid'
    if error and PROVIDER_ERROR.search(str(error)):
        return 'provider_interrupted'
    return 'valid'
