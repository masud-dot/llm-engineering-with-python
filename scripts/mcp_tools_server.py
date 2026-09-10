"""Run the reference application's tools as an MCP server."""

from pydantic import BaseModel, Field

from llmapp.tools.base import SideEffect, ToolSpec
from llmapp.tools.mcp_server import build_server
from llmapp.tools.registry import ToolRegistry

FACTS = {
    "refund window": "Refunds are issued within fourteen days.",
    "escalation": "Unresolved tickets go to a senior engineer.",
}


class LookupArgs(BaseModel):
    topic: str = Field(min_length=1, max_length=100)


def lookup(args: LookupArgs) -> str:
    """Look up a policy statement by topic."""
    return FACTS.get(args.topic, "no entry for that topic")


def main() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="lookup_policy",
            description="Look up a policy statement by topic.",
            arguments=LookupArgs,
            handler=lookup,
            side_effect=SideEffect.READ,
        )
    )
    build_server(registry).run()


if __name__ == "__main__":
    main()
