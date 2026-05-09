"""Agent Registry: name→handler map with capability descriptions for the Planner."""

from dataclasses import dataclass
from typing import Awaitable, Callable

# Handler shape: takes an MCP envelope dict, returns an MCP envelope dict (async).
Handler = Callable[[dict], Awaitable[dict]]


@dataclass
class AgentSpec:
    handler: Handler
    role: str
    inputs: dict[str, str]  # input_key -> human-readable description+type
    output: str  # description of the output payload


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentSpec] = {}

    def register(self, name: str, handler: Handler, role: str,
                 inputs: dict[str, str], output: str) -> None:
        self._agents[name] = AgentSpec(handler, role, inputs, output)

    def get_handler(self, name: str) -> Handler:
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self._agents[name].handler

    def has(self, name: str) -> bool:
        return name in self._agents

    def required_input_keys(self, name: str) -> set[str]:
        return set(self._agents[name].inputs.keys())

    def get_capabilities_description(self) -> str:
        lines = ["Available Agents and their required inputs:"]
        for i, (name, spec) in enumerate(self._agents.items(), 1):
            lines.append(f"\n{i}. AGENT: {name}")
            lines.append(f"   ROLE: {spec.role}")
            lines.append("   INPUTS:")
            for key, desc in spec.inputs.items():
                lines.append(f'   - "{key}": {desc}')
            lines.append(f"   OUTPUT: {spec.output}")
        return "\n".join(lines)
