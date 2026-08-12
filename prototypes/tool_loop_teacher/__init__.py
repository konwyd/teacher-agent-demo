"""Isolated tool-loop teacher prototype.

Nothing in this package is imported by the production application.  The
prototype deliberately reuses public ``app`` contracts without registering a
new runtime path.
"""

from prototypes.tool_loop_teacher.loop import AgentLoop, DeepSeekActionProvider

__all__ = ["AgentLoop", "DeepSeekActionProvider"]
