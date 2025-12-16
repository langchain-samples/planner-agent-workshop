#!/bin/bash
# Quick test script for agent_06 interrupt surfacing

echo "=== Testing Agent 06 Interrupt Surfacing ==="
echo ""
echo "This test will:"
echo "1. Run the supervisor agent"
echo "2. Show when subagent interrupts are surfaced"
echo "3. Demonstrate multiple interrupt handling"
echo ""
echo "Press Ctrl+C to stop if needed"
echo ""

uv run python agents/agent_06_supervisor_multi_agent.py
