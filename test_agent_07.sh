#!/bin/bash
# Quick test script for agent_07 guardrail middleware

echo "=== Testing Agent 07 Guardrail Middleware ==="
echo ""
echo "This test will:"
echo "1. Send a normal calendar query (should work)"
echo "2. Send a malicious query (should be blocked)"
echo ""
echo "Press Ctrl+C to stop if needed"
echo ""

uv run python agents/agent_07_guardrail_middleware.py
