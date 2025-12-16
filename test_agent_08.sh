#!/bin/bash
# Quick test script for agent_08 image handling

echo "=== Testing Agent 08 Image Handling ==="
echo ""
echo "This test will:"
echo "1. Demonstrate custom state with image field"
echo "2. Show how middleware processes images"
echo ""
echo "Press Ctrl+C to stop if needed"
echo ""

uv run python agents/agent_08_image_handling.py
