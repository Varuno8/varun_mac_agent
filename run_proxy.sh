#!/bin/zsh

VARUN_AGENT_DIR="/Users/varuntyagi/Downloads/varun_agent"

# Kill any existing proxy and ALL Clicky instances to prevent duplicate API calls
pkill -f "proxy.py" 2>/dev/null
pkill -f "Clicky" 2>/dev/null
sleep 0.5

# Start the proxy — output goes directly to terminal, not a log file
echo "⏳ Starting proxy..."
"$VARUN_AGENT_DIR/.venv/bin/python" "$VARUN_AGENT_DIR/proxy.py" &
PROXY_PID=$!

# Wait for proxy to be ready (up to 15 seconds)
for i in {1..30}; do
    if curl -s http://localhost:8081/health > /dev/null 2>&1; then
        echo "✅ Proxy ready (PID: $PROXY_PID)"
        echo "📡 Streaming logs below. Press Ctrl+C to stop."
        echo "────────────────────────────────────────────────"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Proxy failed to start."
        exit 1
    fi
    sleep 0.5
done

trap "kill $PROXY_PID 2>/dev/null; echo '🛑 Proxy stopped.'" EXIT
wait $PROXY_PID
