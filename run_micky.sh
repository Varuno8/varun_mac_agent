#!/bin/zsh

VARUN_AGENT_DIR="/Users/varuntyagi/Downloads/varun_agent"

# Pick the most recently built binary between two sources:
#   1. /Applications/Clicky.app — populated by `gh run download` from GitHub Actions builds
#   2. ~/Library/Developer/Xcode/DerivedData/.../Clicky — local Xcode builds
# Whichever is newer wins. This way `bash run_micky.sh` always runs the freshest
# code regardless of whether the last update came from a cloud build or a local rebuild.
APPLICATIONS_BINARY="/Applications/Clicky.app/Contents/MacOS/Clicky"
DERIVED_DATA_BINARY=$(find ~/Library/Developer/Xcode/DerivedData -name "Clicky" -path "*/MacOS/Clicky" -print0 2>/dev/null \
    | xargs -0 ls -td 2>/dev/null \
    | head -1)

# Pick the newer of the two (by mtime). `ls -td` orders newest first.
CANDIDATE_BINARIES=()
[[ -x "$APPLICATIONS_BINARY" ]]    && CANDIDATE_BINARIES+=("$APPLICATIONS_BINARY")
[[ -x "$DERIVED_DATA_BINARY" ]]    && CANDIDATE_BINARIES+=("$DERIVED_DATA_BINARY")

if [[ ${#CANDIDATE_BINARIES[@]} -eq 0 ]]; then
    echo "❌ No Clicky binary found. Build in Xcode (Cmd+B) or download a GitHub Actions artifact to /Applications/Clicky.app."
    exit 1
fi

APP_BINARY=$(ls -td "${CANDIDATE_BINARIES[@]}" | head -1)

BINARY_MTIME=$(stat -f "%Sm" "$APP_BINARY" 2>/dev/null)
echo "📦 Using binary: $APP_BINARY"
echo "   built: $BINARY_MTIME"

# Kill existing instances
pkill -f "Clicky" 2>/dev/null
pkill -f "proxy.py" 2>/dev/null

# Force-free port 8081 — pkill alone races with TCP socket release.
# Kill anything still bound, then wait until the listener is gone.
echo "⏳ Freeing port 8081..."
for i in {1..20}; do
    PIDS_ON_PORT=$(lsof -ti :8081 -sTCP:LISTEN 2>/dev/null)
    if [[ -z "$PIDS_ON_PORT" ]]; then
        break
    fi
    kill -9 $PIDS_ON_PORT 2>/dev/null
    sleep 0.25
done
if [[ -n "$(lsof -ti :8081 -sTCP:LISTEN 2>/dev/null)" ]]; then
    echo "❌ Port 8081 still in use after 5s — aborting."
    lsof -nP -iTCP:8081 -sTCP:LISTEN
    exit 1
fi

# Start the proxy using the venv Python directly (not system Python)
"$VARUN_AGENT_DIR/.venv/bin/python" "$VARUN_AGENT_DIR/proxy.py" > "$VARUN_AGENT_DIR/proxy.log" 2>&1 &
PROXY_PID=$!

# Wait for proxy to be ready (up to 15 seconds).
# Verify the new PID is actually alive AND serving — guards against the case
# where bind() failed and a stale process is responding to /health instead.
echo "⏳ Starting proxy..."
for i in {1..30}; do
    if ! kill -0 $PROXY_PID 2>/dev/null; then
        echo "❌ Proxy PID $PROXY_PID exited. Last 20 lines of proxy.log:"
        tail -20 "$VARUN_AGENT_DIR/proxy.log"
        exit 1
    fi
    if curl -s http://localhost:8081/health > /dev/null 2>&1; then
        echo "✅ Proxy ready (PID: $PROXY_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Proxy failed to start. Check proxy.log for errors."
        cat "$VARUN_AGENT_DIR/proxy.log"
        exit 1
    fi
    sleep 0.5
done

echo "🚀 Starting Micky — logs streaming below. Press Ctrl+C to quit."
echo "────────────────────────────────────────────────────────────────"

# Kill proxy when app exits
trap "kill $PROXY_PID 2>/dev/null; echo '🛑 Proxy stopped.'" EXIT

exec "$APP_BINARY"
