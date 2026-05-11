#!/bin/zsh
set -euo pipefail

REPO="Varuno8/varun_mac_agent"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "⬇️  Fetching latest Clicky build from GitHub Actions..."

# Download the most recent successful Clicky artifact
if ! gh run download --repo "$REPO" --name Clicky --dir "$TMP" 2>/dev/null; then
  echo ""
  echo "❌ No artifact found. Either:"
  echo "   • CI hasn't run yet — push your changes and wait ~5 min"
  echo "   • Check status: gh run list --repo $REPO"
  exit 1
fi

ZIP=$(find "$TMP" -name "*.zip" | head -1)
if [ -z "$ZIP" ]; then
  echo "❌ No zip found in downloaded artifact"
  exit 1
fi

echo "📦 Installing to /Applications/Clicky.app..."
sudo rm -rf /Applications/Clicky.app
ditto -x -k "$ZIP" /Applications/

echo ""
echo "✅ Clicky updated."
echo ""
echo "⚠️  If this is your first cloud build, re-grant TCC permissions:"
echo "   System Settings → Privacy & Security → Accessibility → enable Clicky"
echo "   System Settings → Privacy & Security → Screen Recording → enable Clicky"
echo ""
echo "Then run: ./run_micky.sh"
