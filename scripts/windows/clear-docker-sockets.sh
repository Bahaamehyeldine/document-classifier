#!/bin/sh
# Delete the stale AF_UNIX socket files Docker Desktop leaves under %LOCALAPPDATA%.
# Windows cannot open or delete them (error 1920); WSL can. Called by start-docker-desktop.ps1.
#
#   sh clear-docker-sockets.sh /mnt/c/Users/<you>/AppData/Local
base="${1:?usage: clear-docker-sockets.sh <LOCALAPPDATA as a WSL path>}"
removed=0
for f in "$base"/Docker/run/* "$base"/docker-secrets-engine/*.sock; do
  if [ -e "$f" ]; then
    rm -f -- "$f" && echo "   removed $(basename "$f")" && removed=$((removed + 1))
  fi
done
echo "   $removed stale file(s) removed"
