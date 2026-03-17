#!/usr/bin/env bash
set -euo pipefail

# Compress all stats folders from exp/ into save/stats_archives/
# without modifying anything in exp/.

EXP_ROOT="/home/onyxia/work/espnet/egs2/ml_superb/asr1/exp"
SAVE_ROOT="save/stats_archives"

mkdir -p "${SAVE_ROOT}"

find "${EXP_ROOT}" -maxdepth 1 -type d -name "*stats*" | while read -r stats_dir; do
  base_name="$(basename "${stats_dir}")"
  archive_path="${SAVE_ROOT}/${base_name}.tar.gz"

  echo "Compressing ${stats_dir} -> ${archive_path}"
  tar -czf "${archive_path}" -C "${EXP_ROOT}" "${base_name}"
done

echo "Done. Archives saved under ${SAVE_ROOT}/"
