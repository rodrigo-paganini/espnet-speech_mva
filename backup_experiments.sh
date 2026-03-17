#!/usr/bin/env bash
set -euo pipefail

# Copy selected experiment folders from exp/ into save/, keeping only
# logs, stats, and a small set of checkpoints in the copied version.
# This never modifies or deletes anything inside exp/.

EXP_ROOT="espnet/egs2/ml_superb/asr1/exp"
SAVE_ROOT="save_exps_mono"
ARCHIVE_NAME="exp_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

# Edit this list to the experiment folders you want to back up.
SELECTED_FOLDERS=(
  "asr_train_asr_fbank_single_yor_10min"
  "asr_train_asr_fbank_single_wol_10min"
  "asr_train_asr_fbank_single_swa_10min"
  "asr_train_asr_fbank_single_lug_10min"
  "asr_train_asr_fbank_single_luo_10min"
  "asr_train_asr_fbank_single_orm_10min"
  "asr_train_asr_fbank_single_sna_10min"
  "asr_train_asr_fbank_single_umb_10min"
  "asr_train_asr_fbank_single_kin_10min"
  "asr_train_asr_fbank_single_lin_10min"
  "asr_train_asr_fbank_single_hau_10min"
  "asr_train_asr_fbank_single_ibo_10min"
)

# Checkpoints to keep inside copied experiment dirs.
KEEP_CHECKPOINTS=(
  "valid.loss.best.pth"
  "valid.loss.ave.pth"
  "checkpoint.pth"
)

mkdir -p "${SAVE_ROOT}"

copy_folder() {
  local src_dir="$1"
  local dst_dir="$2"

  if [[ ! -d "${src_dir}" ]]; then
    echo "Skipping missing folder: ${src_dir}"
    return
  fi

  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"
}

prune_checkpoints_in_copy() {
  local target_dir="$1"

  find "${target_dir}" -type f -name "*.pth" | while read -r pth_file; do
    local base
    base="$(basename "${pth_file}")"

    local keep=false
    for wanted in "${KEEP_CHECKPOINTS[@]}"; do
      if [[ "${base}" == "${wanted}" ]]; then
        keep=true
        break
      fi
    done

    if [[ "${keep}" == false ]]; then
      rm -f "${pth_file}"
    fi
  done
}

for folder in "${SELECTED_FOLDERS[@]}"; do
  src="${EXP_ROOT}/${folder}"
  dst="${SAVE_ROOT}/${folder}"

  echo "Copying ${src} -> ${dst}"
  copy_folder "${src}" "${dst}"

  echo "Pruning copied checkpoints in ${dst}"
  prune_checkpoints_in_copy "${dst}"
done

tar -czf "${SAVE_ROOT}/${ARCHIVE_NAME}" -C "${SAVE_ROOT}" "${SELECTED_FOLDERS[@]}"

echo "Backup complete:"
echo "  copied folders under: ${SAVE_ROOT}/"
echo "  archive created at:   ${SAVE_ROOT}/${ARCHIVE_NAME}"
echo
echo "Nothing inside ${EXP_ROOT}/ was modified."
