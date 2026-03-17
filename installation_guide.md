# Installation guide

1. `git clone https://github.com/espnet/espnet.git`
2. Create a virtualenv. In my case, I got some torchaudio issues at the end, for which i needed to downgrade Python3.12
3. Dependencies:

3.1. `pip install -e esnpet`

3.2. S3PRL. You can try `pip install s3prl[all]`, but I got some problems. My workaround:
```bash
git clone https://github.com/s3prl/s3prl.git
pip install -e s3prl
```

3.3.
Sox
```bash
sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all && sudo apt-get install screen
```
3.4. (Optional) Flash Attention (takes forever..., it can be patched and not used).
```bash
pip install flash_attn --no-build-isolation
```

3.5. (only if needed) To fix torchaudio backend issues, downgrade torch and torchchaudio to 2.4.0
```bash
pip install torch==2.4.0 torchaudio==2.4.0
```

3.6 sclite (for evaluation)
```bash
cd espnet/tools
make scnet
```
<!-- ```bash
cd /tmp
git clone https://github.com/usnistgov/SCTK.git
cd SCTK
make config
make all
make install
export PATH="$PWD/bin:$PATH"
which sclite
``` -->
```bash
cd ~/work
mkdir lib
cd lib
git clone https://github.com/usnistgov/SCTK.git
cd SCTK
make config
make all
make install
export PATH="$PWD/bin:$PATH"
which sclite
```
or after installing it
```bash
cd lib/SCTK
make config
make all
make install
export PATH="~/work/lib/SCTK/bin:$PATH"
which sclite
cd ../..
```

3.7 Loralib

```
pip install git+https://github.com/microsoft/LoRA
```

4. Download dataset. One alternative is through huggingface:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
# hf auth login  # not necessary
# hf download ftshijt/mlsuperb_8th --repo-type dataset   # for the full dataset
hf download rodrigo-paganini/ml_superb_nano_set --repo-type dataset  # for a small test version
# downloads in ~/.cache/huggingface/hub/datasets--ftshijt--mlsuperb_8th/...HASH.../, cd to here
mkdir ~/work/data
# unzip eighth_version.zip -d ~/data/eighth_version
unzip subset_for_testing.zip -d ~/work/data/subset_for_testing
```

# Run guide

## Benchmark

Follow the multilingual ml-superb [README](espnet/egs2/ml_superb/asr1/README.md). Essentially
1. Download the data
2. cd to `espnet/egs2/ml_superb/asr1/`
3. Setting up a configuration file (see `conf/tuning/`).
4. Edit the `MLSUPERB` variable in `db.sh` to point to the data folder.
5. Run
```bash
./run_multi.sh --asr_config <your_training_config> --duration {10min, 1h} --lid false --only_lid false
```
For example:
```bash
# ./run_multi.sh --asr_config conf/tuning/train_asr_s3prl_10min.yaml --lid false --only_lid false --nj 2
./run_multi.sh --asr_config conf/tuning/project_1_10min_hl.yaml --duration 10min --lid false --only_lid false --nj 2
```
or to resume training
```bash
./run_multi.sh --stage 11 --nj 2 --asr_config conf/tuning/train_asr_s3prl_10min.yaml
# ./run_multi.sh --stage 11 --nj 2 --asr_config conf/tuning/project_1_10min_hl_houslby.yaml
```
or to validate
```bash
./run_multi.sh --stage 12 --nj 2 --asr_config conf/tuning/project_1_10min_hl_lora.yaml
```

_Note: For running a subset of all the ML-Superb Data, you will need to edit `espnet/egs2/ml_superb/asr1/local/data_prep.py` to the list of sub-datasets contained. For example, `subset_for_testing` contains only commonvoice, fleurs and mexico-el._

_Note 2: A lot of times CPU memory blows up during stats computation. Change nj to 2 in run_multi.sh to avoid this._

6. Experiments logs are saved in `exp/`
### Configuration file changes

Several changes are necessary to run ASR task on such few amount of data

- `num_iters_per_epoch: 500`. This is originally much higher to run the full benchmark.
- `accum_grad: 8` can be tweaked to exploit available resources
- `patience: 5` to prevent an excessively long training.
- wandb setup:
```
use_wandb: true
wandb_project: ml-superb
wandb_name: hubert_large_multilingual_10min
```

### Add ons

WandB
```bash
pip install wandb editdistance
wandb login
```

# Troubleshooting

## Installation

 - `pip install espnet` errors:

For
```
subprocess.CalledProcessError: Command '['./build_bundled.sh', '0.2.0']' returned non-zero exit status 127.
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for sentencepiece
Failed to build sentencepiece
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> sentencepiece
```
Edit [pyproject.toml](espnet/pyproject.toml) such that
Run
```
---    "sentencepiece==0.2.0",
+++    "sentencepiece==0.2.1",
```

# Run Multi



Running exp/asr_stats_multilingual_10min/run.sh. Got torchaudio backend and flash attention problems. Solving through copilot. Import s3prl works.
I would attempt to make proper installation of espnet following https://s3prl.github.io/s3prl/tutorial/installation.html, before trying anything else.
If not, follow copilot edits and patch the problems.