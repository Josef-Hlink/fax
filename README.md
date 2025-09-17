# FAX

<!-- markdownlint-disable MD013 -->

The aim of this project is to investigate the age-old question that has plagued the competitive SSBM scene for ages;

_Who is cooler: the generalist or the specialist?_

To answer this question, we train four AI agents using the same approach (imitation learning with a GPT), only their training data differs.

We have three buckets of data:

1. Fox vs Fox games (FvF) aka dittos or mirror matches
2. Games where one player is Fox and the other is any other character (FvX) or (XvF)
3. Games where neither player is Fox (XvX)

Note that the second bucket can be used to train an agent to play as Fox against any other character, but also to play as any other character against Fox.
This is why we have four agents: FvF, FvX, XvF, and XvX.
More on data in the [Data](#data) section.

All agents will be evaluated at the end of their training on their performance in the ditto matchup (FvF).

## Instructions

If you want to play around with this project, please follow these instructions to set up your own dev environment.
When you get stuck, don't hesitate to create an issue!
I'll probably be working on this after my thesis is done in my free time anyway, so I'll gladly help you out.

### Repo setup

This project uses Astral's [uv](https://github.com/astral-sh/uv) with python 3.11.
I highly recommend you install it, as it will make the setup (and running) of this project much easier, as well as all your future python projects.
The very first thing you should do (after installing uv) is cloning this repo.

```sh
git clone git@github.com:Josef-Hlink/fax.git
```

#### Package: peppi-py

One of the complications we have to deal with setup-wise is that we need the newest version (0.8.6 as of writing) of [peppi-py](https://github.com/hohav/peppi-py) for the replay parsing to work.
Since this version is not yet on PyPI, we have to install and build it from source.
This project is actually a python wrapper around the rust package [peppi](https://github.com/hohav/peppi), so we also need to have rust installed.
I know it's a bit of a hassle, but it's worth it.
To install rust, please follow the instructions on [rust-lang.org](https://www.rust-lang.org/tools/install).

When that's done, clone the peppi-py repo and checkout the 0.8.6 tag:

```sh
git clone git@github.com:hohav/peppi-py.git
cd peppi-py
git checkout v0.8.6
```

#### Package: fax

Now update the path that points to your local peppi-py clone in the [pyproject.toml](pyproject.toml) of our own repo (fax).

```toml
[tool.uv.sources]
# change this line to fit your setup
peppi-py = { path = "/home/jdham/Developer/ext/peppi-py" }
```

To create a .venv and install the dependencies, you can now run a simple `uv sync` in the top-level of this repo.

If you really don't want to use uv, you can also do it manually with the following commands:

```sh
python3 -m venv .venv  # make sure your python binary is 3.11 (or higher)
source .venv/bin/activate
pip install -e .  # pulls deps from pyproject.toml in editable mode

cd path/to/peppi-py
maturin develop  # builds and installs peppi-py into the active venv
cd -  # back to fax
python -c "import peppi_py; print(peppi_py.__file__)"  # verify
```

If you're not using uv, simple replace `uv run <script>` with `python3 <script>` in the rest of these instructions.

#### Configuration

In the top-level of this repo you'll find [defaults.toml](defaults.toml).
This file, together with [config.py](fax/config.py), is used to configure the project.
Any runnable script (marked by a `#!/usr/bin/env python3` shebang) will expose (a subset of) these parameters as command-line arguments, but if you modify defaults.toml you'll rarely have to pass any arguments manually.
For more info on every parameter, see [help.toml](help.toml), I highly recommend you give this a read, as it's the closest thing to an actual documentation page we have right now.
This file is also where the argparser gets its help text from.

Some parameters you'll definitely want to change are the paths to your slippi binary and your SSBM .iso file.
More on those in the next section.

### External dependencies

To install and configure the Slippi launcher please see [slippi.gg](https://slippi.gg) for instructions.
This app is not strictly required for making this project work, but its replay viewer is very useful for one-off replay inspection.

In this project I've used [@vladfi1](https://github.com/vladfi1)'s headless build, which also enables fastforward mode, which is hugely useful.
The prebuilt binary can be found [on his Ishiiruka fork](https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.1.0/Slippi_Online-x86_64-ExiAI.AppImage).
You can also find a mainline build [on his dolphin fork](https://github.com/vladfi1/dolphin/releases/download/slippi-nogui-v0.1.0/Slippi_Netplay_Mainline_NoGui-x86_64.AppImage).

You will need to burn your physical SSBM disk to build an .iso file for any of this to work, or you could try some other way to obtain the .iso.

### Data

The data used is gathered from three zipped archives of ranked netplay replays shared with me by, again, vladfi1 aka xpilot.
Each of these archives contains \~130k high-quality\* .slp replays, totalling \~400k replays at \~300GB zipped.

For my experiments I only need 16.384 games per agent, so I iterate over the zipped archives directly, unzipping and parsing on-the-fly, until all I have enough replays for each agent.
This way we end up only keeping \~180GB of unzipped .slp files that we can then turn into \~40GB of zstd-compressed .mds files. <!-- TODO: verify -->
In order for the agent to train, it will need to unpack them, but by that time you can just delete the original .slp files again.

#### Full data prep pipeline

Note that I do not mention any paths or CLI arguments to the `uv run` commands below, as you should configure these to suit your own setup in [defaults.toml](defaults.toml).

1. Find the links to the desired ranked archives from [vladfi1's dropbox](https://www.dropbox.com/scl/fo/r9qremhl811h6vl6kadfy/AJo-dt9-WC47Qm-s2eRlh9U?rlkey=jn88morgmcy1f1qvc030z5rrd&e=1&st=c6kexo8v&dl=0).
    I pulled the first three.
2. Download them with wget (or any other download manager).
    I ran three instances of wget in parallel (`wget --show-progress -O ./ranked-anonymized-{1,2,3}.zip <link to archive>`), as each download took about an hour to complete with very connectivity.
3. Run `uv run fax/dataprep/index_slp.py` to index the replays into an sqlite database and put the extracted .slp files into their respective buckets.
    This may also take quite some time, an hour or two, depending on your CPU.
4. Delete the archives; we don't need them anymore, but we do need the disk space for the next steps.
5. Run `uv run fax/dataprep/slp_to_mds.py` to convert the .slp files into zstd-compressed MDS shards.
    These shards take up relatively little space, but will be unpacked in the next step.
6. Delete the raw .slp files; this frees up another \~180GB of space.
7. Run `uv run fax/dataprep/stats.py` to calculate the statistics for each of the input features in the datasets.
    These stats are needed for input normalization during training.
    Note that this script will unpack the .mds files, so make sure you have enough disk space; \~20x the size of the compressed sets should be safe.

\* _high-quality in the sense that the replays are from ranked netplay matches (diamond+, so the players are at least decent), and fox is well-represented in the dataset.
However; not all files were usable (corrupted files, unfinished games, etc.).
Luckily we have plenty of data to work with, so I could afford to be picky._

### Training

Once you have your data ready, you can start training the agents.
TODO: write more about training here.
