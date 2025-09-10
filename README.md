# FAX

<!-- markdownlint-disable MD013 -->

The aim of this project is to investigate the age-old question that has plagued the competitive SSBM scene for ages;

_Who is cooler: the generalist or the specialist?_

To answer this question, we train two AI agents using the same approach (imitation learning with a GPT), only using wildly different datasets.
The specialist will train on replays that feature at least one fox, while the generalist will not see a single fox over the course of their entire training regiment.

## Instructions

If you want to play around with this project, please follow these instructions to set up your own dev environment.
When you get stuck, don't hesitate to create an issue!
I'll probably be working on this after my thesis is done in my free time anyway, so I'll gladly help you out.

### Repo setup

This project uses [uv](https://github.com/astral-sh/uv) with python 3.11.
Run `uv sync` to automatically set up a .venv and pull all the dependencies correctly (and fast).
If you don’t have uv installed and don't want to massively improve your python experience, you can also still do the usual:

```sh
python3 -m venv .venv  # make sure your python binary is 3.11 (or higher)
source .venv/bin/activate
pip install -e .  # pulls deps from pyproject.toml in editable mode
```

In the top-level of this repo you'll find [paths.toml](paths.toml).
This is where you should set the paths to your dolphin/slippi binary, your SSBM .iso file, and your datasets.
Later on, I will expand this to also house training hyperparameters and other config options.

### External dependencies

To install and configure Slippi please see [slippi.gg](https://slippi.gg) for instructions.
In this project I've used vladfi's headless build, which also enables fastforward mode, which is hugely useful.
The prebuilt binary can be found [on his Ishiiruka fork](https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.1.0/Slippi_Online-x86_64-ExiAI.AppImage).
You can also find a mainline build [on his dolphin fork](https://github.com/vladfi1/dolphin/releases/download/slippi-nogui-v0.1.0/Slippi_Netplay_Mainline_NoGui-x86_64.AppImage).

You will need to burn your physical SSBM disk to build an .iso file for any of this to work, or you could try some other way to obtain the .iso.

### Data

I was provided a dataset of a little over 116k high-quality* replays by a member of the slippi community through their discord.
The archive (\~92GB) unzipped to \~441GB of raw .slp files.
I then created three different MDS datasets from the raw .slp data:

```tree
mds
.
├── nonfox  (generalist training data)
│   ├── train
│   └── val
├── onefox  (specialist training data)
│   ├── train
│   └── val
└── twofox  (only used for testing at the end)
```

For me the pipeline was as follows:

```sh
cd ~/Data
# Download the ranked anonymized dataset (~92GB, ~1hr)
wget --show-progress -O ./ranked-anonymized-1.zip <link to archive>
# Unzip the dataset (~441GB, ~1hr)
unzip -q ranked-anonymized-1.zip -d ~/Data/ranked-anonymized
# Create the zstd compressed mds datasets (~40GB, ~2hr)  <!-- TODO: verify -->
cd ~/Developer/own/fax
uv run fax/slp_to_mds.py ~/Data/ranked-anonymized
```

\* _high-quality meaning that the replays are from ranked netplay matches (diamond+, so the players are at least decent), and fox is well-represented in the dataset.
However; not all files parsed correctly, and over a thousand games were unfinished (disconnections, crashes, etc.), so I had to filter those out.
I ended up with a little under 115k usable replays._
