# Code for Paper "Diversity, Agreement, and Polarization in Elections" from IJCAI-23


The repository contains two experiment sets, **prefmap** (maps of preferences) and
**dap** (diversity, agreement, polarization). The experiment sets are located in
the directories with their own name.


# General 
This readme was prepared for running the code on Linux and python 3.8+.

In the dap experiment set directory there is directory `mapel`, which contains
the source code of the mapel library that we use for our experiments. 
The scripts of the dap experiment are prepared to exactly use our version of
mapel. Technically, at the beginning of each of these scripts, we add the
respective mapel directory manually to python's `sys.path`. If one has the
current distribution of mapel installed, one should make sure that python does
not try to use the installed copy instead of the one that we ship in this code
bundle.


# Detailed Experiment descriptions

Here we give slightly more details for each experiment separately. For each
section we assume that one is already in the respective code directory, e.g.,
for experiment dap, it is `dap/dap-code`.

The `_real_data` directory, contained in both experiments, contains the real
datasets that we used to generate our real-life based elections, as described in
the paper.

The `images` directory, contained in both experiments, stores the output
pictures of our scripts. It should not be removed!

The `experiments` directory, contained in both experiments, is used by mapel to store all
election data and computed values. It should not be removed! For experiment dap it also
stores the data that we use (see the description of this experiment for a more detailed
discussion on why we put this data in our code appendix instead of letting the user to
generate it).

## prefmap 

This code generates many preference map pictures among which there are Figures
1, 6, 7, and 9. 

Before running this part, install all required python packages listed in
`requirements.txt`. Then install module `permanent` by hand (`pip install permanent`).

The directory contains several python scripts, `requirements.txt`, the
`experiments`, `images`, and `_real_data` directories (the directories are
described in one of the above parts).

The directory contains the python scripts for generating given figures (as indicated by
the scripts' names) and the folders listed above.

Script `figure_9.py` runs quickly, while the other ones take tens of minutes on a
standard laptop.

## dap

Since we used the open-source mapel library we did sometimes some modifications
to it. Thus, in the dap set it is very important to have our codebase to run
the experiments. We ship the modified mapel directly in the dap directory and
all scripts are prepared to use exactly this local version.

Before running this part, install all required python packages listed in
`requirements.txt`. Then install module `permanent` by hand (`pip install permanent`).

The directory contains several python scripts, `requirements.txt`, the
`experiments`, `images`, and `_real_data` directories (the directories are
described in one of the above parts).

The `experiments` directory already contains all the data that we used in our
experiment together with all indices, distances, and other characteristics
computed. The data is there for convenience. The elections can be generated
from scratch as well as the distances and indices can be computed from scratch.
There are, however, three caveats to this. First, it takes a lot of computing
time (a week on a standard laptop). Second, to do so one needs to install the
local version of mapel using pip (go to directory `mapel` and, best in a
virtual environment, run `pip install .`) and then comment the lines at the
beginning of the scripts (those that will be listed later as those that one
runs for getting the pictures) that force the scripts to use the local version
of mapel (the respective lines are pointed by comments in the code). Third, you
need to uncomment the respective lines that generate the data (documented by
comments in the code).


We have three scripts, which can be run using `python filename.py`, that output the pictures.
 - (1) `diversity_map_8_96.py`
 - (2) `diversity_map_8_96_extended.py`
 - (3) `mallows_triangle_8_96.py`

All three scripts have parts that generate the data, and parts that compute the distances/indices
commented out (for the reasons described above).

Each of the scripts concerns one dataset that we did experiments on. The dataset from (1) is used in
the main body of the paper. The remaining two are in the appendix.

After each script's run, a big collection of pictures appear in the `images` directory. Among them,
one can find those presented in the paper. Beware to copy the pictures from this directory before
running the next script; otherwise, some pictures can be replaced by the later script run.

# Acknowledgments

This project is part of the [PRAGMA project](https://home.agh.edu.pl/~pragma/)
which has received funding from the [European Research Council
(ERC)](https://home.agh.edu.pl/~pragma/) under the European Union’s Horizon 2020
research and innovation programme ([grant agreement No
101002854](https://erc.easme-web.eu/?p=101002854)).



