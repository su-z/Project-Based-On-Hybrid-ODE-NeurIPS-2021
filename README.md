# Investigation on Mixing Expert ODEs and Neural ODEs

This repository is a fork of [https://github.com/vanderschaarlab/Hybrid-ODE-NeurIPS-2021](). It is the code I have worked on during a two-week project at the the van der Shaar Lab, at Cambridge.

Within the limited two weeks, there isn't much time for software engineering. I did add some comments, but sorry if you find anything unclear.

The repo contains three directories.

* The directory `old_model` contains the original model of the repo above, but I have changed the data generation programs so that they generate slightly different data, which is studied in my report.
* The directory `new_model` contains the model I describe in section 4 of my writeup. It allows expert variables to depend on latent variables.

These two directories can be merged easily, but I have intentionally kept them separate, to allow running the new and old models in parallel without interfering with each other (to avoid mistake of, for instance, accidently changing files that should be read-only during training).

## How to run the code

The README of the original repo can be found in `old_model/README.md`. When running `old_model`, firstly `cd` into the directory. You must use `bash` or any shell that supports commands like `readarray`. A shell like `sh` which just satisfy the minimum required by POSIX does not work. On macOS, you might need to avoid using the `bash` shipped with the OS and install a new one.

If you follow the instruction in the old readme, you should get the same result as the old repo.   However, if you do   `bash experiments/create_data.sh --mix_ml_into_expert `, (**NOTE** the additional flag added), then you will generate data where expert variables have weak dependence on latent variables. (The flag `--mix_ml_into_expert `will **not** work in `old_model` for shell scripts **other** **than** `create_data.sh`.)

In the `new_model/experiments` directory, the following shell scripts **should** be runned with `--mix_ml_into_expert`:

```shell
bash experiments/Fig3.sh --mix_ml_into_expert
```

### Parallel ODEs

The class `DoubleRocheExpertDecoder` for running two ODE in parallel to improve performance is written in `new_model/model.py`, but unfortunately we do not have enough time to run and test it. It is there for future work.
