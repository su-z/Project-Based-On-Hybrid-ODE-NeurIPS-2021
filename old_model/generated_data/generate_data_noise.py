import argparse
import pickle

import torch

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("generate data noise")
    parser.add_argument("--noise_level", default=0.4, type=float)
    parser.add_argument("--mix_ml_into_expert", default=False, action='store_true')

    args = parser.parse_args()
    noise_level: float = args.noise_level
    mix_ml_into_expert: bool = args.mix_ml_into_expert

    dg = pickle.load(open("data/datafile_dose_exp_test.pkl", "rb"))
    seed = 666
    torch.manual_seed(seed)
    noise = torch.randn_like(dg.measurements) * (noise_level - 0.2)
    with torch.no_grad():
        dg.measurements = dg.measurements + noise
        dg.split_sample()

    with open("data/datafile_dose_noise_{}.pkl".format(noise_level), "wb") as f:
        pickle.dump(dg, f)
