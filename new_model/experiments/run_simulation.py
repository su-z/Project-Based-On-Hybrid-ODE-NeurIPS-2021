import argparse
import pickle

import numpy as np
import torch
import torch.optim as optim

import dataloader
import model
import sim_config
import training_utils

from typing import Optional, Union

device_t = Union["0", "1", "c"]
method_t = Union["expert", "neural", "hybrid"]

def run(
    seed: int,
    elbo: bool,
    device: device_t,
    eval_only: bool,
    init_path: Optional[str],
    data_path: str,
    sample: int, # The sample size
    data_config: sim_config.DataConfig,
    roche_config: sim_config.RochConfig,
    model_config: sim_config.ModelConfig,
    optim_config: sim_config.OptimConfig,
    eval_config: sim_config.EvalConfig,
    encoder_output_dim: Optional[int]=None,
    ablate:bool=False, # Remove somthing to see the changes
    arg_itr:Optional[int] = None,
    mix_ml_into_expert: bool = False,
    double_tol: bool = False,
    double_ode_net: bool = False
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:" + str(device) if device != "c" and torch.cuda.is_available() else "cpu")

    # data config
    n_sample = sample
    obs_dim = data_config.obs_dim # Observation dimension
    latent_dim = data_config.latent_dim # Latent space dimension
    action_dim = data_config.action_dim # "Action" (aka medical treatment) space dimension
    t_max = data_config.t_max # Maximum time
    step_size = data_config.step_size
    output_sigma = data_config.output_sigma # Standard deviation of output
    sparsity = data_config.sparsity # Proportion of values which are zero

    # optim config
    lr = optim_config.lr # Learning rate
    ode_method = optim_config.ode_method
    if arg_itr is None:
        niters: int = optim_config.niters
    else:
        niters: int = arg_itr
    batch_size: int = optim_config.batch_size
    test_freq: int = optim_config.test_freq # Do a test every test_freq steps
    early_stop = optim_config.early_stop # ?

    # with open('data/datafile.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    with open(data_path, "rb") as f: # Load data
        dg: dataloader.DataGeneratorRoche = pickle.load(f) # dg means "data generator"
        # The data generator simulates based on the assumptions and output data files.

    # with open('data/datafile_high_dim.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    dg.set_device(device)

    if not eval_only:
        dg.set_train_size(n_sample)

    print("Training with {} samples".format(n_sample))
    # dg = dataloader.DataGeneratorRoche(n_sample, obs_dim, t_max, step_size,
    #                                    roche_config, output_sigma, latent_dim, sparsity, device=device)
    # dg.generate_data()
    # dg.split_sample()

    # model config
    encoder_latent_ratio = model_config.encoder_latent_ratio
    if encoder_output_dim is None:
        if model_config.expert_only:
            encoder_output_dim = dg.expert_dim
        else:
            encoder_output_dim = dg.latent_dim

    if model_config.neural_ode:
        prior = None
        roche = False
        normalize = False
    else:
        prior = model.ExponentialPrior.log_density
        roche = True
        normalize = True

    best_on_disk = 1e9

    for i in range(optim_config.n_restart):
        encoder = model.EncoderLSTM(
            obs_dim + action_dim,
            int(obs_dim * encoder_latent_ratio),
            encoder_output_dim,
            device=device,
            normalize=normalize,
        )
        decoder = model.RocheExpertDecoder(
            obs_dim,
            encoder_output_dim,
            action_dim,
            t_max,
            step_size,
            roche=roche,
            method=ode_method,
            device=device,
            ablate=ablate,
            mix_ml_into_expert=mix_ml_into_expert,
        ) if not double_ode_net else model.DoubleRocheExpertDecoder(
            obs_dim,
            encoder_output_dim,
            action_dim,
            t_max,
            step_size,
            roche=roche,
            method=ode_method,
            device=device,
            ablate=ablate,
            mix_ml_into_expert=mix_ml_into_expert,
            double_tolarance=double_tol
        )
        

        vi = model.VariationalInference(encoder, decoder, prior_log_pdf=prior, elbo=elbo)

        if eval_only:
            break

        if init_path is not None:
            checkpoint = torch.load(init_path + vi.model_name)
            vi.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            vi.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        params = (
            list(vi.encoder.parameters())
            + list(vi.decoder.output_function.parameters())
            + list(vi.decoder.ode.ml_net.parameters())
            + list(vi.decoder.ode.expert_ml_mix_net_1.parameters())
            + list(vi.decoder.ode.expert_ml_mix_net_2.parameters())
            + list(vi.decoder.ode.expert_ml_mix_net_3.parameters())
            + list(vi.decoder.ode.expert_ml_mix_net_4.parameters())
        )

        optimizer = optim.Adam(params, lr=lr)

        res = training_utils.variational_training_loop(
            niters=niters,
            data_generator=dg,
            model=vi,
            batch_size=batch_size,
            optimizer=optimizer,
            test_freq=test_freq,
            path=model_config.path,
            best_on_disk=best_on_disk,
            early_stop=early_stop,
            shuffle=optim_config.shuffle,
        )
        vi, best_on_disk, training_time = res

    if eval_only:
        best_model = torch.load(path + vi.model_name)
        vi.encoder.load_state_dict(best_model["encoder_state_dict"])
        vi.decoder.load_state_dict(best_model["decoder_state_dict"])
        best_loss = best_model["best_loss"]
        print("Overall best loss: {:.6f}".format(best_loss))

    training_utils.evaluate(vi, dg, batch_size, eval_config.t0)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("PKPD simulation")
    parser.add_argument("--method", choices=["expert", "neural", "hybrid"], default="False", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--sample", default=1000, type=int)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--restart", default=1, type=int)
    parser.add_argument("--arg_itr", default=None, type=int)
    parser.add_argument("--eval", default="n", type=str)
    parser.add_argument("--elbo", default="y", type=str)
    parser.add_argument("--init", default=None, type=str)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--t0", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--data_config", default=None, type=str)
    parser.add_argument("--encoder_output_dim", default=None, type=int)
    parser.add_argument("--data_path", default="data/datafile_dose_exp.pkl", type=str)
    parser.add_argument("--ablate", default=False, type=bool)
    parser.add_argument("--mix_ml_into_expert", default=False, action="store_true")
    parser.add_argument("--double_tol", default=False, action="store_true")
    parser.add_argument("--double_ode_net", default=False, action="store_true")

    args = parser.parse_args()
    method: method_t = args.method
    seed: int = args.seed
    device: device_t = args.device
    path: Optional[str] = args.path
    sample: int = args.sample
    restart: int = args.restart
    eval_only: bool = args.eval == "y"
    init_path: Optional[str] = args.init
    batch_size: int = args.batch_size
    data_path: str = args.data_path
    dc: Optional[str] = args.data_config # data config
    elbo: bool = args.elbo == "y"
    encoder_output_dim: Optional[int] = args.encoder_output_dim
    arg_itr: Optional[int] = args.arg_itr # ?
    mix_ml_into_expert: bool = args.mix_ml_into_expert
    double_tol: bool = args.double_tol
    double_ode_net: bool = args.double_ode_net

    if dc == "dim8":
        data_config = sim_config.dim8_config
    elif dc == "dim12":
        data_config = sim_config.dim12_config
    else:
        data_config = sim_config.DataConfig(n_sample=sample)
    roche_config = sim_config.RochConfig()
    if method == "expert":
        model_config = sim_config.ModelConfig(expert_only=True, path=path)
    elif method == "neural":
        model_config = sim_config.ModelConfig(neural_ode=True, path=path)
    elif method == "hybrid":
        model_config = sim_config.ModelConfig(path=path)

    # todo: try no shuffle
    optim_config = sim_config.OptimConfig(shuffle=False, n_restart=restart, batch_size=batch_size, lr=args.lr)
    eval_config = sim_config.EvalConfig(t0=args.t0)
    run(
        seed,
        elbo,
        device,
        eval_only,
        init_path,
        data_path,
        sample,
        data_config,
        roche_config,
        model_config,
        optim_config,
        eval_config,
        encoder_output_dim,
        args.ablate,
        arg_itr,
        mix_ml_into_expert=mix_ml_into_expert,
        double_tol = double_tol,
        double_ode_net=double_ode_net
    )
