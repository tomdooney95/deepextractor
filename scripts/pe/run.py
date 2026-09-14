import bilby 
import numpy as np
import argparse
import json
from bilby.core.utils import logger
import pickle
import utils
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This script will be used to perform parameter estimation on a user specified set of frame files. 
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type = int,
                        default = 2024,
                        help = "Seed for the random number generator")
    parser.add_argument('--minimum-frequency',
                        type = float,
                        default=20.0,
                        help = "Minimum frequency of the waveform")
    parser.add_argument('--label',
                        type = str,
                        default = 'test',
                        help = "Label to give to the run")
    parser.add_argument('--outdir',
                        type = str,
                        default = 'delete_me',
                        help = "Output directory for the run")
    parser.add_argument('--sampling-frequency',
                        type = float,
                        default = 4096.0,
                        help = "Sampling frequency of the data")
    parser.add_argument('--injection-file',
                        type = str,
                        default = 'pe_cases_n1.pkl',
                        help = "File containing the injection parameters")
    parser.add_argument('--injection-label',
                        type = str,
                        default = 'GW150914')
    parser.add_argument('--debug',
                        type = int,
                        default = 1)
    parser.add_argument('--type', type = str, default = "control", 
                        help = "Type of data used for parameters estimation: control, prediction, dirty." \
                        "control: No glitch introduced" \
                        "prediction: prediction of DeepExtractor (i.e. glitch removed data)" \
                        "dirty: Leave glitch inside the data")
    return parser.parse_args()


def setup_interferometers(injection_file, injection_parameters, minimum_frequency, sampling_frequency, duration, data_type):
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    asd = np.genfromtxt('./aLIGO_O4_high_asd.txt', unpack=True)
    ## NOTE: need to make sure the minimum/maximum frequency is set correctly
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.sampling_frequency = sampling_frequency

    start_time = injection_parameters['geocent_time'] - 3.5 ####  injection_parameters['geocent_time'] - duration + 2

    strain_keys = {
        'prediction': ('deglitched_h1',   'deglitched_l1',   'DeepExtractor prediction'),
        'control': ('whitened_data_h1', 'whitened_data_l1',  'Only signal+noise'),
        'dirty': ('glitchy_h1','glitchy_l1', 'glitch frames'),
    }
    if data_type not in strain_keys:
        print('args.type option not found')
        exit()

    h1_key, l1_key, description = strain_keys[data_type]
    logger.info(f'Setting strain data from {description}')
    collect_strain = {
        'H1': utils.white_time_domain_strain_to_colored_frequency_domain_strain(injection_file[h1_key], asd=asd),
        'L1': utils.white_time_domain_strain_to_colored_frequency_domain_strain(injection_file[l1_key], asd=asd),
    }

    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            collect_strain[ifo.name], start_time=start_time,
            duration=duration, sampling_frequency=sampling_frequency)

    return ifos


def setup_waveform_generator(injection_parameters, minimum_frequency, sampling_frequency):
    waveform_arguments = dict(
        waveform_approximant='IMRPhenomXPHM',
        reference_frequency=50,
        minimum_frequency=minimum_frequency
    )
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=4, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=injection_parameters,
        waveform_arguments=waveform_arguments
    )


def setup_priors(injection_parameters):
    priors = bilby.gw.prior.BBHPriorDict()

    priors["mass_ratio"] = bilby.core.prior.Uniform(minimum=0.1, maximum=1., name='mass_ratio', latex_label='$q$')
    priors["chirp_mass"] = bilby.core.prior.Uniform(10, 100, name='chirp_mass')

    if injection_parameters['chirp_mass'] < priors['chirp_mass'].minimum or injection_parameters['chirp_mass'] > priors['chirp_mass'].maximum:
        logger.info('Chirp mass is not withitn the prior range. Exiting....')
        logger.info(f"Prior range {priors['chirp_mass'].minimum} {priors['chirp_mass'].maximum}")
        exit()

    priors.pop('mass_1')
    priors.pop('mass_2')

    priors['geocent_time'] = bilby.core.prior.Uniform(injection_parameters['geocent_time']-0.1, injection_parameters['geocent_time']+0.1, name="geocent_time")

    priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(minimum=1.,
                                                                       maximum=5000.,
                                                                       name='luminosity_distance',
                                                                       latex_label='$D_L$')

    priors['mass_1'] = bilby.gw.prior.Constraint(name='mass_1', minimum=20, maximum=100)
    priors['mass_2'] = bilby.gw.prior.Constraint(name='mass_2', minimum=20, maximum=100)

    return priors


def run_debug(likelihood, priors, injection_parameters, outdir, label):
    likelihood.noise_log_likelihood()

    likelihood.parameters = injection_parameters

    log_likelihood_ratio_at_injected_value = likelihood.log_likelihood_ratio()

    logger.info(f'Log likleihood ratio at the injected value {log_likelihood_ratio_at_injected_value}')
    logger.info(f'Noise log likleihood {likelihood.noise_log_likelihood()}')

    key = 'chirp_mass'
    N = int(1e3)
    prior_samples = priors[key].sample(N)
    collect_likelihood = []
    injection_parameters.pop('mass_1')
    injection_parameters.pop('mass_2')

    for xx in tqdm(prior_samples):
        sample_parameters = injection_parameters.copy()
        sample_parameters[key] = xx

        likelihood.parameters = sample_parameters
        collect_likelihood.append(likelihood.log_likelihood_ratio())

    fig, ax = plt.subplots(1, 1)
    ax.scatter(prior_samples, collect_likelihood)
    fig.savefig(f"{outdir}/{label}_{key}_likelihood.pdf")

    return injection_parameters


def setup_injection_parameters(injection_file_path, injection_label):
    event_data = pickle.load(open(injection_file_path, 'rb'))[injection_label][0]
    injection_parameters = event_data['injection_parameters']
    injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'],
                                                                                             injection_parameters['mass_2'])
    injection_parameters['mass_ratio'] = injection_parameters['mass_2']/injection_parameters['mass_1']

    return event_data, injection_parameters


def main():
    args = parse_args()
    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)

    strain_dictionary, injection_parameters = setup_injection_parameters(args.injection_file, args.injection_label)

    duration = 4.0

    logger.info(f"Injection parameters are: {injection_parameters}")

    waveform_generator = setup_waveform_generator(injection_parameters, args.minimum_frequency, args.sampling_frequency)

    ifos = setup_interferometers(strain_dictionary, injection_parameters, args.minimum_frequency, args.sampling_frequency, duration, args.type)

    priors = setup_priors(injection_parameters)

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator)

    if args.debug:
        injection_parameters = run_debug(likelihood, priors, injection_parameters, args.outdir, args.label)

    logger.info('Starting sampling....')
    likelihood.parameters = {}
    result = bilby.run_sampler(likelihood = likelihood, priors = priors, npool = 32, verbose = True,
         sampler = 'dynesty', nlive=1000, outdir = args.outdir, label = args.label, naccept = 60,
             check_point_plot = True, check_point_delta_t = 3600, dlogz = 0.1,
       injection_parameters = injection_parameters,  print_method = 'interval-60', sample = 'acceptance-walk')

    result.plot_corner(priors = True)


if __name__=="__main__":
    main()
