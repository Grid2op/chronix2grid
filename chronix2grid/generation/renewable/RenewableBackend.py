from .generate_solar_wind import main
from .generate_solar_wind_gan import main_gan


class RenewableBackend:
    """
    Backend that generates solar and wind production chronics with a spatiotemporal correlated noise model.
    It takes into account specific parameters about the grid, the solar and wind power plants, the regional setting...
    See in documentation *Description of implemented models* for detailed information about the model

    Attributes
    ----------
    out_path: ``str``
        path to output folder for generated chronics
    seed: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys "dt", "planned_std", "Lx", "Ly", "dx_corr", "dy_corr", "short_wind_corr",
        "medium_wind_corr", "long_wind_corr", "solar_corr", "smoothdist", "std_short_wind_noise", "std_solar_noise",
        "std_medium_wind_noise", "std_long_wind_noise", "year_solar_pattern"
    prods_charac: :class: ``pandas.DataFrame``
        data frame with characteristics on wind and solar generators/power plants of the model
    res_config_manager: :class: ``chronix2grid.config.ResConfigManager``
        config manager used to load specific patterns used for the model (solar_pattern)
    write_results: ``bool``
    """
    def __init__(self, out_path, seed, params, loads_charac, res_config_manager, write_results):
        self.write_results = write_results
        self.res_config_manager = res_config_manager
        self.loads_charac = loads_charac
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        """
        Runs the generation model in ``chronix2grid.generation.renewable.generate_solar_wind`` and writes chronics
        """
        solar_pattern = self.res_config_manager.read_specific()
        return main(self.out_path, self.seed, self.params, self.loads_charac, solar_pattern, self.write_results)

class RenewableBackendGAN:
    """
    Backend that generates solar and wind production chronics with a model based on trained neural networks (Generative
    Adversarial Networks or GAN).
    It takes general parameters about generation and specific parameters about how the networks have been trained.
    It uses tensorflow checkpoints in order to load and replay network objects

    Attributes
    ----------
    out_path: ``str``
        path to output folder for generated chronics
    seed: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys "dt", "planned_std", "model_name", "batch_size",
        "n_gens", "n_timesteps", "n_events","dim_inputs", "mu", "sigma", "network_folder".
        It has normally been read by a :class: ``chronix2grid.config.ResConfigManagerGan`` instance
    prods_charac: :class: ``pandas.DataFrame``
        data frame with characteristics on wind and solar generators/power plants of the model
    res_config_manager: :class: ``chronix2grid.config.ResConfigManagerGan``
    write_results: ``bool``
    """
    def __init__(self, out_path, seed, params, prods_charac, res_config_manager, write_results):
        self.write_results = write_results
        self.prods_charac = prods_charac
        self.res_config_manager = res_config_manager
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        """
        Runs the generation model in ``chronix2grid.generation.renewable.generate_solar_wind_gan`` and writes chronics
        """
        return main_gan(self.out_path, self.seed, self.params, self.prods_charac, self.write_results)