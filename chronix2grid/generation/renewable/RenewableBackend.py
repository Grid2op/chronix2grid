from .generate_solar_wind import main


class RenewableBackend:

    def __init__(self, out_path, seed, params, loads_charac, res_config_manager, write_results):
        self.write_results = write_results
        self.res_config_manager = res_config_manager
        self.loads_charac = loads_charac
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        load_weekly_pattern = self.res_config_manager.read_specific()
        return main(self.out_path, self.seed, self.params, self.loads_charac, load_weekly_pattern, self.write_results)
