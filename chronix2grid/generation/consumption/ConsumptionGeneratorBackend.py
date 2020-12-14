from .generate_load import main


class ConsumptionGeneratorBackend:

    def __init__(self, out_path, seed, params, loads_charac, load_config_manager, write_results):
        self.write_results = write_results
        self.load_config_manager = load_config_manager
        self.loads_charac = loads_charac
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        load_weekly_pattern = self.load_config_manager.read_specific()
        return main(self.out_path, self.seed, self.params, self.loads_charac, load_weekly_pattern, self.write_results)
