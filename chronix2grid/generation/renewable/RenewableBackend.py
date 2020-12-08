from .generate_solar_wind import main


class RenewableBackend:

    def __init__(self, out_path, seed, params, loads_charac, load_weekly_pattern, write_results):
        self.write_results = write_results
        self.load_weekly_pattern = load_weekly_pattern
        self.loads_charac = loads_charac
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        main(self.out_path, self.seed, self.params, self.loads_charac, self.load_weekly_pattern, self.write_results)
