from .generate_loss import main


class LossBackend:

    def __init__(self, input_folder, scenario_folder_path,
                                     load, prod_solar, prod_wind,
                                     params, loss_config_manager, write_results = True):
        self.write_results = write_results
        self.loss_config_manager = loss_config_manager
        self.input_folder = input_folder
        self.params = params
        self.scenario_folder_path = scenario_folder_path
        self.load = load
        self.prod_solar = prod_solar
        self.prod_wind = prod_wind

    def run(self):
        self.loss_config_manager.validate_configuration()
        params_loss = self.loss_config_manager.read_configuration()
        return main(self.input_folder, self.scenario_folder_path,
                             self.load, self.prod_solar, self.prod_wind,
                             self.params, params_loss, write_results=self.write_results)
