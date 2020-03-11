

## temperature is simply to reflect the fact that loads is correlated spatially, and so is the
## real "temperature". It is not the real temperature.
temperature_noise = conso.generate_coarse_noise(params, 'temperature')
if compute_renewables:
    solar_noise = ns.generate_coarse_noise(params, 'solar')
    long_scale_wind_noise = ns.generate_coarse_noise(params, 'long_wind')
    medium_scale_wind_noise = ns.generate_coarse_noise(params, 'medium_wind')
    short_scale_wind_noise = ns.generate_coarse_noise(params, 'short_wind')