import carla

WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "ClearNight": carla.WeatherParameters(5.0, 0.0, 0.0, 10.0, -1.0, -90.0, 60.0, 75.0, 1.0, 0.0),
    "CloudyNight": carla.WeatherParameters(60.0, 0.0, 0.0, 10.0, -1.0, -90.0, 60.0, 0.75, 0.1, 0.0),
    "WetNight": carla.WeatherParameters(5.0, 0.0, 50.0, 10.0, -1.0, -90.0, 60.0, 75.0, 1.0, 60.0),
    "WetCloudyNight": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, -1.0, -90.0, 60.0, 0.75, 0.1, 60.0),
    "SoftRainNight": carla.WeatherParameters(60.0, 30.0, 50.0, 30.0, -1.0, -90.0, 60.0, 0.75, 0.1, 60.0),
    "MidRainyNight": carla.WeatherParameters(80.0, 60.0, 60.0, 60.0, -1.0, -90.0, 60.0, 0.75, 0.1, 80.0),
    "HardRainNight": carla.WeatherParameters(100.0, 100.0, 90.0, 100.0, -1.0, -90.0, 100.0, 0.75, 0.1, 100.0),
}
WEATHERS_IDS = list(WEATHERS)
