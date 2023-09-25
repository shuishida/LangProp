from types import SimpleNamespace
import carla

from data_collect.helpers.agent_wrapper import DataCollectAgentWrapper
from data_collect.helpers.scenario_manager import load_scenario
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_indexer import RouteIndexer

from leaderboard.utils.statistics_manager import StatisticsManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class TestWorld(LeaderboardEvaluator):
    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        print("\n\033[1m========= Preparing {} =========".format(config.name))
        print("> Setting up the agent\033[0m")
        # Set up the user's agent, and the timer to avoid freezing the simulation
        agent_class_name = getattr(self.module_agent, 'get_entry_point')()
        self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
        config.agent = self.agent_instance

        # Check and store the sensors
        if not self.sensors:
            self.sensors = self.agent_instance.sensors()
            track = self.agent_instance.track

            DataCollectAgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
        self._prepare_ego_vehicles(config.ego_vehicles, False)
        scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)

        # Night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

        # Load scenario and run it
        load_scenario(self.manager, scenario, self.agent_instance, config.repetition_index)

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, 1)

        # setup
        config = route_indexer.next()

        # run
        self._load_and_run_scenario(args, config)


def setup_test_world(port=2000, tm_port=8000, agent="./src/baselines/fuel/expert/auto_pilot.py",
                     agent_config="./src/baselines/fuel/config/data_collect.yaml"):
    args = SimpleNamespace()
    args.host = "localhost"
    args.port = port
    args.trafficManagerPort = tm_port
    args.trafficManagerSeed = 0
    args.debug = 0
    args.timeout = 60.0
    args.routes = "./leaderboard/data/routes_training.xml"
    args.scenarios = "./leaderboard/data/all_towns_traffic_scenarios_public.json"
    args.agent = agent
    args.agent_config = agent_config
    args.track = "SENSORS"
    args.checkpoint = "./debug_checkpoint.json"

    statistics_manager = StatisticsManager()
    test_world = TestWorld(args, statistics_manager)
    test_world.run(args)

    ego_vehicle = CarlaDataProvider.get_hero_actor()

    return test_world.world, ego_vehicle


if __name__ == "__main__":
    world, agent = setup_test_world()
    print(world, agent)
