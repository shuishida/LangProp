import json
from pathlib import Path

import yaml

from agents.navigation.local_planner import RoadOption
from data_agent.data_io import deserialize_data
from expert.detect_hazards import calc_dist_to_hazards

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("index")
    args = parser.parse_args()

    data_dir = Path(args.data)
    index = int(args.index)

    with open(Path(__file__).parent.parent / "config" / "data_collect.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(data_dir / "agent_info" / ("%04d.json" % index)) as f:
        ego_info = deserialize_data(json.load(f))["ego_info"]

    with open(data_dir / "actors_info" / ("%04d.json" % index)) as f:
        actors_info = deserialize_data(json.load(f))

    command = RoadOption.STRAIGHT

    print(calc_dist_to_hazards(ego_info, actors_info, config))
