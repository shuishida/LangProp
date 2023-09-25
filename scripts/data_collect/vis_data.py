import argparse

from data_agent.visualize import create_debug_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    create_debug_images(args.data)
