import argparse
from datetime import datetime
from pathlib import Path
import os

from torch.utils.data import DataLoader

from langprop.module import LPModule, RunConfig
from lmdrive.dataset import DrivingLMDataset
from lmdrive.trainer import LMDriveTrainer


DATA_ROOT_BASE = Path(os.environ["DATA_ROOT_BASE"])


def eval_lm_policy(train_path: str, test_path: str, run_name: str = "", root_dir: Path = "", train_batch_size: int = 1000, test_batch_size: int = 1000,
                   infraction_lookahead: int = 1, infraction_penalty: int = -10, epochs=20):
    train_dataset = DrivingLMDataset(train_path, load_jpg=(), load_npy=(), infraction_lookahead=infraction_lookahead)
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    test_dataset = DrivingLMDataset(test_path, load_jpg=(), load_npy=(), infraction_lookahead=infraction_lookahead)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    model = LPModule.from_template(name="predict_speed_and_steering", root=Path(__file__).parent / "models")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + run_name
    trainer = LMDriveTrainer(model, RunConfig(run_name=run_name,
                                              root_dir=root_dir / run_name,
                                              exception_score=-10, trackers_config={"main": {"priority_decay_rate": 0.0}},
                                              n_responses=2, n_top_choices=2, max_keep=16, save_config=dict(infraction_penalty=infraction_penalty)))
    trainer.fit(train_loader, test_loader, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=DATA_ROOT_BASE / "langprop/expert/training/offline_dataset")
    parser.add_argument("--testing", default=DATA_ROOT_BASE / "langprop/expert/testing/offline_dataset")
    parser.add_argument("--root_dir", default=DATA_ROOT_BASE / "langprop/lmdrive_offline")
    parser.add_argument("--run_name", default="predict_speed_and_steering")
    args = parser.parse_args()

    eval_lm_policy(args.training, args.testing, args.run_name, args.root_dir)
