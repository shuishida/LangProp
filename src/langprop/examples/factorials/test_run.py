import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from langprop.examples.factorials.dataset import FactorialDataset
from langprop.module import LPModule, RunConfig
from langprop.trainer import LPTrainer


class Trainer(LPTrainer):
    def preprocess(self, data):
        index, label = data
        return (index,), None, label

    def score(self, result, labels) -> float:
        return float(result == labels)


def test_run(run_name: str = "factorial", batch_size: int = 128, epochs=1):
    train_dataset = FactorialDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: x)
    model = LPModule.from_template(name="get_factorial", root=Path(__file__).parent)
    trainer = Trainer(model, RunConfig(run_name=run_name))
    trainer.fit(train_loader, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="factorial")
    args = parser.parse_args()

    test_run(args.run_name)
