from pathlib import Path
from langprop.prompt import extract_block


def test_policy():
    with open(Path(__file__).parent / "checkpoint.txt", "r") as f:
        response = f.read()

    py_code = extract_block(response)
    exec(py_code)
    policy = locals()["get_factorial"]

    dirpath = Path(__file__).parent

    for i in range(10):
        result = policy(i)
        print(i, result)


if __name__ == "__main__":
    test_policy()
