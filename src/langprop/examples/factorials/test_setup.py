from pathlib import Path

from langprop.lm_api import call_chatgpt, template_to_query


def test_setup():
    dirpath = Path(__file__).parent

    with open(dirpath / "get_factorial/setup.txt", "r") as f:
        template = f.read()

    setup_prompt, setup_query = template_to_query(template, function_name="get_factorial")
    print(setup_prompt)
    results = call_chatgpt(setup_query)
    print(results)
    with open(dirpath / "checkpoint.txt", "w") as f:
        for result in results:
            f.write(result)
            f.write("\n")


if __name__ == "__main__":
    test_setup()
