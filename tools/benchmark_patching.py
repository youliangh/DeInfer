import os
import click

from typing import Union, List


def read_all(filepath: Union[str, os.PathLike]) -> List[str]:
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines


@click.command()
@click.option("--dir", type=str, required=True, help="Patch all .py file in this directory")
def patch(**kwargs):
    file_dir = kwargs["dir"]

    # Check if the directory exists
    if not os.path.exists(file_dir):
        raise RuntimeError("Invalid directory path.")

    # Get all the .py files in the directory
    py_files = [f for f in os.listdir(file_dir) if f.endswith(".py")]

    for filename in py_files:
        filepath = os.path.join(file_dir, filename)

        # Look for the first function definition clause
        lines = read_all(filepath)
        line_number = 0
        for idx, line in enumerate(lines):
            if "if __name__ ==" in line:
                line_number = idx

        # Insert the model registration code (in reversed order)
        lines.insert(line_number, "ModelRegistry.register_model('SVDLlamaForCausalLM', SVDLlamaForCausalLM)\n\n")
        lines.insert(line_number, "from effide.vllm_plugin.model_executor.models.llama import SVDLlamaForCausalLM\n")
        lines.insert(line_number, "from vllm import ModelRegistry\n")
        lines.insert(line_number, "ModelRegistry.register_model('SVDOPTForCausalLM', SVDOPTForCausalLM)\n\n")
        lines.insert(line_number, "from effide.vllm_plugin.model_executor.models.opt import SVDOPTForCausalLM\n")
        lines.insert(line_number, "from vllm import ModelRegistry\n")

        # Write the patched code back to the file
        with open(filepath, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    patch()
