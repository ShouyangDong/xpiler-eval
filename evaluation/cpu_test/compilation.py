import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from evaluation.macros import CPP_MACROS as macro
from evaluation.utils import run_cpp_compilation as run_compilation


def compile_file(file_name):
    base_name = os.path.basename(file_name)

    with open(file_name, "r") as f:
        code = f.read()

    code = macro + code
    bak_file_name = file_name.replace(".cpp", "_bak.cpp")

    with open(bak_file_name, mode="w") as f:
        f.write(code)

    so_name = base_name.replace(".cpp", ".so")
    so_name = os.path.join(
        os.path.join(os.getcwd(), "evaluation/data/cpp_code_test/"), so_name
    )

    success, output = run_compilation(so_name, bak_file_name)
    os.remove(bak_file_name)

    if success:
        subprocess.run(["rm", so_name])
        return True
    else:
        print(output)
        return False


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "evaluation/data/cpp_code_test/*.cpp")
    )

    # Use ThreadPoolExecutor to execute file compilation in parallel.
    counter = 0
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_file, files), total=len(files))
        )

    # Count the number of files successfully compiled.
    counter = sum(results)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************CPP Compilation successful rate ",
        counter / len(files),
    )
