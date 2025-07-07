import subprocess

def run_dlboost_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-march=icelake-server",
                "-O3",
                file_name,
                "-o",
                so_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_cpp_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["g++", "-shared", "-fPIC", "-O3", file_name, "-o", so_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_mlu_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "cncc",
                "-shared",
                "-fPIC",
                "--bang-mlu-arch=mtp_592",
                "-O3",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_cuda_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "nvcc",
                "-Xcompiler",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_hip_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "hipcc",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def search_op_info(op_info_dict, file_name):
    matching_items = []
    print('file_name: ", file_name)
    for item in op_info_dict:
	current_file_name = f"{item['op_name']}_{item['op_args']}_{item['input_shape']}"
	if current_file_name == file_name:
	    matching_items.append(item)
    if len(matching_items) != 1:
	raise ValueError("Error: Multiple Matches!')
    return matching_items[0]


def parse_code(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
	file_content = file.read()
    return file_content
