import os
import re


def remove_cpp_comments(code):
    """Remove C++ style // comments, while preserving // inside strings.

    Uses regex to match either string literals or comments, and only removes
    the comment parts.
    """
    # Regex pattern to match:
    #   - Double-quoted strings (with escaped characters support)
    #   - Single-quoted character literals (with escaped characters support)
    #   - // comments (from // to end of line)
    pattern = r"""
        "(?:\\.|[^"\\])*"        # Match double-quoted string (handles escapes)
        |                         # OR
        '(?:\\.|[^'\\])*'         # Match single-quoted character (handles escapes)
        |                         # OR
        //.*$                     # Match // to end of line (comment)
    """

    def replace(match):
        matched = match.group(0)
        # If the match starts with //, remove it (it's a comment)
        if matched.startswith("//"):
            return ""
        # Otherwise, it's a string or char literal, keep it unchanged
        return matched

    # Apply substitution with MULTILINE and VERBOSE flags
    # MULTILINE: ^ and $ match line boundaries
    # VERBOSE: Allows whitespace and comments in the regex
    result = re.sub(pattern, replace, code, flags=re.MULTILINE | re.VERBOSE)
    return result


def process_cu_file(file_path):
    """Read a .cu or .cpp file, remove all // comments, and write back to the
    file.

    Preserves // inside strings and character literals.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = remove_cpp_comments(content)

        # Write the modified content back to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


def process_directory(directory, extensions=".cu"):
    """Recursively walk through the given directory and process files with the
    specified extension(s).

    Args:
        directory (str): Root directory to traverse.
        extensions (str or tuple): File extension(s) to process, e.g. ".cu" or (".cu", ".cpp").
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                process_cu_file(file_path)


# === Usage Example ===
if __name__ == "__main__":
    target_dir = "./KernelBench/BANG"  # Change to your target directory
    process_directory(target_dir, ".mlu")  # Process all .mlu files
