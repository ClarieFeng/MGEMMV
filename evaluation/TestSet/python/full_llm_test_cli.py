import argparse
import time
from pathlib import Path

from origin_llm_test import SCRIPT_DIR, process_files


def main():
    parser = argparse.ArgumentParser(description="CLI wrapper around full_llm_test.py with custom dataset paths.")
    parser.add_argument("--base-directory", default="..", help="Dataset root relative to this script. Default: ..")
    parser.add_argument("--module-min", type=int, default=1, help="Minimum module id. Default: 1")
    parser.add_argument("--module-max", type=int, default=27, help="Maximum module id. Default: 27")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of tests per module. Default: 5")
    args = parser.parse_args()

    base_directory = str((SCRIPT_DIR / args.base_directory).resolve())

    start_time = time.time()
    process_files(base_directory, args.module_min, args.module_max, args.num_tests)
    end_time = time.time()

    print(f"\n[Total Runtime] {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
