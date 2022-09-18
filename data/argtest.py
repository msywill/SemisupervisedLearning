import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test', default="input", type=str, help='input files')
    parser.add_argument('--output_path', default="output", type=str, help='result dir')
    args = parser.parse_args()
    print(args.test)
    print(args.output_path)