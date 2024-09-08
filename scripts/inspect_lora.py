from safetensors import safe_open
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parse lora.safetensors file")
    parser.add_argument("file_path", type=str, help="Path to lora.safetensors file")
    parser.add_argument(
        "--print-values", action="store_true", help="Print values in addition to keys"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = args.file_path
    print_values = args.print_values

    tensors = {}
    with safe_open(file_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            print(f"Key: {k}")
            if print_values:
                print(f"Value: {tensors[k]}")
                print(f"Value shape: {tensors[k].shape}")
                print(f"Value dtype: {tensors[k].dtype}")
            print("------------------------")


if __name__ == "__main__":
    main()
