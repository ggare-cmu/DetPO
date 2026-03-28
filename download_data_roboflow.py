from rf100vl import download_rf20vl_fsod
import argparse

def main():
    parser = argparse.ArgumentParser(description='download datasets for object detection')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='dir where data is there')
    args = parser.parse_args()
    download_rf20vl_fsod(path=args.data_dir)


if __name__ == "__main__":
    main()