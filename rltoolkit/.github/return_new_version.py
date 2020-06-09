import argparse


def get_new_version(old_version: str) -> str:
    version = old_version.split(".")
    patch = int(version[-1])
    version[-1] = str(patch + 1)
    version = ".".join(version)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", metavar="V", type=str)
    args = parser.parse_args()
    version = args.version.split(".")
    patch = int(version[-1])
    version[-1] = str(patch + 1)
    version = ".".join(version)
    print(version)
