import os


def get_current_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_abs_path(file_path: str) -> str:
    return os.path.join(get_current_dir(), file_path)


def get_stop_words(file_path: str) -> list[str]:
    with open(get_abs_path(file_path), "r") as file:
        return file.read().split()
