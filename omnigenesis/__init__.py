def main():
    from .app import main as _main

    return _main()


def run():
    from .app import run as _run

    return _run()


__all__ = ["main", "run"]
