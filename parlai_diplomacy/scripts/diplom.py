#!/usr/bin/env python3

import parlai_diplomacy.utils.loading as load
from parlai.core.script import superscript_main


def main():
    load.register_all_agents()
    load.register_all_tasks()
    superscript_main()


if __name__ == "__main__":
    main()
