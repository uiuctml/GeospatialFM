import argparse

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Geospatial training", add_help=add_help)
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument(
    #     "--no-resume",
    #     action="store_true",
    #     help="Whether to not attempt to resume from the checkpoint directory. ",
    # )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--exp_name", default=None, type=str, help="Experiment name")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument("--device", nargs='+', default=[0], type=int, help="GPU device to use")
    # parser.add_argument("--learning_rate", type=float, default=None, help="Override the Learning Rate from config (for sweep)")

    return parser