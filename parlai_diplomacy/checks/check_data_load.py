"""
Check that all of the data loads without crashing
"""
import parlai.utils.logging as logging
import parlai_diplomacy.utils.loading as load
import contextlib
import io


load.register_all_agents()
load.register_all_tasks()


STREAM_TEACHER_LST = [
    "state_order_chunk",
    "order_history_order_chunk",
    "speaker_token_order_chunk",
    "dummy_token_order_chunk",
    "state_message_order_chunk",
    "message_state_order_chunk",
    "order_history_message_order_chunk",
    "message_order_history_order_chunk",
    "message_order_chunk",
    "dialogue_chunk",
    "message_history_state_dialogue_chunk",
    "state_message_history_dialogue_chunk",
    "message_history_order_history_dialogue_chunk",
    "message_history_order_history_state_dialogue_chunk",
]


@contextlib.contextmanager
def capture_output():
    """
    Suppress all logging output into a single buffer.
    Use as a context manager.
    >>> with capture_output() as output:
    ...     print('hello')
    >>> output.getvalue()
    'hello'
    """
    sio = io.StringIO()
    with contextlib.redirect_stdout(sio), contextlib.redirect_stderr(sio):
        yield sio


def display_data(opt):
    """
    Run through a display data run.
    :return: stdout_train
    """
    import parlai.scripts.display_data as dd

    with capture_output() as _:
        parser = dd.setup_args()
        parser.set_params(**opt)
        popt = parser.parse_args([])

    with capture_output() as train_output:
        popt["datatype"] = "train:stream"
        dd.display_data(popt)

    return train_output.getvalue()


def test_load():
    """
    Does display_data reach the end of the loop?
    """
    output = {}
    for teacher in STREAM_TEACHER_LST:
        logging.warn(f"Testing load for teacher: {teacher}")
        opt = {
            "task": teacher,
            "num_examples": 1,
            "display_verbose": True,
        }
        # TODO: assert something about the output?
        str_output = display_data(opt)
        logging.warn(str_output)
        output[teacher] = str_output

    logging.success(
        "\n\nFinished running through all streaming teachers, writing example output to file."
    )
    stdout = ""
    for teach, out in output.items():
        stdout += f"{teach}\n{out}\n{'=' * 50}\n\n"

    fle = "/tmp/diplomacy_test_data_logs.txt"
    with open(fle, "w") as f:
        f.write(stdout)

    logging.warn(f"Output written to file: {fle}")


if __name__ == "__main__":
    test_load()
