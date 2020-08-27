"""Tools to load a config with overrides and includes."""
from typing import Any, Dict, Optional, Sequence, Tuple
import logging
import pathlib
import re

import google.protobuf.text_format

import conf.conf_pb2

ProtoMessage = Any

CONF_ROOT = pathlib.Path(conf.conf_pb2.__file__).parent
PROJ_ROOT = pathlib.Path(__file__).parent.parent
EXT = ".prototxt"
INCLUDE_KEY = "I"


def overrides_to_dict(overrides: Sequence[str]) -> Dict[str, str]:
    d = {}
    for override in overrides:
        try:
            name, value = override.split("=", 1)
        except ValueError:
            raise ValueError(f"Bad override: {override}. Expected format: key=value")
        d[name] = value
    return d


def _resolve_include(
    path: str, include_dirs: Sequence[pathlib.Path], mount_point: str
) -> pathlib.Path:
    """Tries to find the config in include_dirs and returns full path.

    path is either a full path or a relative path (relive to one of include_dirs)
    """
    if path.startswith("/"):
        full_path = pathlib.Path(path)
        if not full_path.exists():
            raise ValueError(f"Cannot find include path {path}")
        return full_path
    if path.endswith(EXT):
        path = path[: -len(EXT)]
    possible_includes = []
    mount_point = mount_point.strip(".")
    if mount_point:
        include_dirs = list(include_dirs) + [
            p / mount_point.replace(".", "/") for p in include_dirs
        ]
    for include_path in include_dirs:
        full_path = include_path / (path + EXT)
        if full_path.exists():
            return full_path
        elif full_path.parent.exists():
            possible_includes.extend(
                str(p.resolve())[len(str(include_path.resolve())) : -len(EXT)].lstrip("/")
                for p in full_path.parent.iterdir()
                if str(p).endswith(EXT)
            )

    err_msg = f"Cannot find include {path}"
    if possible_includes:
        err_msg += ". Possible typo, known includes:\n%s" % "\n".join(possible_includes)
    raise ValueError(err_msg)


def _parse_overrides(
    overrides: Sequence[str], include_dirs: Sequence[pathlib.Path]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Separate overrides into include-overrides and scalar overrides."""
    override_dict = overrides_to_dict(overrides)
    include_dict, scalar_dict = {}, {}
    for key, value in override_dict.items():
        if key.startswith(INCLUDE_KEY):
            key = key[1:].lstrip(".")
            _resolve_include(value, include_dirs, key)
            include_dict[key] = value
        else:
            scalar_dict[key] = value
    return include_dict, scalar_dict


def _get_sub_config(cfg: ProtoMessage, mount: str) -> ProtoMessage:
    if not mount:
        return cfg
    subcfg = cfg
    for key in mount.split("."):
        if not hasattr(subcfg, key):
            raise ValueError("Cannot resolve path '%s' in config:\n%s" % (mount, cfg))
        subcfg = getattr(subcfg, key)
    return subcfg


def _apply_scalar_override(cfg: ProtoMessage, mount: str, value: str) -> None:
    assert mount, "Scalaer override with empty key!"
    # We want something like recursive_seattr(cfg, mount, value). But we
    # need to handle the recursive parts and also cast value to correct
    # type.
    mount_parent, key = mount.rsplit(".", 1) if "." in mount else ("", mount)
    subcfg = _get_sub_config(cfg, mount_parent)
    if type(subcfg).__name__ == "ScalarMapContainer":
        # Shortcut for maps.
        subcfg[key] = value
        return
    if type(subcfg).__name__ == "RepeatedScalarContainer":
        # Shortcut for arrays.
        try:
            key = int(key)
        except ValueError:
            raise ValueError(f"Got non-integer key {key} for repeated feild {mount_parent}")
        if key != -1 and not 0 <= key <= len(subcfg):
            raise ValueError(
                f"Cannot acess element {key} in list {mount_parent} that has {len(subcfg)}"
                " elements. Use '-1' to append an element"
            )
        if key == -1 or key == len(subcfg):
            subcfg.append(value)
        else:
            subcfg[key] = value
        return

    if not hasattr(subcfg, key):
        raise ValueError("Cannot resolve path '%s' in config:\n%s" % (mount, cfg))
    attr_type = type(getattr(subcfg, key))
    if type(attr_type).__name__ == "GeneratedProtocolMessageType":
        raise ValueError("Trying to set scalar '%s' for message type '%s'" % (value, mount))
    if attr_type is bool:
        value = value.lower()
        assert value in ("true", "false", "0", "1"), value
        value = True if value in ("true", "1") else False
    elif attr_type is int and not value.isdigit():
        # If enum is redefined we have to search in the parrent object
        # for all enums.
        for maybe_enum_object in type(subcfg).__dict__.values():
            if isinstance(
                maybe_enum_object, google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper
            ):
                if value in maybe_enum_object.keys():
                    value = dict(maybe_enum_object.items())[value]
                    break
    try:
        value = attr_type(value)
    except ValueError:
        raise ValueError(
            "Value for %s should be of type %s. Cannot cast provided value %s to this type"
            % (mount, attr_type, value)
        )
    setattr(subcfg, key, value)


def _parse_text_proto_into(path, msg):
    with path.open() as stream:
        proto_text = stream.read()
    proto_text = re.sub(r"\{\{ *ROOT_DIR *\}\}", str(PROJ_ROOT), proto_text)
    try:
        google.protobuf.text_format.Merge(proto_text, msg)
    except google.protobuf.text_format.ParseError:
        logging.error(
            "Got an exception while parsin proto from %s into type %s. Proto text:\n%s",
            path,
            type(msg),
            proto_text,
        )
        raise


def _apply_include(
    cfg: ProtoMessage, mount: str, include: str, include_dirs: Sequence[pathlib.Path]
) -> None:
    path = _resolve_include(include, include_dirs, mount)
    assert path is not None, include
    subcfg = _get_sub_config(cfg, mount)
    _parse_text_proto_into(path, subcfg)


def load_cfg(
    config_path: pathlib.Path,
    overrides: Sequence[str],
    extra_include_dirs: Sequence[pathlib.Path] = tuple(),
) -> Tuple[str, conf.conf_pb2.MetaCfg]:
    """Loads message of type Cfg from the file and applies overrides and includes.

    Order of config compostion. Later components override earlier components.
        * Default includes from config_path.
        * The content of config in config_path.
        * Includes from overrides.
        * Scalar overrides.

    Returns a pair (task name, task_config) where config is the actual value
    of task oneof.
    """
    root_cfg = conf.conf_pb2.MetaCfg()
    _parse_text_proto_into(config_path, root_cfg)
    task = root_cfg.WhichOneof("task")
    if not task:
        raise ValueError("Bad config - no specific config specified:\n%s" % root_cfg)
    default_includes = root_cfg.includes

    include_dirs = []
    include_dirs.append(config_path.resolve().parent)
    include_dirs.append(CONF_ROOT / "common")
    include_dirs.extend(extra_include_dirs)

    # Resolve one of.
    final_cfg = conf.conf_pb2.MetaCfg()
    task_cfg = getattr(final_cfg, task)

    # First apply default includes.
    for include in default_includes:
        _apply_include(task_cfg, include.mount, include.path, include_dirs)

    # Apply local config.
    task_cfg.MergeFrom(getattr(root_cfg, task))

    include_overides, scalar_overideds = _parse_overrides(overrides, include_dirs)
    logging.debug("Include overrides: %s", include_overides)
    logging.debug("Scalar overrides: %s", scalar_overideds)

    # Apply includes.
    for mount, include in include_overides.items():
        _apply_include(task_cfg, mount, include, include_dirs)

    # Apply scalar overrides.
    for mount, value in scalar_overideds.items():
        _apply_scalar_override(task_cfg, mount, value)
    return task, final_cfg


def save_config(cfg: ProtoMessage, path: pathlib.Path):
    with path.open("w") as stream:
        stream.write(google.protobuf.text_format.MessageToString(cfg))
        stream.write("\n")
