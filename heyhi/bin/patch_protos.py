"""Patches generated proto messages to add extra fields.

Generated *_pb2.py files look like this:

    from google.protobuf import reflection as _reflection
    ...
    ActionRecallTask = _reflection.GeneratedProtocolMessageType('ActionRecallTask', (_message.Message,), dict(
    DESCRIPTOR = _ACTIONRECALLTASK,
    __module__ = 'conf.conf_pb2'
    # @@protoc_insertion_point(class_scope:fairdiplomacy.ActionRecallTask)
    ))

_reflection.GeneratedProtocolMessageType is esstentially a type() function
that construcs a new class. We replace this function with
magicaly_wrapped_proto_builder(). The new builder alters the dics with new
fields from extra_fields() function.
"""

from google.protobuf import message as _message

import abc
import inspect
import pathlib

TAG = "## PATCHED WITH HEYHI"

CONF_INCLUDE_PREFIX = "conf."


class _FrozenConf:
    heyhi_patched = True

    @classmethod
    @abc.abstractmethod
    def get_proto_class(cls) -> type:
        pass

    def __init__(self, *args, **kwargs):
        assert not args or not kwargs, "Either (msg, fields) or **kwargs"
        if args:
            [fields, msg] = args
            self._fields = fields
        else:
            kwargs = {
                k: (v.to_editable() if isinstance(v, _FrozenConf) else v)
                for k, v in kwargs.items()
            }
            msg = self.get_proto_class()(**kwargs)
            self._fields = msg.to_frozen().__dict__["_fields"]
        assert isinstance(msg, self.get_proto_class()), (type(msg), self.get_proto_class())
        self._msg = msg

    def __repr__(self):
        return self._msg.__repr__()

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                "Bad field '%s'. Known: %s" % (name, ", ".join(sorted(self._fields)))
            )

    def __setattr__(self, name, value):
        if name != "_fields" and name != "_msg":
            raise RuntimeError(f"Attempt to set {name}={value} on read-only config")
        super().__setattr__(name, value)

    def __getstate__(self):
        return self._msg

    def __setstate__(self, msg):
        self._msg = msg
        self._fields = msg.to_frozen().__dict__["_fields"]

    def __str__(self):
        return str(self._msg)

    def to_str_with_defaults(self):
        return self._msg.to_str_with_defaults()

    def WhichOneof(self, name):
        import warnings

        warnings.warn("do not WhichOneof", DeprecationWarning)
        return getattr(self, f"which_{name}")

    def to_dict(self, *, with_defaults=False, with_all=False):
        return self._msg.to_dict(with_defaults=with_defaults, with_all=with_all)

    def is_frozen(self):
        return True

    def to_frozen(self):
        return self

    def to_editable(self):
        copy = type(self._msg)()
        copy.MergeFrom(self._msg)
        return copy


def create_frozen_class(proto_msg_class, name):
    fields = dict(get_proto_class=classmethod(lambda cls: proto_msg_class))
    klass = type(f"Frozen{name}", (_FrozenConf,), fields)
    return klass


def _extra_fields(msg_name, descriptor):
    """Extra methods to add to each proto message."""

    # Imports and typing has to be put here as the whole function will be
    # copied to _pb2.py.
    from typing import List, Dict, Union

    Scalar = Union[None, str, float, int, bool]
    NestedDictList = Union[Dict[str, "NestedDictList"], List["NestedDictList"], Scalar]

    name2field = dict((f.name, f) for f in descriptor.fields)

    # Mapping from a key from oneof's type to oneof name.
    oneof_str_name = {}
    # Mapping from fields within oneof to the name of the oneof.
    field_to_oneof_name = {}
    for oneof in descriptor.oneofs:
        field_to_oneof_name[oneof.name] = oneof.name
        field_to_oneof_name[oneof.name + "_which"] = oneof.name
        oneof_str_name[oneof.name + "_which"] = oneof.name
        for field in oneof.fields:
            field_to_oneof_name[field.name] = oneof.name

    def to_frozen(self):
        set_fields = frozenset(f[0].name for f in self.ListFields())

        # Mapping from fields within oneof to the name of the oneof.
        field_to_oneof_name = {}
        for oneof in descriptor.oneofs:
            for field in oneof.fields:
                field_to_oneof_name[field.name] = oneof.name

        def maybe_to_dict(msg):
            if isinstance(msg, _message.Message):
                return msg.to_frozen()
            return msg

        ret = {}
        for name, field in name2field.items():
            if name in field_to_oneof_name:
                chosen_oneof = self.WhichOneof(field_to_oneof_name[name])
                if chosen_oneof is None:
                    value = None
                elif name != chosen_oneof and name != field_to_oneof_name[name]:
                    value = None
                else:
                    value = getattr(self, chosen_oneof)
            else:
                value = getattr(self, name)
            if isinstance(value, _message.Message):
                value = maybe_to_dict(value)
            elif field.label == field.LABEL_REPEATED:
                if type(value).__name__.split(".")[-1] == "ScalarMapContainer":
                    value = {x: maybe_to_dict(value[x]) for x in value}
                else:
                    value = tuple(maybe_to_dict(x) for x in value)
            else:
                # A scalar.
                assert (
                    isinstance(value, (float, str, int, bool)) or value is None
                ), f"Excepted a value for {name} to be a scalar. Got {value}"
                if field.enum_type is not None:
                    value = field.enum_type.values_by_number[value].name
                if name not in set_fields and not field.has_default_value:
                    value = False if field.type == field.TYPE_BOOL else None
            ret[name] = value
        for oneof in descriptor.oneofs:
            chosen = self.WhichOneof(oneof.name)
            ret["which_" + oneof.name] = chosen
            ret[oneof.name] = ret[chosen] if chosen is not None else None
        return FROZEN_SYM_BD[descriptor.full_name](ret, self)

    def to_str_with_defaults(self):
        msg = type(self)(**self.to_dict(with_defaults=True))
        return str(msg)

    def to_dict(self, *, with_defaults=False, with_all=False) -> Dict[str, NestedDictList]:
        """Returns the configs as a nested dict.

        By default will emulate protobuf's default, i.e., return only fields
        that are set explicitly and remove empy messages and dict.

        If with_defaults is set, then will add to the dict all fields that
        have explicitly set defauits.

        If with_all is set, then with_defaults is activated and additionally
        will output every single field with either the value or None. All
        empty messages/lists will be shown.

        For oneofs we only return the chosen field of any. If not fields
        chosen, we return nothing.

        See ToDict tests in `test_conf.py` for more examples.
        """
        ret = {}
        set_fields = frozenset(f[0].name for f in self.ListFields())
        if with_all:
            with_defaults = True

        def maybe_to_dict(msg):
            if isinstance(msg, _message.Message):
                return msg.to_dict(with_defaults=with_defaults, with_all=with_all)
            return msg

        for name, field in name2field.items():
            if name in field_to_oneof_name:
                # For oneof fields we either print nothing, of the field that is set.
                chosen_oneof = self.WhichOneof(field_to_oneof_name[name])
                if name != chosen_oneof:
                    continue
            value = _message.Message.__getattribute__(self, name)
            if isinstance(value, _message.Message):
                value = maybe_to_dict(value)
                if not with_all and not value and name not in field_to_oneof_name:
                    continue
            elif field.label == field.LABEL_REPEATED:
                if type(value).__name__.split(".")[-1] == "ScalarMapContainer":
                    value = {x: maybe_to_dict(value[x]) for x in value}
                else:
                    value = [maybe_to_dict(x) for x in value]
                if not with_all and not value:
                    continue
            else:
                # A scalar.
                assert (
                    isinstance(value, (float, str, int, bool)) or value is None
                ), f"Excepted a value for {name} to be a scalar. Got {value}"
                if field.enum_type is not None:
                    value = field.enum_type.values_by_number[value].name
                if name not in set_fields and not with_all:
                    if not field.has_default_value or not with_defaults:
                        continue
                if name not in set_fields and not field.has_default_value:
                    value = False if field.type == field.TYPE_BOOL else None
            ret[name] = value
        return ret

    def is_frozen(self):
        return False

    def to_editable(self):
        return self

    funcs = {
        "to_dict": to_dict,
        "heyhi_patched": True,
        "to_str_with_defaults": to_str_with_defaults,
        "to_frozen": to_frozen,
        "to_editable": to_editable,
        "is_frozen": is_frozen,
    }
    return funcs


def _magicaly_wrapped_proto_builder(builder):
    if builder.__name__ == "new_builder":
        # No double patching.
        return builder

    def new_builder(name, bases, fields):
        return builder(name, bases, dict(fields, **_extra_fields(name, fields["DESCRIPTOR"])))

    return new_builder


def patch_pb2(pb2_path):
    with open(pb2_path) as stream:
        content = stream.read()
    if TAG in content:
        return

    old_full_classes = []
    is_msg_descriptor = False
    for line in content.split("\n"):
        if " = _descriptor." in line:
            is_msg_descriptor = line.strip().endswith("_descriptor.Descriptor(")
        if line.startswith("  full_name='") and is_msg_descriptor:
            old_full_classes.append(line.split("=")[1].strip().strip(",").strip("'"))

    # Creating patched version of generated proto file. Will be saved in original files.
    lines = []

    # Patch includes to use "internal_*" files.
    for line in content.split("\n"):
        lines.append(line)
        if "@@protoc_insertion_point(imports)" in line:
            lines.append(f"{TAG} START")
            lines.append("import abc")

            lines.append(inspect.getsource(_extra_fields))
            lines.append(inspect.getsource(_magicaly_wrapped_proto_builder))
            lines.append("from google.protobuf import reflection as _reflection")
            lines.append(
                "_reflection.GeneratedProtocolMessageType = _magicaly_wrapped_proto_builder("
                "_reflection.GeneratedProtocolMessageType)"
            )
            lines.append(f"{TAG} end\n")

    def remove_package_name(full_object_path):
        return klass.split(".", 1)[1]  # Stripping "fairdiplomacy."

    lines.append("if 'FROZEN_SYM_BD' not in globals():")
    lines.append("   globals()['FROZEN_SYM_BD'] = {}")
    lines.append(inspect.getsource(_FrozenConf))
    lines.append(inspect.getsource(create_frozen_class))
    for klass in reversed(old_full_classes):
        short_klass = remove_package_name(klass)
        lines.append(
            f"FROZEN_SYM_BD['{klass}'] = create_frozen_class(_sym_db.GetSymbol('{klass}'), '{short_klass}')"
        )
        lines.append(f"Frozen{short_klass} = FROZEN_SYM_BD['{klass}']")

    with open(pb2_path, "w") as stream:
        stream.write("\n".join(lines))

    # Creating a user facing set of classes.
    lines = []
    lines.append(f"{TAG} START")
    lines.append(f"from {CONF_INCLUDE_PREFIX}%s import *" % pb2_path.name.rsplit(".", 1)[0])
    lines.append("# Create new classes and flat names for them.")

    for klass in reversed(old_full_classes):
        short_klass = remove_package_name(klass)
        if len(short_klass.split(".")) == 1:
            lines.append(f"Proto_{short_klass} = {short_klass}")
        lines.append(f"{short_klass} = FROZEN_SYM_BD['{klass}']")
    lines.append(f"{TAG} end\n")
    with open(str(pb2_path).replace("_pb2", "_cfgs"), "w") as stream:
        stream.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("conf_pb2_paths", nargs="+", type=pathlib.Path)
    for pb2_path in parser.parse_args().conf_pb2_paths:
        if pb2_path.name.startswith("internal_"):
            print("Skipping", pb2_path)
        else:
            print("Patching", str(pb2_path))
            patch_pb2(pb2_path)
