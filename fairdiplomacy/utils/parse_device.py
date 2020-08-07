def parse_device(device):
    try:
        if device.startswith("cuda") or device.startswith("cpu"):
            return device
    except AttributeError:
        pass

    return f"cuda:{device}"
