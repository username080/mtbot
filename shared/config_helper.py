def fetch(config_dict, key):
    """
    Fetches a key from the configuration dictionary.
    Raises an explicit placeholder error and kills the program if missing.
    """
    if key not in config_dict:
        raise ValueError(f"PLACEHOLDER ERROR: Missing configuration key '{key}'")
    return config_dict[key]
