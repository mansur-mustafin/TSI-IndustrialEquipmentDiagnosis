from dataclasses import dataclass


@dataclass
class KGClientConfig:
    ttl_path: str

    @classmethod
    def from_yaml(cls, yaml_path: str, verbose: bool = False) -> "KGClientConfig":
        import yaml

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        if verbose:
            print(f"Loaded KG client config from: {yaml_path}\n")
            print(yaml.dump(raw, sort_keys=False))

        data = raw.get("data", {})

        return cls(**data)
