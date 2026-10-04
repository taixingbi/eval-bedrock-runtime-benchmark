"""Generate Experiment B after all three Experiment A baselines are confirmed."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bedrock_benchmark.context_history import prepare
from bedrock_benchmark.models import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--template', type=Path, default=Path('experiments/diagnostic-context-history.yaml'))
    parser.add_argument('--profiles', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        model = load_models(names=[args.model])[0]
        configs = prepare(args.template, args.profiles, model)
        paths = [args.output_dir / (cfg['name'] + '.yaml') for cfg in configs]
        if any(p.exists() for p in paths):
            raise ValueError('Output files already exist; choose a new directory to preserve baseline provenance')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, cfg in zip(paths, configs):
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f'{path}: {cfg["sweep"]["values"]} RPS')
    except (ValueError, KeyError, IndexError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
