# Runtime thresholds (production)

The runtime loads per-head classification thresholds from a JSON file if present. This allows you to deploy the evaluation-time thresholds without changing code or rebuilding the add-on.

## Search order

1. `THRESHOLDS_JSON` environment variable (absolute path)
2. `/data/thresholds.json` (Home Assistant add-on data share)
3. `/config/ellie_thresholds.json` (Home Assistant config share)

The first existing file wins.

## JSON schema

```json
{
  "poop_threshold": 0.80,
  "copro_threshold": 0.17,
  "frames": 24,
  "size": 224,
  "timestamp": 1730996205
}
```

Only `poop_threshold` and `copro_threshold` are used at runtime. Other keys are informational.

## How to export the file

After running evaluation:

```bash
python -m ml.eval.eval_val --outdir datasets/ellie_multihead_vX
# writes datasets/ellie_multihead_vX/thresholds.json
```

Copy to the add-on data share (persistent across updates):

```bash
cp datasets/ellie_multihead_vX/thresholds.json /data/thresholds.json
```

Or place under the HA config share:

```bash
cp datasets/ellie_multihead_vX/thresholds.json /config/ellie_thresholds.json
```

Optionally, set an explicit path via env var in your supervisor options:

```json
{
  "environment": {
    "THRESHOLDS_JSON": "/data/models/my_thresholds.json"
  }
}
```

## Code reference

- Loader: `watcher/settings.py::_maybe_apply_thresholds_from_file()`
- Usage: thresholds override `poop_thresh` and `copro_thresh` in `Settings`


