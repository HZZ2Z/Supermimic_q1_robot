# Local changes in reward_footflip

Bundled from the MIT-licensed SuperMimic source identified by `shared/supermimic/upstream.json` (upstream commit `118338b19dcd14f768d1e7de5a48d04b78987688`). The upstream hash list records provenance, not the modified working files.

- `robot.py`: sole-point kinematics and an optional next-state kinematics cache.
- `reward_tuning.py`: foot-slip and near-ground downward-velocity auxiliary terms, using the existing reward entrypoint.

The source is vendored so cloning this branch preserves these changes. Upstream licenses and attribution remain in place.

| Modified file | SHA-256 at publication |
|---|---|
| `src/supermimic/robot.py` | `2985fe43f36ca495774c056637591e38a2f10d4eaa245ad448180e4787089cf3` |
| `src/supermimic/reward_tuning.py` | `3be99c51a46a8300b562fc5ef8733d69f32250f4fc2cc8c55a2f82586dc09c3b` |
