# Best Ablations

## Interaction 01

Best run: `12_output_track20_nocontact1000_cdrift10000_smoothness1000_statobj500_intersect0_20_armpose`

Stored at:

`best_outputs/interaction_01/12_output_track20_nocontact1000_cdrift10000_smoothness1000_statobj500_intersect0_20_armpose`

### Selected Hyperparameters

| Setting | Value |
| --- | ---: |
| tracking weight | `20` |
| object CD2D start/end | `0 / 0` |
| object part CD2D start/end | `0 / 0` |
| object smooth translation start/end | `1000 / 1000` |
| object smooth rotation start/end | `1000 / 1000` |
| static object smooth multiplier | `500` |
| human pose start/end | `10 / 10` |
| human pose smooth start/end | `100 / 100` |
| human-object intersection start/end | `0 / 20` |
| no-contact start/end | `1000 / 1000` |
| contact drift start/end | `10000 / 10000` |
| object-object intersection start/end | `0 / 0` |
| arm pose optimization | `on` |
| optimized arm chains | `person_1: right` |
| object scale optimization | `off` |
| final object scales | `iron=1.0`, `ironing_board=1.0` |
| Adam iterations | `10000` |
| Adam learning rate | `0.0001` |
| SDF resolution | `128` |

### Result Snapshot

| Metric | Value |
| --- | ---: |
| best total loss | `0.3254906535` |
| best iteration | `8950` |
| tracking raw | `0.0029997190` |
| no-contact raw | `0.0001978151` |
| contact drift raw | `0.0000011787` |
| human-object intersection raw | `0.0006668018` |
| human max pose delta | `5.0473 deg` |
| human mean pose delta | `2.1738 deg` |

### Notes

This is the current best balanced setting for interaction 01. It keeps hand-object drift very low, keeps object scale fixed, uses only the PAG-selected right arm pose correction, and avoids the instability seen with stronger no-contact weights.
