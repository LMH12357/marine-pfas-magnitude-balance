# Version 2.0.0

The analysis now consumes the corrected compilation and source reconstruction
through explicit inputs. Environmental extraction and workbook updates share
field/provenance definitions and preserve source-reported measurements.

Coastal routing is shared across preparation, validation and source scenarios.
The wet graph excludes land-interior-crossing edges and preserves all source
entries. Reviewed receiving-water anchors are numerical attachment points, not
replacement measurements.

Model validation, source scenarios and evidence calculations were regenerated
from the corrected inputs. The model-panel dictionary and data provenance are
versioned. Existing seeds and model-selection rules are retained.

Additional regional diagnostics use dense paths, an alternative composition
model, training-group deletion and paired SHAP interaction decomposition. These
have separate provenance and preserve the production model.

The preceding fixed numerical products are removed from the current branch to
prevent accidental reuse with new code. Their complete history remains at
commit `6685c27e34602e4c207d1316d645cbc24c204499`. Corrected derived data are
prepared separately from this code release.
