# exp333_task251_mask_inventory

## Hypothesis
Build a diagnostic inventory for task251 closed zero-component mask and color1 insertion evidence.

## Result
- data available: `False`
- historical rule: `Fill every 0-component that does not touch the grid border and whose 4-neighbor boundary colors are exactly {2}; write color 1.`
- full pass evidence: `266_pass_0_fail`
- inventory: `inventory.csv`

## Decision
No submit. This diagnostic confirms task251 should continue through mask + color1 lowering, not pure GridSample.

## Next
Run a rule reconfirmation and mask cost probe after the NeuroGolf raw task data is present.

