# Training, Hyperparameter Optimizing and Evaluation.

## Training the Model with default hyperparameter setting.
Run the commands from the repository root.

```bash
python scripts/train.py data=wlasl100 model=postnorm_transformer epochs=1500
```

- `data`: `wlasl100`, `wlasl300`, `wlasl1000`, `wlasl2000`, `jsl0`, `ksl0`
- `model`: `postnorm_transformer` (SSTAN), `prenorm_transformer`, `stgcn`, `ctrgcn`

### Hyperparameter
please check the [default.yaml](./conf/default.yaml)
