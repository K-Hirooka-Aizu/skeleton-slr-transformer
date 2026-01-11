# Training, Hyperparameter Optimizing and Evaluation.

## Training the Model with default hyperparameter setting.
```bash
python ./scripts/train.py \
        data=wlasl100 \ # (wlasl300, wlasl1000, wlasl2000, jsl0, ksl0)
        model=transformer \ # (prenormtransformer, stgcn, ctrgcn)
        epochs=1500
```
### Hyperparameter
please check the [default.yaml](./conf/default.yaml)
