# MLP data scales

w0 shape: (`data_size, perceptron_size`)

b0 shape: (`perceptron_size,`)

w1 shape: (`perceptron_size, num_classes`)

b1 shape: (`num_classes,`)

Here we set `data_size=2` and `num_classes=2`.

| input size | perceptron size | inter size(/MB) | inter percentage(%) |
| :-----: | :-----: | :-----: | :-----: |
| 400 | 3 | 0.0366898 | 82.6644 |
| `400*8` | 3 | 0.293037 | 82.7468 |
| `400*8` | `3*8` | 1.31883 | 95.5445 |
| `400*8` | `3*8*8` | 9.52516 | 99.3253 |
| `400*8` | `3*8*8*8` | 75.1758 | 99.88 |
| `400*8` | `3*8*8*8*8` | 600.381 | 99.9508 |
| `400*8*2` | `3*8*8*8*8` | 1200.53 | 99.9703 |
| `400*8*4` | `3*8*8*8*8` | 2400.82 | 99.9801 |
| `400*8*8` | `3*8*8*8*8` | 4801.41 | 99.9849 |
