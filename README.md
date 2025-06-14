# RWKV Inverse Folding

This project is focused on inverse folding using the RWKV model. It includes various scripts and configurations for training and inference.

## Project Structure

- `baseline_rnainverse.py`: Baseline script for RNA inverse folding.
- `train.py`: Script for training the model.
- `rna_train.json` and `rna_test.json`: JSON files for training and testing data.
- `requirements.txt`: Python dependencies.
- `rna_generate.py`: Script for RNA sequence generation.
- `inference.py`: Script for running inference.
- `inference.sh`: Shell script for inference execution.
- `demo-training-run.sh`: Shell script for running a demo training session.
- `demo-training-prepare.sh`: Shell script for preparing demo training.
- `dataset_generatr.py`: Script for generating datasets.
- `src/`: Source code directory containing model, trainer, and utility scripts.
  - `utils.py`: Utility functions.
  - `trainer.py`: Training logic.
  - `binidx.py`: Binary indexing utilities.
  - `dataset.py`: Dataset handling.
  - `model.py`: Model architecture.
- `cuda/`: Directory containing CUDA source files.
  - `wkv7_cuda.cu`, `wkv7_op.cpp`: CUDA and C++ files for WKV version 7.
  - `wkv6state_op.cpp`, `wkv6state_cuda.cu`: CUDA and C++ files for WKV version 6 state.
  - `wkv6_op.cpp`, `wkv6_cuda.cu`: CUDA and C++ files for WKV version 6.
  - `wkv5_cuda.cu`, `wkv5_op.cpp`: CUDA and C++ files for WKV version 5.
- `tokenizer/`: Directory containing tokenizer scripts and vocabulary.
  - `rwkv_vocab_v20230424.txt`: Vocabulary file.
  - `rwkv_tokenizer.py`: Tokenizer implementation.

## Getting Started

1. **Install Dependencies**: Use the `requirements.txt` to install necessary Python packages.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**: Execute the `train.py` script to start training.
   ```bash
   bash demo-training-prepare.sh
   bash demo-training-run.sh
   ```

3. **Inference**: Use the `inference.py` script to perform inference.
   ```bash
   bash inference.sh
   ```

## Acknowledgement

This project is based on or inspired by the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository by BlinkDL.  
We thank the authors for open-sourcing their work.

RWKV is a novel RNN with transformer-level performance, and the original implementation can be found at:
https://github.com/BlinkDL/RWKV-LM

## License

This project is licensed under the MIT License. See the LICENSE file for more details. 
