#!/bin/bash
set -e
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
#./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
#pip uninstall tensorflow
#pip install /tmp/tensorflow_pkg/tensorflow-2.1.3-cp35-cp35m-linux_x86_64.whl
