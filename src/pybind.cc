#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "operator.h"

PYBIND11_MODULE(TorchEXTLib, m)
{
  m.def("my_add", &AddCUDA, py::arg("a"),
        py::arg("b"), py::arg("c"));
}