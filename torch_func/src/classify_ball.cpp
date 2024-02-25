#include <torch/extension.h>

torch::Tensor ClassifyBallCUDA(
    const torch::Tensor &center,
    const torch::Tensor &radius,
    const torch::Tensor &color,
    const float dis_thr,
    const float color_thr);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("classify_ball", &ClassifyBallCUDA, "classify ball with distance and color");
}