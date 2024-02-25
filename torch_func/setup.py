from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="ClassifyBall",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="classify_ball",
            sources=["src/classify_ball_cuda.cu", "src/classify_ball.cpp"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
