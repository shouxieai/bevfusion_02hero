import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
# 第一步修改，CppExtension替换成了CUDAExtension

# ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR, "include")]
# sources = glob.glob("test02/*.cpp") + glob.glob("test02/*.cu")# ['test02/interpolation.cpp', 'test02/interpolation_kernel.cu']

# =================test02--test05使用这个=======================================
setup(
    name='setup_name', 
    version="1.0",
    author="121",
    author_email="111.qq.com",
    description="cppcuda test",
    long_description="cppcuda test",
    ext_modules=[
        CUDAExtension(  # 第一步
            name='ext_modules_name_a', 
            sources=[       # 第二步，sources=sources。sources部分，使用glob包，筛选cu\cpp文件。手写较麻烦。
                "test02/interpolation.cpp",
                "test02/interpolation_kernel.cu"
            ],
            # include_dirs=include_dirs,  # 第三步，如果cuda函数的声明没有写在cpp里，而是单独有.h的头文件。可以加这个
            extra_compile_args={  # 第四步，编译的额外的参数，可加可不加。 O2 优化用
                'cxx':['-O2'],
                'nvcc':['-O2']}
            ),
        CUDAExtension(  # 第一步
            name='ext_modules_name',  
            sources=[       # 第二步，sources=sources。sources部分，使用glob包，筛选cu\cpp文件。手写较麻烦。
                "test03/interpolation_plus.cpp",
                "test03/interpolation_kernel_plus.cu"
            ],
            # include_dirs=include_dirs,  # 第三步，如果cuda函数的声明没有卸载cpp里，而是单独有.h的头文件。可以加这个
            extra_compile_args={  # 第四步，编译的额外的参数，可加可不加。 O2 优化用
                'cxx':['-O2'],
                'nvcc':['-O2']}
            ),
        CUDAExtension(  
            name='ext_modules_name2',  
            sources=[  
                "test04/interpolation_plus2.cpp",
                "test04/interpolation_kernel_plus2.cu"
            ],
            # include_dirs=include_dirs,  
            extra_compile_args={  
                'cxx':['-O2'],
                'nvcc':['-O2']}
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension} 
)


# ==============test01请打开下方注释下方是纯cpp的=======================
from setuptools import setup
from torch.utils import cpp_extension 
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))# 向前文件的父目录
# include_dirs = [os.path.join(ROOT_DIR, "include")]

setup(
    name='setup_name', # package的名称。就是python中import中使用的名称
    version="1.0",
    author="121",
    author_email="111.qq.com",
    description="cppcuda test",
    long_description="cppcuda test",
    ext_modules=[
        cpp_extension.CppExtension(
            name='ext_modules_name',  # 
            sources=['test01/test085.cpp'],# 如果有多个，使用逗号隔开
            extra_compile_args={
                'cxx':['-O2'],
                'nvcc':['-O2']}
            )
        ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension} 
)


#=========下方是导包方式不同，实际是一样的。

# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, BuildExtensions

# setup(
#     name='lltm_cpp',
#     version="1.0",
#     author="121",
#     author_email="111.qq.com",
#     description="cppcuda test",
#     long_description="cppcuda test",
#     ext_modules=[
#         CppExtension(
#             'test085', 
#             ['test085.cpp'])
#         ],
#     cmdclass={
#         'build_ext': BuildExtensions}
# )

