import glob
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
# 第一步修改，CppExtension替换成了CUDAExtension

# ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR, "include")]
# sources = glob.glob("test02/*.cpp") + glob.glob("test02/*.cu")# ['test02/interpolation.cpp', 'test02/interpolation_kernel.cu']

# =================test02--test05使用这个=======================================
# setup(
#     name='setup_name', 
#     version="1.0",
#     author="121",
#     author_email="111.qq.com",
#     description="cppcuda test",
#     long_description="cppcuda test",
#     ext_modules=[
#         CUDAExtension(  # 第一步
#             name='ext_modules_name_a', 
#             sources=[       # 第二步，sources=sources。sources部分，使用glob包，筛选cu\cpp文件。手写较麻烦。
#                 "test02/interpolation.cpp",
#                 "test02/interpolation_kernel.cu"
#             ],
#             # include_dirs=include_dirs,  # 第三步，如果cuda函数的声明没有写在cpp里，而是单独有.h的头文件。可以加这个
#             extra_compile_args={  # 第四步，编译的额外的参数，可加可不加。 O2 优化用
#                 'cxx':['-O2'],
#                 'nvcc':['-O2']}
#             ),
#         CUDAExtension(  # 第一步
#             name='ext_modules_name',  
#             sources=[       # 第二步，sources=sources。sources部分，使用glob包，筛选cu\cpp文件。手写较麻烦。
#                 "test03/interpolation_plus.cpp",
#                 "test03/interpolation_kernel_plus.cu"
#             ],
#             # include_dirs=include_dirs,  # 第三步，如果cuda函数的声明没有卸载cpp里，而是单独有.h的头文件。可以加这个
#             extra_compile_args={  # 第四步，编译的额外的参数，可加可不加。 O2 优化用
#                 'cxx':['-O2'],
#                 'nvcc':['-O2']}
#             ),
#         CUDAExtension(  
#             name='ext_modules_name2',  
#             sources=[  
#                 "test04/interpolation_plus2.cpp",
#                 "test04/interpolation_kernel_plus2.cu"
#             ],
#             # include_dirs=include_dirs,  
#             extra_compile_args={  
#                 'cxx':['-O2'],
#                 'nvcc':['-O2']}
#             ),
#         ],
#     cmdclass={
#         'build_ext': BuildExtension} 
# )


# ==============test01请打开下方注释下方是纯cpp的=======================
# from setuptools import setup
# from torch.utils import cpp_extension 
# import os

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))# 向前文件的父目录
# # include_dirs = [os.path.join(ROOT_DIR, "include")]

# setup(
#     name='setup_name', # package的名称。就是python中import中使用的名称
#     version="1.0",
#     author="121",
#     author_email="111.qq.com",
#     description="cppcuda test",
#     long_description="cppcuda test",
#     ext_modules=[
#         cpp_extension.CppExtension(
#             name='ext_modules_name',  # 
#             sources=['test01/src/test085.cpp'],# 如果有多个，使用逗号隔开
#             extra_compile_args={
#                 'cxx':['-O2'],
#                 'nvcc':['-O2']}
#             )
#         ],
#     cmdclass={
#         'build_ext': cpp_extension.BuildExtension} 
# )


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



import torch
import os
def make_cuda_ext( # 包装了一个 扩展Extension的方法。有gpu则用CUDAExtension，没有则CppExtension
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = [] # 初始化宏定义和额外的编译参数
    extra_compile_args = {"cxx": [] + extra_args} # 编译选项(涉及c++编译的知识)

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1": # 如果环境变量中有FORCE_CUDA，即使没有gpu也会尝试进行cuda扩展
        define_macros += [("WITH_CUDA", None)] # 添加一个宏，在voxelization.h中有使用
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [ #定义nvcc编译器的参数。
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        sources += sources_cuda # 添加CUDA源文件
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name), # 会发现名字是module.name
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    
setup(
        name="085Explain_cpp",
        packages=find_packages(),
        include_package_data=True,
        package_data={"085Explain_cpp": ["*/*.so"]},
        classifiers=[ # 原数据 https://packaging.python.org/en/latest/tutorials/packaging-projects/
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",
        ext_modules=[
            make_cuda_ext(
                name="bbb",
                module="test01",
                sources=[
                    "src/test085.cpp"
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )