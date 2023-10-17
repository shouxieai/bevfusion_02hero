"""
第一步，写setup.py
编写一个使用setuptools来编译我们的C++代码的setup.py脚本来构建我们的C++扩展

"""
from setuptools import setup, Extension
from torch.utils import cpp_extension 

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

# CppExtension是围绕setuptools.Extension的一个便捷包装器，它传递正确的包含路径并将扩展的语言设置为C++
# Extension(
#    name='lltm_cpp',
#    sources=['lltm.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++')