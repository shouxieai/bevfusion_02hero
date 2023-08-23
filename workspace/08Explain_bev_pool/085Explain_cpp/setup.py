from setuptools import setup
from torch.utils import cpp_extension 

setup(
    name='my_package_name', # package的名称。就是python中import中使用的名称
    version="1.0",
    author="yxy",
    author_email="111.qq.com",
    description="cppcuda test",
    long_description="cppcuda test",
    ext_modules=[
        cpp_extension.CppExtension(
            'my_package_name',  # 
            ['test01/test085.cpp']) # 如果有多个，使用逗号隔开
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
#     author="yxy",
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

