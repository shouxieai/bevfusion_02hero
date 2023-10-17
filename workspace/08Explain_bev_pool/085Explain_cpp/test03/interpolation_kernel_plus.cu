#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel( //kernel的返回值一直是void
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,//输入不会变，加const
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp //内存逐渐变化，不用const
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x; //理解为block的索引*blockDim的维度 + thread线程索引
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    //接下来是边界判断，两种写法

    // if (n < feats.size(0) && f < feats.size(2)){ //第二种写法
    //     //内部写你的代码
    // }

    if(n >= feats.size(0) || f >= feats.size(2)) return ; //第一种写法

    const scalar_t u = (points[n][0]+1)/2; //第n个点到x轴的距离, x取值均在-1至1之间。所加1除2是对坐标值归一化。
    const scalar_t v = (points[n][1]+1)/2; //第n个点到y轴的距离
    const scalar_t w = (points[n][2]+1)/2; //第n个点到z轴的距离

    //获得权重。
    /*
    每个边长度为1
          f1。----------。f2
            |    | v    |
            |----。     |
            |  u        |
          f3。----------。f4
    
    f = u*v*f4 + (1 - u)*v*f3 +u*(1-v)*f2+(1-u)(1-v)*f1
    下方代码无非是推广到3维

    底下一圈4个点，乘w   上面一圈4点 乘
    
    */
    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c; // a+b+c=1-vw  这里就是省了个乘法。实际还是v*w
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + 
                                    b * feats[n][1][f] +
                                    c * feats[n][2][f] +
                                    d * feats[n][3][f]) +
                                u * (a * feats[n][4][f] +
                                     b * feats[n][5][f] +
                                     c * feats[n][6][f] +
                                     d * feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options()); //如果at::TensorOptions  feat_interp与feats的参数一样，就用feats.options()
    // torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device)); //如果at::TensorOptions，不一致。

    /*
    2. 构思算法
    思考哪些维度是可以平行运算的
    N，F 这两个维度都是可以平行运算的。
    */
    //const dim3 threads(16, 16); 
   //一般定义为256(根据硬件来)。即threads中所有维度相乘，不大于256
   //这里作者认为N, F两个维度，不会一个特别多，一个特别少。所以都分配成16
   //16*16 = 256(需多尝试,16 16  如果NF不均衡 8 32  256也可以尝试512)

    //const int threds = 256;
    const dim3 threads(16, 16);

    /*
    3. 计算gridDim
    可复制，较固定
    */
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y); //整数类型相除，仍然是整数
    
    /*
    4.AT_DISPATCH_FLOATING_TYPES首先告诉内部的数据进行的是浮点数运算
        包含float32   float64 
    4.1 如果想用半精度计算，使用AT_DISPATCH_FLOATING_TYPES_HALF(除了float32 float64，还支持float16)
            - AT_DISPATCH_ALL_TYPES 支持所有类型
    4.2 整数运算T_DISPATCH_INTEGRAL_TYPES

    The purpose of AT_DISPATCH_FLOATING_TYPES is to take care of this dispatch for us. 
    It takes a type (gates.type() in our case), a name (for error messages) and a lambda function. 
    Inside this lambda function, the type alias scalar_t is available and is defined as the type that the tensor 
    actually is at runtime in that context. As such, if we have a template function (which our CUDA kernel will be), 
    we can instantiate it with this scalar_t alias, and the correct function will be called. In this case, 
    we also want to retrieve the data pointers of the tensors as pointers of that scalar_t type. 
    If you wanted to dispatch over all types and not just floating point types (Float and Double), you can use AT_DISPATCH_ALL_TYPES.
    AT_DISPATCH_FLOATING_TYPES 的目的是为我们处理这种分发。它接受一个类型（在我们的情况下是 gates.type()）、
    一个名称（用于错误消息）和一个 lambda 函数。在这个 lambda 函数内部，类型别名 scalar_t 是可用的，
    并且在该上下文中被定义为张量在运行时实际上是什么类型。因此，如果我们有一个模板函数（我们的 CUDA 内核将是一个模板函数），
    我们可以用这个 scalar_t 别名实例化它，将会调用正确的函数。在这种情况下，我们还想以 scalar_t 类型的指针形式获取张量的数据指针。
    如果你想对所有类型进行分发，而不仅仅是浮点类型（Float 和 Double），你可以使用 AT_DISPATCH_ALL_TYPES。

    scalar_t 是可用的，并且在该上下文中被定义为张量在运行时实际上是什么类型。
    */
    //第一个参数，tensor的类型。 如果feats.type()类型是float，那么核函数的返回值就是float
    //第二个参数，制定一个函数名称。这里名称选择与cuda函数名称一致
    //第三个参数，试一试lambda表达式，
        //内部是一个核函数
            //核函数的入参，是3个
    //scalar_t 代表第一个参数的类型，这里是feats.type()。


    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            //第一个参数是类型，scalar_t表示feats的类型暂时不确定是float32还是64，根据实际情况来定。有一定灵活性
                //当然也可以写死，比如写float，但是如果以后进来的是float64，就会报错。
            //第二个参数是该参数(feats)的维度 3  表示feats是3维的tensor
            //第三个torch::RestrictPtrTraits，表示必须使用 __restrict__ 关键字
                //__restrict__ 告诉编译器feats与其他参数不会有交集，即它们不会指向相同的内存区域
                    //这允许编译器生成更优化的代码，因为它可以确保在函数内部对 a 或 b 的任何更改都不会影响到另一个指针
            //第四个参数：size_t  一般会根据scalar_t类型变化而变化(目前理解，是类似数据类型的size，不准确)。  一般不用改这个参数。
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
        /*
        packed_accessor是CUDA版本的访问器， CPU版本是accessor，下方是使用方法
            torch::Tensor foo = torch::rand({12, 12});
            auto foo_a = foo.accessor<float,2>();       2表示foo这个tensor有几维
            assert foo is 2-dimensional and holds floats.
        访问器有点：访问器对象具有相对高级的接口，具有 .size() 和 .stride() 方法以及多维索引。高效地访问CPU张量上的数据
        Packed访问器将尺寸和步长数据复制到其结构内，而不是指向它。这使我们能够将其传递给CUDA内核函数，并在其中使用其接口。
        */

    }));

    //如果核函数的参数是固定的，代码可以简化很多。scalar_t，size_t可以不要
        //scalar_t 变类型，我们确定了类型。scalar_t就换成了float
        //size_t也是因为类型确定了，就自动计算出来了，不用给了。
        //缺点：类型固定了。不灵活，输入float64都可能会报错

    //     AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", ([&] {
    //     lltm_cuda_forward_kernel<<<blocks, threads>>>(
    //         feats.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
    //         points.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
    //         feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits>()，
    //          注意：如果此时有一个不是tensor的类型的入参。直接写就行，不用packed_accessor
    //          int a   <---
    //         );

    // }));
    
    return feat_interp; //5.最后return结果
    /*
    因为torch::Tensor trilinear_fw_cu  的返回值定死了是torch::Tensor，所以只能回传一个tensor

    如果返回值类型是 std::vector<torch::Tensor> 就可以  return {feat_interp , feat_inter2, ...}回传值
    记得声明处也要修改。
    
    */
}