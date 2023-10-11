ir_version: 7
opset_import {
  version: 11
}
producer_name: "pytorch"
producer_version: "1.10"
graph {
  node {
    input: "image" //输入的节点的名字
    input: "conv.weight"
    input: "conv.bias"
    output: "3" //3 仅仅是个名字！！，中间过程中node的输入输出一般仅用数字表示。下方的node节点的output名字不是数字，而是"output"
    name: "Conv_0"
    op_type: "Conv"  //主要看这里，是算子
    //当op_type是Conv时，一般有下面的attribute
    attribute {
      name: "dilations"
      type: INTS
      ints: 1
      ints: 1
    }
    attribute {
      name: "group"
      type: INT
      i: 1
    }
    attribute {
      name: "kernel_shape"
      type: INTS
      ints: 3
      ints: 3
    }
    attribute {
      name: "pads"
      type: INTS
      ints: 1
      ints: 1
      ints: 1
      ints: 1
    }
    attribute {
      name: "strides"
      type: INTS
      ints: 1
      ints: 1
    }
    doc_string: "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py(442): _conv_forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py(446): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1090): _slow_forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/tmp/ipykernel_1106822/2355770593.py(17): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1090): _slow_forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(118): wrapper\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(127): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(1166): _get_trace_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(388): _trace_and_get_graph_from_model\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(437): _create_jit_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(493): _model_to_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(724): _export\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(107): export\n/usr/local/lib/python3.8/dist-packages/torch/onnx/__init__.py(316): export\n/tmp/ipykernel_1106822/2355770593.py(27): <module>\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3508): run_code\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3448): run_ast_nodes\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3269): run_cell_async\n/usr/local/lib/python3.8/dist-packages/IPython/core/async_helpers.py(129): _pseudo_sync_runner\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3064): _run_cell\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3009): run_cell\n/usr/local/lib/python3.8/dist-packages/ipykernel/zmqshell.py(546): run_cell\n/usr/local/lib/python3.8/dist-packages/ipykernel/ipkernel.py(422): do_execute\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(740): execute_request\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(412): dispatch_shell\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(505): process_one\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(516): dispatch_queue\n/usr/lib/python3.8/asyncio/events.py(81): _run\n/usr/lib/python3.8/asyncio/base_events.py(1859): _run_once\n/usr/lib/python3.8/asyncio/base_events.py(570): run_forever\n/usr/local/lib/python3.8/dist-packages/tornado/platform/asyncio.py(195): start\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelapp.py(736): start\n/usr/local/lib/python3.8/dist-packages/traitlets/config/application.py(1043): launch_instance\n/usr/local/lib/python3.8/dist-packages/ipykernel_launcher.py(17): <module>\n/usr/lib/python3.8/runpy.py(87): _run_code\n/usr/lib/python3.8/runpy.py(194): _run_module_as_main\n"
  }
  node {
    input: "3"
    output: "output"
    name: "Relu_1"
    op_type: "Relu"
    doc_string: "/usr/local/lib/python3.8/dist-packages/torch/nn/functional.py(1299): relu\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/activation.py(98): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1090): _slow_forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/tmp/ipykernel_1106822/2355770593.py(18): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1090): _slow_forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(118): wrapper\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(127): forward\n/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py(1102): _call_impl\n/usr/local/lib/python3.8/dist-packages/torch/jit/_trace.py(1166): _get_trace_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(388): _trace_and_get_graph_from_model\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(437): _create_jit_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(493): _model_to_graph\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(724): _export\n/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py(107): export\n/usr/local/lib/python3.8/dist-packages/torch/onnx/__init__.py(316): export\n/tmp/ipykernel_1106822/2355770593.py(27): <module>\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3508): run_code\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3448): run_ast_nodes\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3269): run_cell_async\n/usr/local/lib/python3.8/dist-packages/IPython/core/async_helpers.py(129): _pseudo_sync_runner\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3064): _run_cell\n/usr/local/lib/python3.8/dist-packages/IPython/core/interactiveshell.py(3009): run_cell\n/usr/local/lib/python3.8/dist-packages/ipykernel/zmqshell.py(546): run_cell\n/usr/local/lib/python3.8/dist-packages/ipykernel/ipkernel.py(422): do_execute\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(740): execute_request\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(412): dispatch_shell\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(505): process_one\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelbase.py(516): dispatch_queue\n/usr/lib/python3.8/asyncio/events.py(81): _run\n/usr/lib/python3.8/asyncio/base_events.py(1859): _run_once\n/usr/lib/python3.8/asyncio/base_events.py(570): run_forever\n/usr/local/lib/python3.8/dist-packages/tornado/platform/asyncio.py(195): start\n/usr/local/lib/python3.8/dist-packages/ipykernel/kernelapp.py(736): start\n/usr/local/lib/python3.8/dist-packages/traitlets/config/application.py(1043): launch_instance\n/usr/local/lib/python3.8/dist-packages/ipykernel_launcher.py(17): <module>\n/usr/lib/python3.8/runpy.py(87): _run_code\n/usr/lib/python3.8/runpy.py(194): _run_module_as_main\n"
  }
  name: "torch-jit-export"
  initializer {
    dims: 1
    dims: 1
    dims: 3
    dims: 3
    data_type: 1
    name: "conv.weight"
    raw_data: "\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?\000\000\200?"
  }
  initializer {
    dims: 1
    data_type: 1
    name: "conv.bias"
    raw_data: "\000\000\000\000"
  }
  input {
    name: "image"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "batch"
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_param: "height"
          }
          dim {
            dim_param: "width"
          }
        }
      }
    }
  }
  output {
    name: "output"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "batch"
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_param: "height"
          }
          dim {
            dim_param: "width"
          }
        }
      }
    }
  }
}