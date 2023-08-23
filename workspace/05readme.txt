05dataset-test.py并不算独立的py文件
1） 仍需要bevfusion的环境才能运行
2） 第710行、第73行仍然需要bevfusion的数据、yaml配置文件路径。

好处：能够将dataset、dataloader一定意义上独立出来，便于单独观察二者的流程，熟悉pipeline。