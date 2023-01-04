# Homework 3

Public repository and stub/testing code for Homework 3 of 10-714.

* Part1: Python array operations
    * reshape: 只能处理compact的数组，只需要重新生成正常的strides即可
    * permute: 按照axes的顺序交换shape和strides
    * broadcast_to：只需要将shape为1的对应的stride改为0即可
    * getitem：根据切片形状求出shape，对每个stride都乘* step，然后修改offset为第一个元素出现的位置
* Part 5: CPU Backend - Matrix multiplication
    * AlignedDot只需要向编译器声明我使用的内存是对齐的，编译器会自动使用向量化加速。MatmulTiled中注意尽可能的“复用”
* Part 6: CUDA Backend - Compact and setitem
    * Compact 中根据gid求索引，用进制转化的思想
* Part 8: CUDA Backend - Reductions
    * 作业只要求用最暴力的迭代方法。sum、max等操作又叫扫描，有更高效的并行扫描算法，我的github中有[实现并行扫描](https://github.com/zzhbrr/parallel-scan-prefix-sum)。
* Part 9: CUDA Backend - Matrix multiplication
    * 直接用了并行计算课上实现的代码。基本思想是每个block都是16x16大小，将out划分为很多块，然后一个block内共同求这16*16大小的结果。当时有一些细节，卡了我很久。