## JIT Cache Invalidation

我们将Exec的Host page设置为Read-only，进入signal handler后检查对应的Guest page，发现是W&X就将这个执行SMC的block转换为解释执行（不再编译）。
在dispatcher内我们每写入一次内存就用 `addr & MASK` 得到page base，在block执行完毕后进行batch invalidation，这样只需要获取一次write lock。

对于`W^X`的系统，我们只需要在权限切换的那一刻进行invalidation，具体如下：
- RW -> RX: None （代码还没有被JIT执行过 OR 代码已经被invalidate）
- RX -> RW: Invalidate all blocks in the page （我们假设所有代码都会被修改）

使用类似Radix Tree的结构来存储一个Guest Page内的所有已编译JIT Cache，快速搜索并invalidate所有相关block。

执行的时候也可以走二级分表查找，目前是一个大的`HashMap<addr, addr>`，先保持可用。

对于Invalidation，我们可以利用现有的Late-linking slot机制，遍历JIT Arena里的Data段（需要按16bytes对齐 `(guest, host)` pair），找到需要invalidate的guest并清空slot。

## Execution Suspension

suspend的时候保存ucontext（rust再写一套跨平台context）到tls，然后修改exit_reason为特殊的值表示这个exit是由polling page引起的中断而不是jit invalidation导致的（后者无需恢复上下文），返回dispatcher后如果查询发现是这个中断就读取host特定的context，并走一个恢复用的专用汇编函数。
考虑到suspend thread这种调用情况很少，本身也比较重，这种slow path overhead可以接受。