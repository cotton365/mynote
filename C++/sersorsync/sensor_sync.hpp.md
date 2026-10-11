### 34 inline
表示“允许重复定义，解决链接层面的多重定义问题”
### 39 auto
auto by_time = [](const SensorData& a, const SensorData& b) {
      if (a.timestamp_us != b.timestamp_us) {
          return a.timestamp_us < b.timestamp_us;
      }
      return a.sequence < b.sequence;
  };

  拆语法:

  - [] — 捕获列表,空 = 不抓外部变量
  - (const SensorData& a, ...) — 参数:两个 SensorData 的 const 引用。const = 只读不改;& = 不复制,直接用原对象
  - { ... } — 函数体
  - auto — 类型自动推断。"by_time 的类型就是右边那坨东西的类型"

  关键误解纠正: 它不是"输出 a 的比 b 小"。它是一个函数,参数 a、b,返回 bool:a 是否该排在 b 前面。排第一位的 if
  是规则:"时间戳不同,就比时间戳;相同才轮到比序号"。

  调用方式:by_time(x, y) 返回 true/false。你之前没见过 auto x = [](...) {...} 这种写法,所以整块读不懂。

### 45 sort
  std::sort(cameras.begin(), cameras.end(), by_time);

  "把 cameras 从 begin 到 end 重新排,排法按 by_time 判断谁该在前"。三个参数:开头、结尾、比较器。
  比较器 = "小于函数":by_time(a, b) 为 true 意思是 a 应该排在 b 前面。sort 内部反复调用它比较任意两元素,直到整个数组按它升序。
  为什么这样写: 默认 sort 用 < 排数字。但 SensorData 不是数字,排哪个字段?by_time
  就是自定义规则:按时间戳升序,时间戳相同按序号。这一行就是架构里"按相机时间戳升序"的代码实现。


### 50