**采样实现：**  
**一** **初始化：**  
_#@save_￼**def** multibox_prior(data, sizes, ratios):￼ _"""生成以每个像素为中心具有不同形状的锚框"""_￼ in_height, in_width = data.shape[-2:]￼ device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)￼ boxes_per_pixel = (num_sizes + num_ratios - 1)￼ size_tensor = torch.tensor(sizes, device=device)￼ ratio_tensor = torch.tensor(ratios, device=device)￼
 
**①获取输入图片尺寸****,****记为****in_height in_xidth**
 
**②提取****r,s****基本信息，这里一堆****r s****都是自己提前写好的参数**
 
**第三行：每个像素生成多少个** **anchor****？**
 ![Exported image](Exported%20image%2020260722114852-0.png)  

**为什么不用全组合？** 全组合会产生大量形状极端的 anchor（比如又小又扁），这些 anchor 几乎不可能匹配真实目标，反而增加计算量和训练难度。
 
**第四、五行：把列表转成张量**为后续的**向量化批量计算**做准备  
size_tensor = torch.tensor(sizes, device=device)￼ratio_tensor = torch.tensor(ratios, device=device)￼
         

**二** **初始化****anchor box****：**  
_# 为了将锚点移动到像素的中心，需要设置偏移量。_￼ _# 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5_￼ offset_h, offset_w = 0.5, 0.5￼ steps_h = 1.0 / in_height _# 在y轴上缩放步长_￼ steps_w = 1.0 / in_width _# 在x轴上缩放步长_￼  
_# 生成锚框的所有中心点_￼ center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h￼ center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w￼ shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')￼ shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)￼  
_# 生成“boxes_per_pixel”个高和宽，_
 ![Exported image](Exported%20image%2020260722114853-1.png)   
_# 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)__￼_ _w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[__0__]),__￼_ _sizes[__0__] * torch.sqrt(ratio_tensor[__1__:])))\__￼_ _* in_height / in_width # 处理矩形输入__￼_ _h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[__0__]),__￼_ _sizes[__0__] / torch.sqrt(ratio_tensor[__1__:])))_
 ![Exported image](Exported%20image%2020260722114856-2.png)

根据组合计算出anchor box 的w h

![Exported image](Exported%20image%2020260722114857-3.png)   
_# 除以2来获得半高和半宽_￼ anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(￼ in_height * in_width, 1) / 2
 ![Exported image](Exported%20image%2020260722114859-4.png)

注意这里是只有偏移量，w h 中心点还没有出现
 ![Exported image](Exported%20image%2020260722114902-5.png)     

_# 每个中心点都将有“boxes_per_pixel”个锚框，_￼ _# 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次_￼ out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],￼ dim=1).repeat_interleave(boxes_per_pixel, dim=0)￼ output = out_grid + anchor_manipulations￼ **return** output.unsqueeze(0)  
基于已经计算好的中心点

![Exported image](Exported%20image%2020260722114904-6.png)

这里才能得到真正的 $\left(x_{1},y_{1},x_{2},y_{2}\right)$ 。
       
为了显示以图像中以某个像素为中心的所有锚框，定义下面的==show_bboxes==函数来在图像上绘制多个边界框。
    
_#@save_￼**def** show_bboxes(axes, bboxes, labels=**None**, colors=**None**):￼ _"""显示所有边界框"""_￼ **def** _make_list(obj, default_values=**None**):￼ **if** obj **is** **None**:￼ obj = default_values￼ **elif** **not** isinstance(obj, (list, tuple)):￼ obj = [obj]￼ **return** obj￼  
_make_list函数的作用就是将输入的obj（也就是标签和框的颜色等信息统一转换为列表）
   

labels = _make_list(labels)￼ colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c']) 初始化标签和颜色
   

**for** i, bbox **in** enumerate(bboxes):￼ color = colors[i % len(colors)] 颜色循环使用￼ rect = d2l.bbox_to_rect(bbox.detach().numpy(), color) ￼ axes.add_patch(rect) 画框，贴图片上面
   

**if** labels **and** len(labels) \> i:￼ text_color = 'k' **if** color == 'w' **else** 'w'￼ axes.text(rect.xy[0], rect.xy[1], labels[i],￼ va='center', ha='center', fontsize=9, color=text_color,￼ bbox=dict(facecolor=color, lw=0))  
添加文字标签