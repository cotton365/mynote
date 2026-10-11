**IOU**
 ![Exported image](Exported%20image%2020260722114911-0.png)

交并比可以衡量真实的图片上面的框范围和机器画出来的框的匹配程度  
对应代码  
_#@save_￼**def** box_iou(boxes1, boxes2):￼ _"""计算两个锚框或边界框列表中成对的交并比"""_￼ box_area = **lambda** boxes: ((boxes[:, 2] - boxes[:, 0]) *￼ (boxes[:, 3] - boxes[:, 1]))￼ _# boxes1,boxes2,areas1,areas2的形状:_￼ _# boxes1：(boxes1的数量,4),_￼ _# boxes2：(boxes2的数量,4),_￼ _# areas1：(boxes1的数量,),_￼ _# areas2：(boxes2的数量,)_￼ areas1 = box_area(boxes1)￼ areas2 = box_area(boxes2)￼ _# inter_upperlefts,inter_lowerrights,inters的形状:_￼ _# (boxes1的数量,boxes2的数量,2)_￼ inter_upperlefts = torch.max(boxes1[:, **None**, :2], boxes2[:, :2])￼ inter_lowerrights = torch.min(boxes1[:, **None**, 2:], boxes2[:, 2:])￼ inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)￼ _# inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)_￼ inter_areas = inters[:, :, 0] * inters[:, :, 1]￼ union_areas = areas1[:, **None**] + areas2 - inter_areas￼ **return** inter_areas / union_areas
 \> 来自 \<[https://zh.d2l.ai/chapter_computer-vision/anchor.html](https://zh.d2l.ai/chapter_computer-vision/anchor.html)\>     

锚框与真实边界框的匹配算法  
给定图像，假设锚框是 $A_{1} , A_{2} , \ldots , A_{n_{a}}$ ，真实边界框是 $B_{1} , B_{2} , \ldots , B_{n_{b}}$ ，其中 $n_{a} \geq n_{b}$ 。 让我们定义一个矩阵 $X \in R^{n_{a} \times n_{b}}$ ，其中第 $i$ 行、第 $j$ 列的元素 $x_{i j}$ 是锚框 $A_{i}$ 和真实边界框 $B_{j}$ 的IoU。 该算法包含以下步骤。   1. 在矩阵 $X$ "中找到最大的元素，并将它的行索引和列索引分别表示 $i_{1}$ "和 $j_{1}$ 。然后将真实边界框 $B_{j_{1}}$ 分配给锚框 $A_{i_{1}}$ 。这很直观，因为 $A_{i_{1}}$ 和 $B_{j_{1}}$ 是所有锚框和真实边界框配对中最相近的。在第一个分配完成后，丢弃矩阵中 $i_{1}^{t h}$ 行和 $j_{1}^{t h}$ 列中的所有元素。  
2. 在矩阵 $X$ "中找到剩余元素中最大的元素，并将它的行索引和列索引分别表示 $i_{2}$ "和 $j_{2}$ 。我们将真实边界框 $B_{j_{2}}$ 分配给锚框 $A_{i_{2}}$ ，并丢弃矩阵中 $i_{2}^{t h}$ 行和 $j_{2}^{t h}$ 列中的所有元素。  
3. 此时，矩阵 $X$ "中两行和两列中的元素已被丢弃。我们继续，直到丢弃掉矩 $X$ "中 $n_{b}$ 列中的所有元素。此时已经为这 $n_{b}$ 个锚框各自分配了一个真实边界框。  
4. 只遍历剩下的 $n_{a} - n_{b}$ 个锚框。例如，给定任何锚框 $A_{i}$ ，在矩阵 $X$ "的 $i^{t h}$ "行中找到与 $A_{i}$ 的IoU最大的真实边界框 $B_{j}$ ，只有当此IoU大于预定义的阈值时，才将 $B_{j}$ 分配给 $A_{i}$ 。

![Exported image](Exported%20image%2020260722114913-1.png)

代码：  
_#@save_￼**def** assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):￼ _"""将最接近的真实边界框分配给锚框"""_￼ num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]￼ _# 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU_￼ jaccard = box_iou(anchors, ground_truth)￼ _# 对于每个锚框，分配的真实边界框的张量_￼ anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,￼ device=device)￼ _# 根据阈值，决定是否分配真实边界框_￼ max_ious, indices = torch.max(jaccard, dim=1)￼ anc_i = torch.nonzero(max_ious \>= iou_threshold).reshape(-1)￼ box_j = indices[max_ious \>= iou_threshold]￼ anchors_bbox_map[anc_i] = box_j￼ col_discard = torch.full((num_anchors,), -1)￼ row_discard = torch.full((num_gt_boxes,), -1)￼ **for** _ **in** range(num_gt_boxes):￼ max_idx = torch.argmax(jaccard)￼ box_idx = (max_idx % num_gt_boxes).long()￼ anc_idx = (max_idx / num_gt_boxes).long()￼ anchors_bbox_map[anc_idx] = box_idx￼ jaccard[:, box_idx] = col_discard￼ jaccard[anc_idx, :] = row_discard￼ **return** anchors_bbox_map
 
**标记类别和偏移量**

![Exported image](Exported%20image%2020260722114914-2.png)

**对应代码**  
_#@save_￼**def** offset_boxes(anchors, assigned_bb, eps=1e-6):￼ _"""对锚框偏移量的转换"""_￼ c_anc = d2l.box_corner_to_center(anchors)￼ c_assigned_bb = d2l.box_corner_to_center(assigned_bb)￼ offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]￼ offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])￼ offset = torch.cat([offset_xy, offset_wh], axis=1)￼ **return** offset
   

后续就是编号环节，我们会将标记为背景的锚框记为0，其他的有效的框依次为1，2，3，4，·······  
_#@save_￼**def** multibox_target(anchors, labels):￼ _"""使用真实边界框标记锚框"""_￼ batch_size, anchors = labels.shape[0], anchors.squeeze(0)￼ batch_offset, batch_mask, batch_class_labels = [], [], []￼ device, num_anchors = anchors.device, anchors.shape[0]￼ **for** i **in** range(batch_size):￼ label = labels[i, :, :]￼ anchors_bbox_map = assign_anchor_to_bbox(￼ label[:, 1:], anchors, device)￼ bbox_mask = ((anchors_bbox_map \>= 0).float().unsqueeze(-1)).repeat(￼ 1, 4)￼ _# 将类标签和分配的边界框坐标初始化为零_￼ class_labels = torch.zeros(num_anchors, dtype=torch.long,￼ device=device)￼ assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,￼ device=device)￼ _# 使用真实边界框来标记锚框的类别。_￼ _# 如果一个锚框没有被分配，标记其为背景（值为零）_￼ indices_true = torch.nonzero(anchors_bbox_map \>= 0)￼ bb_idx = anchors_bbox_map[indices_true]￼ class_labels[indices_true] = label[bb_idx, 0].long() + 1￼ assigned_bb[indices_true] = label[bb_idx, 1:]￼ _# 偏移量转换_￼ offset = offset_boxes(anchors, assigned_bb) * bbox_mask￼ batch_offset.append(offset.reshape(-1))￼ batch_mask.append(bbox_mask.reshape(-1))￼ batch_class_labels.append(class_labels)￼ bbox_offset = torch.stack(batch_offset)￼ bbox_mask = torch.stack(batch_mask)￼ class_labels = torch.stack(batch_class_labels)￼ **return** (bbox_offset, bbox_mask, class_labels)
 \> 来自 \<[https://zh.d2l.ai/chapter_computer-vision/anchor.html](https://zh.d2l.ai/chapter_computer-vision/anchor.html)\>