![Exported image](Exported%20image%2020260722114749-0.png)  

这里面有三个最重要的元件（图里面标注的）  
①：Text Encoder  
②：diffusion model  
③：Decoder解码器  
这里②生成的是中间产物，是真正输出图片的浓缩版本，后续Decoder的任务是只要把这个浓缩的信息全部展开，变成看得懂的图片即可
      

==Text Encoder==

![Exported image](Exported%20image%2020260722114753-1.png)  
![Exported image](Exported%20image%2020260722114755-2.png)  

在文生图模型里，把 text encoder 做大（用更强的 T5-XXL）比把扩散模型 U-Net 做大效果好得多！
      

==FID====的计算==：  
1.用==CNN====提取特征==，最后得到的是很高维度的一个向量（比如2048维）  
2.生成图这样处理，得到一堆向量，真实图也这样  
3.==向量在对应的高位坐标的分布看作是一团云==，生成图的就是生成云，真实图的就是真实云  
4.近似为高斯分布，计算两个高斯云的距离，重叠情况  
距离越小（两团云重叠越多、中心越近、形状越像） →  
FID 越低 → 生成质量越好。
   
![Exported image](Exported%20image%2020260722114756-3.png)  

CLIP的计算：  
CLIP的计算就比较简单粗暴，直接拿非常多的文字和图片训练（400M），训练出来text和image的编码器，目的就是根据训练好的数据，编码生成向量，向量越近就说明两==者越匹配==
      

==Decoder====解码器==
 
==1.====简单的尺寸变换==

![Exported image](Exported%20image%2020260722114801-4.png)        

==2.====其他类型==
   
![Exported image](Exported%20image%2020260722114811-5.png)

==VAE====：==

![Exported image](Exported%20image%2020260722114814-6.png)

这个实际上是一切的开端，我们首要的其实是训练出一个完整的VAE的体系

![Exported image](Exported%20image%2020260722114820-7.png)

接下来，我们不要动encoder和deconder了，所有的真实图片给generation model前，都用encoder编码压缩，其他的不变。

![Exported image](Exported%20image%2020260722114821-8.png)

这个样子下，扩散模型输出的就不是正常图片，而是这个VAE空间下可以处理的latent representation，这样最后的decoder就可以解码成正常的图片了。