# ==关于====diffusion model====而言他跟====CNN====有些不同的一点在于==

# ==卷积神经网络它是训练卷积核里的参数然后在不断下采样提取出来最终的分类结果或者是打分情况==

# ==但是====diffusion model====是生成模型（生成一个图这种），而不是打分或分类==
    ![Exported image](Exported%20image%2020260722114727-0.png)

通过严重程度来denoise

![Exported image](Exported%20image%2020260722114729-1.png)

predictor通过图片和step编号（对应严重程度）生成估计的noise，然后用预测的noise去噪  
目标在于训练predictor

![Exported image](Exported%20image%2020260722114732-2.png)

图片手动加噪，进行diffusion process

![Exported image](Exported%20image%2020260722114736-3.png)

这里有了数据集，就可以训练了
   

==接下来加入文字，我们说一段话，他就能生成图片==  
我们需要足够多的数据集，像  
LAION-5B 数据集（约58亿对）  
从而得到几十亿对“图片 + 自然语言描述”（caption）
 ![Exported image](Exported%20image%2020260722114738-4.png)

我们denoise的时候，只要加上文字输入，predicter要看图片+文字叙述+step编号，输出噪声，然后减掉后去噪

![Exported image](Exported%20image%2020260722114740-5.png)