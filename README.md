# 最优路径
使用遗传算法来寻找最短路径问题

主要使用两种遗传算法：
1、使用select方法对新的种群进行挑选得到新的种群
2、挑选两个基因并进行对比，使用胜利的基因对失败的基因进行更改

得到结果分别如下图所展示：


![image](https://user-images.githubusercontent.com/72426381/129063885-c0b80167-b702-4190-852c-1686bed0815c.png)

由于没有固定每个地点的距离所以显示的图像稍有差异

![image](https://user-images.githubusercontent.com/72426381/129063921-3723b023-857f-4800-9c46-2650ddece37c.png)


3、使用进化策略1+1 ES方法实现最优路径优化，出现了些许问题，不是很容易收敛，思前想后大概是在基因进行遗传产生child那一步没法得到更全面的方案，在路径优化问题_ES.py文件中make_child方法实现了产生child两种类型的策略，但是效果都不是很理想。
因此如果想实现比较好的效果可能需要多训练。
![image](https://user-images.githubusercontent.com/72426381/129170284-e9ae9a4c-8742-4575-92df-7bef219539df.png)
