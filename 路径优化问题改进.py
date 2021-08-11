import numpy as np
import matplotlib.pyplot as plt

pop_size = 500     #每次计算的方案数量，也就是路径优化的方案数量
n_epoch = 800     #算法遗传的次数
dna_size = 20     #二十个位置
cross_rate = 0.8      #交配比例
mutate_rate = 0.1     #编译比例

class GA:
    def __init__(self,pop_size,n_poch,dna_size,cross_rate,mutate_rate):
        self.pop_size = pop_size
        self.n_epoch = n_epoch
        self.dna_size = dna_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

        self.pop = np.vstack([np.random.permutation(self.dna_size) for i in range(self.pop_size)])
        self.city = np.random.rand(self.dna_size,2)      #随机生成坐标点
        print(self.pop.shape)

    def get_fitness(self,parent_1,parent_2):

        way_1 = self.city[parent_1]
        way_2 = self.city[parent_2]
        # print(way_1.shape,way_2.shape)

        x_1 = sum(np.sqrt(np.square(np.diff(way_1[:,0]))+np.square(np.diff(way_1[:,1]))))
        x_2 = sum(np.sqrt(np.square(np.diff(way_2[:,0]))+np.square(np.diff(way_2[:,1]))))
        return x_1>x_2

    def cross(self,winner,loser):
        inx = []
        for i in range(0,self.dna_size):
            if np.random.rand()<cross_rate:
                inx.append(True)
            else:
                inx.append(False)
        child_1 = winner[:-1][inx]
        child_2 = winner[:-1][np.isin(winner[:-1],child_1,invert=True)]


        loser = np.append(child_1,child_2)
        loser = np.append(loser,loser[0])
        # print(loser)
        return loser

    def mutate(self,loser):
        for i in range(1,self.dna_size):
            if np.random.rand()<mutate_rate:
                idx = np.random.randint(1,self.dna_size,1)
                loser[i],loser[idx] = loser[idx],loser[i]
        return loser

    def get_best(self):
        x = np.empty(self.pop.shape, dtype=np.float64)
        y = np.empty(self.pop.shape, dtype=np.float64)
        for i, d in enumerate(self.pop):
            distance = self.city[d]
            x[i, :] = distance[:, 0]
            y[i, :] = distance[:, 1]
        x_, y_ = x[:, 0].reshape(-1, 1), y[:, 0].reshape(-1, 1)

        # 下面这两句是计算是否返回原来地方的最优路径
        x = np.concatenate((x, x_), axis=1)
        y = np.concatenate((y, y_), axis=1)
        total_distance = np.empty(len(x), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(x, y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.dna_size * 2 / total_distance)
        num = self.pop[np.argmax(fitness)]
        num = [str(i) for i in num]
        num = '>'.join(num) + '>' + num[0]

        # print(' '.join(self.pop[np.argmax(fitness)]))
        return x[np.argmax(fitness)], y[np.argmax(fitness)], total_distance[np.argmax(fitness)], num

    def evolve(self,n):
        plt.ion()
        for ii in range(self.n_epoch):
            x_,y_,total_dis,num = self.get_best()
            plt.cla()
            plt.scatter(self.city[:,0],self.city[:,1],c='r')
            plt.plot(x_,y_,c='b')
            plt.text(0.1,0.7,'{0}'.format(num))
            plt.text(0.8,0.8,'{0}'.format(str(total_dis)))
            plt.title('num_{0} figure'.format(str(ii)))
            plt.pause(0.05)
            print('num_{0} best distance:{1}'.format(ii,str(total_dis)))
            for _ in range(n):
                idx_1 = np.random.randint(0, self.pop_size, 2)
                parent_1, parent_2 = self.pop[idx_1[0]], self.pop[idx_1[1]]
                parent_1,parent_2 = np.append(parent_1,parent_1[0]),np.append(parent_2,parent_2[0])
                if self.get_fitness(parent_1, parent_2):
                    winner, loser = parent_2, parent_1
                else:
                    winner, loser = parent_1, parent_2
                loser = self.cross(winner, loser)
                loser = self.mutate(loser)
                self.pop[idx_1[0]], self.pop[idx_1[1]] = winner[:-1], loser[:-1]
        plt.ioff()
        plt.show()
ga = GA(pop_size=pop_size,n_poch=n_epoch,dna_size=dna_size,cross_rate=cross_rate,mutate_rate=mutate_rate)
ga.evolve(200)