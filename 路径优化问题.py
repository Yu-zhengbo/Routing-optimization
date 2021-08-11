import numpy as np
import matplotlib.pyplot as plt

city_position = np.random.rand(20,2)
# plt.scatter(city_position[:,0],city_position[:,1])
# plt.show()

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
    # def translateDNA(self):
    #     pass

    def get_fitness(self):
        x = np.empty(self.pop.shape,dtype=np.float64)
        y = np.empty(self.pop.shape, dtype=np.float64)
        for i,d in enumerate(self.pop):
            distance = self.city[d]
            x[i,:] = distance[:,0]
            y[i,:] = distance[:,1]
        x_,y_ = x[:,0].reshape(-1,1),y[:,0].reshape(-1,1)

        #下面这两句是计算是否返回原来地方的最优路径
        x = np.concatenate((x,x_),axis=1)
        y = np.concatenate((y,y_),axis=1)
        total_distance = np.empty(len(x),dtype=np.float64)
        for i,(xs,ys) in enumerate(zip(x,y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs))+np.square(np.diff(ys))))
        fitness = np.exp(self.dna_size*2/total_distance)
        num = self.pop[np.argmax(fitness)]
        num = [str(i) for i in num]
        num = '>'.join(num)+'>'+num[0]

        # print(' '.join(self.pop[np.argmax(fitness)]))
        return fitness,x[np.argmax(fitness)],y[np.argmax(fitness)],total_distance[np.argmax(fitness)],num



    def select(self):
        fitness,x_,y_,_ ,_= self.get_fitness()
        print(fitness.shape)

        idx = np.random.choice(np.arange(len(self.pop)),self.pop_size,replace=True,p=fitness/fitness.sum())

        return self.pop[idx]

    def cross(self,parent,pop):
        if np.random.rand()<self.cross_rate:
            i_ = np.random.randint(0,len(pop),size=1)
            idx_ = np.random.randint(0,2,self.dna_size).astype(np.bool)
            child_1 = parent[idx_]
            child_2 = pop[i_,np.isin(pop[i_].ravel(),child_1,invert=True)]
            print(child_2.shape,child_1.shape)
            parent = np.hstack((child_1,child_2))
        return parent

    def mutate(self,child):
        for i in range(self.dna_size):
            if np.random.rand() < self.mutate_rate:
                swap = np.random.randint(0,self.dna_size,size=1)
                child[i],child[swap] = child[swap],child[i]
        return child



    def evolve(self):
        plt.ion()
        for i in range(self.n_epoch):
            fitness, x_best, y_best,dis,num = self.get_fitness()
            print(num)
            self.pop = self.select()
            plt.cla()
            plt.scatter(self.city[:, 0], self.city[:, 1], c='r')
            plt.plot(x_best, y_best)
            plt.text(0, 0.7, '{0}'.format(num))
            plt.text(0.8,0.8,'{0}'.format(str(dis)))
            plt.pause(0.05)
            pop_copy = self.pop.copy()
            pop_new = self.pop.copy()

            for child in self.pop:
                child = self.cross(child,pop_copy)
                pop_new = np.vstack((pop_new,child))
            # self.pop = pop_new[self.pop_size:,:]
            self.pop = pop_new
        plt.ioff()
        plt.show()

ga = GA(pop_size=pop_size,n_poch=n_epoch,dna_size=dna_size,cross_rate=cross_rate,mutate_rate=mutate_rate)
ga.evolve()

