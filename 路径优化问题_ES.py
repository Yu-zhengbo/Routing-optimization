import numpy as np
import matplotlib.pyplot as plt

dna_size = 20
n_epoch = 100
mut_strength = 20

class EA:
    def __init__(self,dna_size,n_epoch,mut_strength):
        self.dna_size = dna_size
        self.n_epoch = n_epoch
        self.mut_strength = mut_strength
        self.mutation = 0.5
        self.parent = np.random.permutation(self.dna_size)
        # print(self.parent)
        self.city = np.random.rand(self.dna_size,2)

    def get_fitness(self,parent):
        x = self.city[:,0][parent]
        y = self.city[:,1][parent]
        x = np.append(x,x[0])
        y = np.append(y,y[0])
        xs = np.diff(x)
        ys = np.diff(y)

        return 1/sum(np.sqrt(np.square(xs)+np.square(ys)))

    def make_child(self):
        kid = self.parent.copy()
        # for i in range(self.mut_strength):
        #     idx = np.random.choice(np.arange(self.dna_size),2,replace=False)
        #     # print(idx)
        #     kid[idx[0]],kid[idx[1]] = kid[idx[1]],kid[idx[0]]
        # return kid

        for i in range(self.dna_size):
            if np.random.rand()<self.mutation:
                idx = np.random.randint(0,self.dna_size,1)
                kid[i],kid[idx] = kid[idx],kid[i]
        return kid
        # idx = []
        # for i in range(self.dna_size):
        #     if np.random.rand<self.mutation:
        #         idx.append()


    def kill_child(self,parent,child):
        fit_parent = self.get_fitness(parent)
        fit_child = self.get_fitness(child)
        # print(fit_parent)
        # print(fit_child)
        if fit_child>fit_parent:
            self.mutation -= 0.01
            parent = child
        else:
            self.mutation += 0.01
        return parent

    def get_plot(self):
        x_ = self.city[self.parent][:,0]
        y_ = self.city[self.parent][:,1]
        x_ = np.append(x_, x_[0])
        y_ = np.append(y_, y_[0])
        xs = np.diff(x_)
        ys = np.diff(y_)
        plt.plot(x_,y_,c='b')
        plt.text(0.7,0.8,'total distance:%.4f'%sum(np.sqrt(np.square(xs)+np.square(ys))))

    def evolve(self):
        plt.ion()
        for i in range(self.n_epoch):
            print(str(i+1)+':mut_strength:'+str(self.mut_strength))
            print(str(i + 1) + ':mutation:' + str(self.mutation))
            plt.cla()
            plt.scatter(self.city[:,0],self.city[:,1],c='r')
            self.get_plot()
            plt.title('num_{0}_fig'.format(i+1))
            plt.pause(0.05)
            child = self.make_child()
            self.parent = self.kill_child(self.parent,child)


        plt.ioff()
        plt.show()






ea = EA(dna_size,n_epoch,mut_strength)
# ea.get_fitness(ea.parent)
ea.evolve()
# print(np.random.randint(-2,3,5))