from functools import reduce, cache, cached_property

from colorama import Fore, Style
from scipy.stats import expon
import simpy
from simpy.core import Environment
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  #MacOS自带字体
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

SIMULATION_TIME = 1440
REFER_RATIO = 0.1
RANDOM_SEED = 123
ACCURACY = 0.01

# 用每分钟表达的到达率、服务率
LAMBDA_LIST = [30, 20, 15, 12]
MU_LIST = [3, 2.4, 2, 1.71]
# 用每小时表达的到达率、服务率
LAMBDA_LIST = [each/60 for each in LAMBDA_LIST]
MU_LIST = [each/60 for each in MU_LIST]

KIND_LIST = ['A', 'B', 'C', 'D']

ALPHA = 0.1
BETA = 0.4
CHI = 2


def customPrint(actualTime, message, tip=None):
    if tip is None:
        print(Fore.YELLOW + f"{format(actualTime,'.2f')}: " + Style.RESET_ALL, message)
    else:
        print(Fore.YELLOW + f"{format(actualTime,'.2f')}: " + Style.RESET_ALL, message, 
        Fore.GREEN + "\t"*3 + f"[{tip}]" + Style.RESET_ALL)

def generate_random_color():
    r = np.random.random()
    g = np.random.random()
    b = np.random.random()
    return (r,g,b)


class Request:
    def __init__(self, intervalTime, arrivalTime, serviceTime, kind: str, requestId: int):
        self.intervalTime = intervalTime
        self.arrivalTime = arrivalTime
        self.serviceTime = serviceTime
        self.kind = kind
        self.requestId = requestId

        self._serverId = None
        self._isScheduled = False # 被安排
        self._scheduledTime = np.nan
        self.inProcess = False # 服务中
        self.processingTime = np.nan # 开始处理的时刻
        self.isServed = False # 服务完成
        self._finishTime = np.nan
        
        # NOTE: 对于SO，arrivalTime非常接近scheduledTime
        self._waitTime = SIMULATION_TIME - self.arrivalTime

    @property
    def isScheduled(self):
        return self._isScheduled

    @property
    def serverId(self):
        return self._serverId

    @serverId.setter
    def serverId(self, id):
        self._serverId = id

    @property
    def finishTime(self):
        return self._finishTime
    
    @finishTime.setter
    def finishTime(self, time):
        self._finishTime = time
        self._waitTime = self._finishTime - (self.arrivalTime + self.serviceTime)

    @property
    def waitTime(self):
        return self._waitTime

    @property
    def scheduledTime(self):
        return self._scheduledTime

    @scheduledTime.setter
    def scheduledTime(self, time):
        self._isScheduled = True
        self._scheduledTime = time


class RequestGenerator:
    def __init__(self, lambda_, mu, kind):
        self.lambda_ = lambda_
        self.mu = mu
        self.kind = kind
        self.rvIntervalTime = expon(scale=1/lambda_)
        self.rvServiceTime = expon(scale=1/mu)

    @cache
    def generateRequestList(self):
        np.random.seed(RANDOM_SEED)
        size = int(SIMULATION_TIME * self.lambda_ * 1.1) # 留余量，待截取
        intervalTimeArray = self.rvIntervalTime.rvs(size=size)
        arrivalTimeArray = intervalTimeArray.cumsum()
        serviceTimeArray = self.rvServiceTime.rvs(size=size)

        # NOTE: 截断仿真结束时还未生成的请求
        num = np.where(arrivalTimeArray <= SIMULATION_TIME)[0][-1]

        return [ Request(intervalTimeArray[i], arrivalTimeArray[i], serviceTimeArray[i], self.kind, i) 
            for i in range(num) ]


class MixedRequestGenerator:
    def __init__(self, lambda_list=LAMBDA_LIST, mu_list=MU_LIST, kind_list=KIND_LIST):
        self.requestGeneratorList = []
        self.requestLists = []

        for lambda_, mu, kind in zip(lambda_list, mu_list, kind_list):
            requestGenerator = RequestGenerator(lambda_, mu, kind)
            self.requestGeneratorList.append(requestGenerator)
            requestList = requestGenerator.generateRequestList()
            self.requestLists.append(requestList)
        
        mixedRequestList = reduce(lambda x,y: x + y, self.requestLists)
        self.mixedRequestList = sorted(mixedRequestList, key=lambda x: x.arrivalTime)

    def plotDistribute(self, save=False):
        for ind in range(len(KIND_LIST)):
            kind = KIND_LIST[ind]
            intervalTimeList = [request.intervalTime for request in self.requestLists[ind]]
            serviceTimeList = [request.serviceTime for request in self.requestLists[ind]]
            self.plotSingleDistribute(kind, intervalTimeList, serviceTimeList, save=save)

        kind = "Mixed"
        intervalTimeList = [request.intervalTime for request in self.mixedRequestList]
        serviceTimeList = [request.serviceTime for request in self.mixedRequestList]
        self.plotSingleDistribute(kind, intervalTimeList, serviceTimeList, save=save)

    def plotSingleDistribute(self, kind, intervalTimeList, serviceTimeList, save=False):
        """TODO:绘制每个请求的kde和总的hist"""
        with plt.style.context('_mpl-gallery'):
            colors = [ generate_random_color() for i in range(4) ]
            fig, axes = plt.subplots(2, 2)
            axes[0,0].hist(intervalTimeList, color=colors[0])
            axes[0,0].set_xlabel("间隔时间")
            axes[0,0].set_ylabel("频数")
            sns.kdeplot(intervalTimeList, ax=axes[0,1], color=colors[1])
            axes[0,1].set_xlabel("间隔时间")
            axes[0,1].set_ylabel("概率密度")
            axes[1,0].hist(serviceTimeList, color=colors[2])
            axes[1,0].set_xlabel("服务时间")
            axes[1,0].set_ylabel("频数")
            sns.kdeplot(serviceTimeList, ax=axes[1,1], color=colors[3])
            axes[1,1].set_xlabel("服务时间")
            axes[1,1].set_ylabel("概率密度")
            fig.set_size_inches(20,12)
            fig.suptitle(f"{kind}类请求的间隔时间和服务时间分布", fontsize=20)
            fig.tight_layout()
            if save:
                plt.savefig(f"distribute/distribute{kind}.png")
            else:
                plt.show()


class Server:
    def __init__(self, env: Environment, serverId: int, maxQueueLength=None, prioritized=False):
        self._queue = []
        self.env = env
        self.serverId = serverId
        self.maxQueueLength = maxQueueLength
        self.prioritized = prioritized

    def isEmpty(self):
        if len(self._queue) == 0:
            return True
        else:
            return False

    def isFull(self):
        if self.maxQueueLength is None:
            return False
        if len(self._queue) == self.maxQueueLength:
            return True
        else:
            return False

    @property
    def queueLength(self):
        return len(self._queue)

    @property
    def currentRequest(self):
        if self.isEmpty():
            return None
        else:
            if not self.prioritized:
                return self._queue[0]
            else:
                fastFirstRequest = min(self._queue, key=lambda request: request.kind)
                return fastFirstRequest

    def run(self):
        while True:
            if not self.isEmpty():
                self.currentRequest.inProcess = True
                self.currentRequest.processingTime = self.env.now
                yield self.env.timeout(self.currentRequest.serviceTime)
                self.currentRequest.inProcess = False
                self.currentRequest.finishTime = self.env.now
                self.currentRequest.isServed = True
                kind = self.currentRequest.kind
                requestId = self.currentRequest.requestId
                # customPrint(self.env.now, f"request-{kind}{requestId}完成于server-{self.serverId}", 
                #     f"若无延迟: {format(self.currentRequest.arrivalTime + self.currentRequest.serviceTime, '.2f')}")
                self.remove(self.currentRequest)
            else:
                yield self.env.timeout(ACCURACY)

    def append(self, request: Request):
        self._queue.append(request)

    def remove(self, request: Request):
        self._queue.remove(request)

    def __len__(self):
        return len(self._queue)


class Cloud:
    def __init__(self, env: Environment, serverNum, method: str="FIFO"):
        # 方法可选：FIFO, SSF, 其他(SO)
        self.env = env
        self.serverNum = serverNum
        self.method = method
        if self.method == "FIFO" or self.method == "SSF":
            self.serverList = [Server(env, serverId=i, maxQueueLength=1) for i in range(serverNum)]
        else:
            self.serverList = [Server(env, serverId=i) for i in range(serverNum)]
        self.mixedRequestGenerator = MixedRequestGenerator()
        self.recordInterval = 1
        self.queueLengthList = []
        self.averageWaitTimeList = []

    def getRequestQueue(self):
        """分配器端的队列"""
        requestList = self.mixedRequestGenerator.mixedRequestList
        return [request for request in requestList if request.arrivalTime < self.env.now and not request.isScheduled]

    def getServerRequestQueue(self):
        """服务器端的队列"""
        queueList = []
        for server in self.serverList:
            if len(server) > 1:
                queue = server._queue[1:]
            else:
                queue = []
            queueList.append(queue)
        return queueList

    def record(self):
        while True:
            yield self.env.timeout(1)
            if self.method == "FIFO" or self.method == "SSF":
                queueLength = len(self.getRequestQueue())
            else:
                queueLength = sum([len(queue) for queue in self.getServerRequestQueue()])
            self.queueLengthList.append(queueLength)

            now = self.env.now
            arrivedRequestList = [request for request in self.mixedRequestGenerator.mixedRequestList if request.arrivalTime < now ]
            # 处理完成的列表
            servedWaitTimeList = [ request.waitTime for request in arrivedRequestList if request.isServed]
            # 正在处理的列表
            processingWaitTimeList = [ request.processingTime - request.arrivalTime for request in arrivedRequestList if request.inProcess and not request.isServed ]
            # 已分配，未处理
            unProcessingWaitTimeList = [ now - request.arrivalTime for request in arrivedRequestList if not request.inProcess and not request.isServed and request.isScheduled ]
            # 还未被分配的列表
            unScheduledWaitTimeList = [ now - request.arrivalTime for request in arrivedRequestList if not request.isScheduled ]
            waitTimeList = servedWaitTimeList + processingWaitTimeList + unProcessingWaitTimeList + unScheduledWaitTimeList
            if len(waitTimeList)==0:
                self.averageWaitTimeList.append(0)
            else:
                self.averageWaitTimeList.append(np.mean(waitTimeList))

    def arrive(self):
        lastArrivalTime = 0
        for request in self.mixedRequestGenerator.mixedRequestList:
            kind = request.kind
            requestId = request.requestId

            arrivalTime = request.arrivalTime
            mixedIntervalTime = arrivalTime - lastArrivalTime
            yield self.env.timeout(mixedIntervalTime)
            lastArrivalTime = arrivalTime

            # customPrint(self.env.now, f"request-{kind}{requestId}到达")

    def schedule(self):
        if self.method == "FIFO":
            while True:
                requestQueue = self.getRequestQueue()
                for request in requestQueue:
                    kind = request.kind
                    requestId = request.requestId
                    while True:
                        for server in self.serverList:
                            if not server.isFull():
                                server.append(request)
                                request.scheduledTime = self.env.now
                                request.serverId = server.serverId
                                # customPrint(self.env.now, f"request-{kind}{requestId}委派给server-{server.serverId}")
                                break
                        if request.isScheduled:
                            break
                        yield self.env.timeout(ACCURACY)
                yield self.env.timeout(ACCURACY)
        elif self.method == "SSF":
            while True:
                requestQueue = self.getRequestQueue()
                if len(requestQueue) > 0:
                    fastFirstRequest = min(requestQueue, key=lambda request: request.kind)
                    kind = fastFirstRequest.kind
                    requestId = fastFirstRequest.requestId
                    while True:
                        for server in self.serverList:
                            if not server.isFull():
                                server.append(fastFirstRequest)
                                fastFirstRequest.scheduledTime = self.env.now
                                fastFirstRequest.serverId = server.serverId
                                # customPrint(self.env.now, f"request-{kind}{requestId}委派给server-{server.serverId}")
                                break
                        if fastFirstRequest.isScheduled:
                            break
                        yield self.env.timeout(ACCURACY)
                else:
                    yield self.env.timeout(ACCURACY)
        else:
            while True:
                requestQueue = self.getRequestQueue()
                if len(requestQueue) > 0:
                    firstRequest = requestQueue[0]
                    kind = firstRequest.kind
                    requestId = firstRequest.requestId

                    # 获取最佳服务器ID
                    valueList = []
                    for server in self.serverList:
                        serverId = server.serverId
                        requestQueue = self.getServerRequestQueue()[serverId]
                        L = len(requestQueue)
                        mixedRequestList = self.mixedRequestGenerator.mixedRequestList

                        endTime = self.env.now
                        fromTime = endTime - SIMULATION_TIME*REFER_RATIO
                        referredScheduledRequestList = [request for request in mixedRequestList 
                            if request.serverId == serverId and
                            request.scheduledTime < endTime and request.scheduledTime > fromTime ]
                        referredServedRequestList = [request for request in mixedRequestList 
                            if request.serverId == serverId and
                            request.finishTime < endTime and request.finishTime > fromTime ]

                        if len(referredServedRequestList) == 0:
                            E = 0
                        else:
                            E = np.mean([request.serviceTime for request in referredServedRequestList])

                        if len(referredServedRequestList) == 0:
                            U = 0
                        else:
                            U = len(referredScheduledRequestList)/len(referredServedRequestList)
                        value = ALPHA * E + BETA * L + CHI * U
                        valueList.append(value)
                        selectedServerId = valueList.index(min(valueList))
                    # 获得最佳服务器
                    for server in self.serverList:
                        if server.serverId == selectedServerId:
                            selectedServer = server

                    selectedServer.append(firstRequest)
                    firstRequest.scheduledTime = self.env.now
                    firstRequest.serverId = selectedServerId
                    # customPrint(self.env.now, f"request-{kind}{requestId}委派给server-{selectedServerId}")
                yield self.env.timeout(ACCURACY)

    def start(self):
        # 运行服务
        for server in self.serverList:
            self.env.process(server.run())
        # 生成请求
        self.env.process(self.arrive())
        # 分配请求
        self.env.process(self.schedule())
        # 记录数据
        self.env.process(self.record())

    def exportRecord(self):
        mixedRequestList = self.mixedRequestGenerator.mixedRequestList
        DataFrame({
            "id": [request.kind+str(request.requestId) for request in mixedRequestList],
            "serverId": [request.serverId for request in mixedRequestList],
            "isScheduled": [request.isScheduled for request in mixedRequestList],
            "inProcess": [request.inProcess for request in mixedRequestList],
            "isServed": [request.isServed for request in mixedRequestList],
            "arrivalTime": [round(request.arrivalTime,2) for request in mixedRequestList],
            "scheduledTime": [round(request.scheduledTime,2) for request in mixedRequestList],
            "processingTime": [request.processingTime for request in mixedRequestList],
            "finishTime": [round(request.finishTime,2) for request in mixedRequestList],
            "serviceTime": [round(request.serviceTime,2) for request in mixedRequestList],
            "waitTime": [ round(request.waitTime,2) for request in mixedRequestList],
        }).to_csv(f"record/record{self.method}-{self.serverNum}.csv")
        for ind in range(len(KIND_LIST)):
            kind = KIND_LIST[ind]
            requestList = self.mixedRequestGenerator.requestLists[ind]
            DataFrame({
            "id": [request.kind+str(request.requestId) for request in requestList],
            "serverId": [request.serverId for request in requestList],
            "isScheduled": [request.isScheduled for request in requestList],
            "inProcess": [request.inProcess for request in requestList],
            "isServed": [request.isServed for request in requestList],
            "arrivalTime": [round(request.arrivalTime,2) for request in requestList],
            "scheduledTime": [round(request.scheduledTime,2) for request in requestList],
            "processingTime": [request.processingTime for request in requestList],
            "finishTime": [round(request.finishTime,2) for request in requestList],
            "serviceTime": [round(request.serviceTime,2) for request in requestList],
            "waitTime": [ round(request.waitTime,2) for request in requestList],
            }).to_csv(f"record/record{self.method}-{self.serverNum}{kind}.csv")

    def plotAverageWaitTime(self, save=False):
        fig = plt.figure(figsize=(20,12))
        with plt.style.context('ggplot'):
            plt.plot(self.averageWaitTimeList)
            plt.xlabel(r"时间($min$)")
            plt.ylabel(r"等待时间($min$)")
            plt.title(f"平均等待时间-{self.method}-{self.serverNum}", fontsize=20)
            if save:
                plt.savefig(f"waitTime/waitTime-{self.method}-{self.serverNum}.png")
            else:
                plt.show()

    def getResults(self):
        mixedRequestList = self.mixedRequestGenerator.mixedRequestList
        # 保存记录
        self.exportRecord()
        # 绘制等待时间图
        self.plotAverageWaitTime(save=True)
        # 打印参数和结果
        mixedRequestNum = len(mixedRequestList)
        servedMixedRequestList = [request for request in mixedRequestList if request.isServed]
        servedMixedRequestNum = len(servedMixedRequestList)
        meanWaitTime = np.mean([request.waitTime for request in mixedRequestList])
        meanQueueLength = np.mean(self.queueLengthList)
        print(f"优化方法: {self.method}")
        print(f"服务器数: {self.serverNum}")
        print(f"仿真时间: {SIMULATION_TIME}")
        for ind in range(len(KIND_LIST)):
            kind = KIND_LIST[ind]
            lambda_ = LAMBDA_LIST[ind]
            mu = MU_LIST[ind]
            requestNum = len(self.mixedRequestGenerator.requestLists[ind])
            print(f"请求生成器{kind}", end=": ")
            print(f"参数: 到达率-{format(lambda_,'.2f')}, 服务率-{format(mu,'.2f')}, 生成请求数-{requestNum}")
        print(f"平均等待时间: {format(meanWaitTime,'.2f')}")
        print(f"平均队列长度: {format(meanQueueLength, '.2f')}")
        print(f"总服务客户数: {servedMixedRequestNum}/{mixedRequestNum}")


def simulate(method="FIFO", serverNum=4):
    env = simpy.Environment()
    cloud = Cloud(env, serverNum=serverNum, method=method)
    cloud.start()
    env.run(until=SIMULATION_TIME)
    cloud.getResults()


if __name__ == "__main__":
    simulate(method="SSF", serverNum=80)

