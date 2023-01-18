import datetime 
from lib import simulate, MixedRequestGenerator

startTime = datetime.datetime.now()
# 生成请求并绘制分布
mixedRequestGenerator = MixedRequestGenerator()
mixedRequestGenerator.plotDistribute(save=True)

# 仿真并记录数据
for method in ["FIFO", "SSF", "SO"]:
    for serverNum in [4, 20, 40, 60, 80, 48, 52]:
        simulate(method=method, serverNum=serverNum)
        print("==========================================================")

endTime = datetime.datetime.now()
print(f"运行时间：{endTime - startTime}")