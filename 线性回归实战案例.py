"""
情景：
小明发现一个规律：
假设自己学习1小时，则考试成绩为2分
假设自己学习2小时，则考试成绩为4分
假设自己学习3小时，则考试成绩为6分

请用线性回归的知识建立一个模型，判断学习时间与考试成绩之间的关系
并猜测小明学习4小时后的考试成绩
"""


# 第一步：定义数据集

# 定义数据特征
x_data = [1,2,3]
# 定义数据标签
y_data = [2,4,6]
# 初始化参数w
w = 4

# 第二步：定义线性回归模型
def forward(x):
    return x * w

# 第二步：定义损失函数
def cost(xs, ys):
    costvalue = 0
    for x,y in zip(xs, ys):
        y_pred = forward(x)
        costvalue += (y_pred - y)**2
    return costvalue / len(xs)

# 第三步：定义计算梯度的函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w -y)
    return grad / len(xs)

# 第四步：设置轮数
for epoch in range(100): # 假设训练100次
    # 计算按误差损失
    cost_val = cost(x_data, y_data)
    # 计算梯度
    grad_val = gradient(x_data, y_data)
    w = w - 0.01 * grad_val
    print(f"训练轮次：{epoch} w={w} loss={cost_val}")



print(f"小明学习4小时后，预计考试成绩为：{forward(4)}")




