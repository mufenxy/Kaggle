import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def naive_bayes_demo():
    """
    朴素贝叶斯分类实例函数
    """
    # 构造二维分类数据
    X,y = make_classification(
        n_samples=200,n_features=2,n_classes=2,
        n_informative=2,n_redundant=0,random_state=42
    )
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    # 训练高斯朴素贝叶斯模型
    model = GaussianNB()
    model.fit(X_train,y_train)

    # 模型预测
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print(f"准确率：{acc:.2f}")

# 可视化分类边界
# 获取特征1的最小值减1和最大值加1，用于确定绘图的x轴范围（留出一定余量使图形更美观）
    X_min,X_max = X[:,0].min()-1,X[:,0].max()+1
# 获取特征2的最小值减1和最大值加1，用于确定绘图的y轴范围（留出一定余量使图形更美观）
    y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
# 生成网格点坐标矩阵：np.linspace在X_min到X_max间生成200个等间隔点，同理生成y轴点；np.meshgrid将两组点转换为网格矩阵xx（x坐标矩阵）和yy（y坐标矩阵）
    xx,yy = np.meshgrid(np.linspace(X_min,X_max,200),np.linspace(y_min,y_max,200))
# 对网格点进行预测：np.c_按列拼接xx.ravel()（将xx展平为一维数组）和yy.ravel()，形成所有网格点的特征矩阵；model.predict预测每个网格点的类别
    z = model.predict(np.c_[xx.ravel(),yy.ravel()])
# 将预测结果z的形状重塑为与网格矩阵xx相同的形状，以便后续绘制等高线
    z = z.reshape(xx.shape)

# 创建一个8x6英寸的图形对象
    plt.figure(figsize=(8,6))
# 绘制分类区域填充图：xx和yy为网格坐标，z为每个网格的类别，alpha=0.4设置透明度，cmap指定颜色映射（coolwarm为冷暖色）
    plt.contourf(xx,yy,z,alpha=0.4,cmap=plt.cm.coolwarm)
# 绘制训练集数据点：X_train[:,0]和X_train[:,1]为特征坐标，c=y_train按类别着色，marker='o'使用圆形标记，label='Train'设置图例标签
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,marker='o',label='Train')
# 绘制测试集数据点：参数含义类似训练集，marker='s'使用正方形标记，edgecolor='k'设置点的边缘为黑色，label='Test'设置图例标签
    plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker='s',edgecolor='k',label='Test')
# 设置x轴标签为"Feature 1"
    plt.xlabel("Feature 1")
# 设置y轴标签为"Feature 2"
    plt.ylabel("Feature 2")
# 设置图形标题为"Naive Bayes Classification"
    plt.title("Naive Bayes Classification")
# 显示图例（展示Train和Test对应的标记样式）
    plt.legend()
# 显示图形
    plt.show()
if __name__ == '__main__':
    naive_bayes_demo()