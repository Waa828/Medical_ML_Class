import numpy as np
from scipy.integrate import odeint # 你可能需要安装scipy, pip install scipy
import matplotlib.pyplot as plt

# 定义动力学反应模型函数
def kinetics(condition):
    """
    计算竞争副反应中目标产物的产率。
    参数:
    - condition: 包含 [T, tre, Ccat] 的列表，其中：
        - T: 温度 (°C)
        - tre: 反应时间 (min)
        - Ccat: 催化剂浓度 (mM)
    返回:
    - yield_R: 目标产物 R 的产率
    """
    # 反应物的初始浓度 (单位: M)
    CA0 = 0.167  # 反应物 A 的初始浓度 (M)
    CB0 = 0.250  # 反应物 B 的初始浓度 (M)
  
    # 主反应和竞争副反应的反应参数
    AR = 3.1e7   # 主反应的指前因子 (L^(1/2) mol^(−3/2) s^(−1))
    EAR = 55     # 主反应的活化能 (kJ/mol)
    EAS = 100    # 副反应的活化能 (kJ/mol)
    As = 1e12    # 副反应的指前因子 (L^(1/2) mol^(−3/2) s^(−1))
  
    # 解包条件：温度 (°C)、停留时间 (min) 和催化剂浓度 (mM)
    T = condition[0]
    tre = condition[1]
    Ccat = condition[2]

    # 单位转换：
    T = T + 273.15     # 将温度从摄氏度转换为开尔文
    tre = tre * 60     # 将停留时间从分钟转换为秒
    Ccat = Ccat / 1000 # 将催化剂浓度从 mM 转换为 g/L

    # 使用阿伦尼乌斯方程计算速率常数
    R = 8.314 # 气体常数 R J/(mol*K)
    kr = Ccat ** 0.5 * AR * np.exp(-EAR / (T * R / 1000))  # 主反应的速率常数
    ks = As * np.exp(-EAS / (T * R / 1000))                # 副反应的速率常数

    # 定义表示反应动力学的 ODE 系统
    def reaction(w, time):
        # 解包浓度：a = [A], b = [B], c = [产物 R], d = [副产物 S]
        a, b, c, d = w
  
        # 定义每个物种的变化率：
        f1 = -kr * a * b             # 反应物 A 的消耗率（仅主反应）
        f2 = -kr * a * b - ks * b    # 反应物 B 的消耗率（主反应和副反应）
        f3 = kr * a * b              # 产物 R 的生成率（主反应）
        f4 = ks * b                  # 副产物 S 的生成率（副反应）
        return [f1, f2, f3, f4]

    # 处理时间为0的情况
    if tre <= 0:
        return 0.0

    # 为数值积分创建时间数组
    # 模拟从 0 到 tre/10 秒，时间步长为 0.001 秒。
    # 注意：使用 tre/10 是一种建模选择，可能代表全部停留时间的一部分。
    time = np.arange(0, tre / 10, 0.001)

    # 确保时间数组不为空
    if len(time) == 0:
        return 0.0

    # 使用给定的初始条件求解 ODE 系统：
    # [A] 从 CA0 开始，[B] 从 CB0 开始，产物 [R] 和 [S] 均从 0 开始。
    re = odeint(reaction, (CA0, CB0, 0.0, 0.0), time)

    # 从模拟结果中提取产物 R 的最终浓度。
    # re[-1, :] 给出最后一个时间点；第三个元素（索引 2）对应 [R]。
    cr = re[-1, :][2]

    # 计算产物 R 相对于反应物 A 初始浓度的产率
    y = cr / CA0
  
    # 将计算出的产率四舍五入到小数点后4位
    yield_R = round(y, 4)
    return yield_R


def plot_yield_vs_time():
    """
    任务1：当温度为60°C、催化剂浓度为3 mM时，绘制反应产率随反应时间的变化曲线。
    """
    T = 60  # °C
    Ccat = 3  # mM

    # 反应时间范围：0 到 120 分钟
    time_range = np.linspace(0, 120, 100)
    yields = []

    for tre in time_range:
        condition = [T, tre, Ccat]
        y = kinetics(condition)
        yields.append(y)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(time_range, yields, 'b-', linewidth=2)
    plt.xlabel('Reaction Time (min)', fontsize=12)
    plt.ylabel('Yield of R', fontsize=12)
    plt.title(f'Yield-Time (T={T}°C, Ccat={Ccat} mM)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 120)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('yield-time.png', dpi=300)
    plt.show()
    print("Task 1 plot saved to 'yield-time.png'")

    return time_range, yields


def plot_yield_vs_temperature():
    """
    任务2：当反应时间为40分钟、催化剂浓度为3 mM时，绘制反应产率随温度的变化曲线。
    """
    tre = 40  # min
    Ccat = 3  # mM

    # 温度范围：20 到 100 °C
    temp_range = np.linspace(20, 100, 100)
    yields = []

    for T in temp_range:
        condition = [T, tre, Ccat]
        y = kinetics(condition)
        yields.append(y)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(temp_range, yields, 'r-', linewidth=2)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Yield of R', fontsize=12)
    plt.title(f'Yield-Temperature (tre={tre} min, Ccat={Ccat} mM)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('yield-temperature.png', dpi=300)
    plt.show()
    print("Task 2 plot saved to 'yield-temperature.png'")

    return temp_range, yields


def plot_yield_vs_catalyst():
    """
    任务3：当温度为60°C、反应时间为40分钟时，绘制反应产率随催化剂浓度的变化曲线。
    """
    T = 60  # °C
    tre = 40  # min

    # 催化剂浓度范围：0.5 到 10 mM
    cat_range = np.linspace(0.5, 10, 100)
    yields = []

    for Ccat in cat_range:
        condition = [T, tre, Ccat]
        y = kinetics(condition)
        yields.append(y)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(cat_range, yields, 'g-', linewidth=2)
    plt.xlabel('Catalyst Concentration (mM)', fontsize=12)
    plt.ylabel('Yield of R', fontsize=12)
    plt.title(f'Yield-Catalyst Concentration (T={T}°C, tre={tre} min)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 10)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('yield-catalyst.png', dpi=300)
    plt.show()
    print("Task 3 plot saved to 'yield-catalyst.png'")

    return cat_range, yields


def run_all_tasks():
    """
    运行所有三个任务
    """
    print("=" * 50)
    print("Running Task 1: Yield vs Reaction Time")
    print("=" * 50)
    plot_yield_vs_time()

    print("\n" + "=" * 50)
    print("Running Task 2: Yield vs Temperature")
    print("=" * 50)
    plot_yield_vs_temperature()

    print("\n" + "=" * 50)
    print("Running Task 3: Yield vs Catalyst Concentration")
    print("=" * 50)
    plot_yield_vs_catalyst()

    print("\n" + "=" * 50)
    print("All tasks completed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tasks()