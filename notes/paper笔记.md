标题：Fluidic Topology Optimization with an Anisotropic Mixture Model (2022)

N-S方程：
$$
\rho\dfrac{\mathrm D \boldsymbol v}{\mathrm D t} =\rho(\dfrac{\part \boldsymbol v}{\part t} + \boldsymbol v\cdot \nabla \boldsymbol v)  = \mu\nabla^2\boldsymbol v - \nabla p^*
$$
其中$p^*$为测压管压强，$p^* = p + \rho gz$，求导得$\nabla p^* = \nabla p - \rho\boldsymbol{g}$.

斯托克斯流：粘性力远大于惯性力的流，雷诺数远大于1，惯性力可被忽略，可以理解为粘性非常强，几乎不会动，于是对流项$\boldsymbol v\cdot \nabla \boldsymbol v$被忽略，N-S方程变为
$$
\rho\dfrac{\part \boldsymbol v}{\part t}+ \nabla p^* = \mu\nabla^2\upsilon
$$
如果考虑定常状态，即时变项也为0，得到
$$
\nabla p^* = \mu\nabla^2\upsilon
$$
此时（测压管）压强与粘性作用近似抵消（二者都比惯性项显著的多），可以认为合外力为0，该流体处于平衡状态，这种流动状态称为蠕动流或斯托克斯流。

## 各向同性斯托克斯方程

首先作者给出各向同性准不可压缩的斯托克斯流（我真应该去看看DU 2020）下的约束速度场的公式：
$$
\min\limits_{\boldsymbol v}\int_\Omega \mu||\nabla\boldsymbol v||^2_F\;\mathrm dx + \int_\Omega \lambda(\nabla\cdot\boldsymbol v)^2\;\mathrm dx,\\
\begin{align*}
     \text{ s.t.}\  &\boldsymbol v(\boldsymbol x) = \boldsymbol v_D(\boldsymbol x),\forall \boldsymbol x\in\part\Omega_D.\\
    &\boldsymbol v(\boldsymbol x) \cdot\boldsymbol n(\boldsymbol x) = 0,\forall \boldsymbol x\in\part\Omega_F.\\
\end{align*}
$$

- $\Omega$代表**流体存在的区域**，$\Omega\subset \R^d(d=2,3)$
- $\mu$代表动力粘度（Dynamic viscosity，Pa\*s或N\*s/m^2），$\lambda$代表不可压缩程度，$\lambda\to+\infin$代表不可压缩的斯托克斯流
- $||\cdot||_F$代表矩阵的Frobenius norm, $||A||_F = \sqrt{\sum\limits_i\sum\limits_j|a_{ij}|^2} = \sqrt{\text{trace}(A\cdot A)}$，在向量上就是向量的L2范数
- $\part\Omega$代表定义在域边界的边界条件，$\part\Omega = \part\Omega_D\cup\part\Omega_F\cup\part\Omega_O$，分别代表：
  - Dirichlet边界，此边界将流体速度指定为一个速度$\boldsymbol v_D$，要么是在入口指定为入口速度，要么是在no-slip边界指定为$\boldsymbol v_D$.
  - Free-slip边界，要求流体速度沿边界法向的投影为0
  - Open边界，不对速度作出限制，自动满足0牵引力条件（这应该是在DU 2020中讲的？），一般是适合建模在自由流体的出口处
- DU 2020是提出了这一在准不可压缩斯托克斯流动模型及装置设计问题中有数值优势的方程，并提出了流体装置的计算设计管线，但它们的方法是被参数空间限制的。
- $\nabla\cdot\boldsymbol v$速度的散度，同流体力学不可压缩方程中的那项。$\nabla\boldsymbol v$代表什么我应该去看看DU 2020

## 各向异性斯托克斯方程

各向异性、准不可压缩斯托克斯流体的能量最小化方程：
$$
\min\limits_{\boldsymbol v} E_{m,\mu}[\boldsymbol v] + E_{m,\lambda}[\boldsymbol v] + E_{f}[\boldsymbol v],\\
\text{s.t.}\ \boldsymbol v(\boldsymbol x) = \boldsymbol v_D(\boldsymbol x),\forall \boldsymbol x\in\part\mathcal B_D.
$$
where
$$
\begin{align}
E_{m,\mu}[\boldsymbol v] &:= \int_\mathcal B\mu||\nabla \boldsymbol v\boldsymbol K^{\frac{1}{2}}_m(\boldsymbol x)||^2_F\mathrm dx\\
E_{m,\lambda}[\boldsymbol v] &:= \int_\mathcal B\lambda(\boldsymbol x)(\nabla\cdot \boldsymbol v)^2\mathrm dx\\
E_{f}[\boldsymbol v] &:= \int_\mathcal B||\boldsymbol K^{\frac{1}{2}}_f(\boldsymbol x)||^2_2\mathrm dx
\end{align}
$$

- $m,f$建模material（表示phase在某点是各向同性 or 各向异性）和frictional（表示phase在某点受到的外摩擦力）的影响
- $\boldsymbol K_m = \boldsymbol I, \boldsymbol K_f = \boldsymbol 0$时方程退化为各向同性下的方程
- 首先将流体域$\Omega$改成了axis-aligned，足够大的，包含$\Omega$的box $\mathcal B\subset\R^d(d=2,3)$，两者的不同是，$\Omega$只包含流体域，而$\mathcal B$包含流体域+固体域
- 引入代表material和frictional影响的两个对称半正定矩阵$\boldsymbol K_m,\boldsymbol K_f:\mathcal B\to S^d_+$, （这个式子的意思应该是，$\boldsymbol K_m,\boldsymbol K_f$接收一个$\mathcal B$中的坐标$x$，然后返回该$x$下的材料、摩擦性质，返回的是一个$d$维正向量，用$S^d_+$表示）
  - 对称：$A_{ij} = A_{ji}$
  - 半正定（Positive Semi-Definite）：对于任意非零向量 $x$，都有$x^TAx \geq 0$，特点：
    - 所有的特征值都非负。
    - 对应于非零特征值的特征向量线性无关。
    - 可以作为各种优化问题的约束条件，尤其是在凸优化领域中。
    - 在统计学中，协方差矩阵是对称正半定矩阵的一个例子，因为协方差矩阵描述了各个随机变量之间的相关性。
- 将$\lambda$换为空间分布的域而非常数，$\lambda:\mathcal B\to\R^+$，接收与$\mathcal B$同维度的$x$返回一个正数
- 引入$\boldsymbol K_m,\boldsymbol K_f, \lambda$使得新材料的模型可以对不同方向的速度作出各向异性的回复？why
- 边界划分（边界为什么是$\mathcal B$的偏导？partition是“划分”的意思，指的是将边界分割成若干互不重叠的部分）：$\part\mathcal B = \part\mathcal B_D\cup\part\mathcal B_O$
  - Dirichllet：用在流体系统的进口（$v_D$被设置为规定值）和固体项的边界（$v_D = 0$）
  - Open：如前，模拟零牵引力，用在自由流体的出口
  - 现在只有两个并不是取消了solid-fluid边界上no-slip和free-slip条件，而是将其的表示纳入到$\boldsymbol K_m,\boldsymbol K_f, \lambda$的选择中

$\mathcal B$中的每个格子有$\boldsymbol K_m,\boldsymbol K_f, \lambda$，精心选择它们就可以表示solid, fluid项和no-slip, free-slip边界。

- Fluid-phase material：$\boldsymbol K_m = \boldsymbol I,\boldsymbol K_f = \boldsymbol 0, \lambda = \lambda_0$，表示在流体域$\Omega$内流体受到的外摩擦力为0，此时方程变为各向同性斯托克斯方程
- Solid-phase material：$\boldsymbol K_m = \boldsymbol I,\boldsymbol K_f = k_f \boldsymbol I \ \text{where} \ k_f\to+\infty, \lambda = \lambda_0$，根据方程（4），此时solid-phase内部的$v$只能取0。$\boldsymbol K_m ,\lambda $的取值此时对solid内部的$v$的结果没有影响，但建议这么取，得到一个各向同性、准不可压缩的solid material
- No-slip-boundary material：$\boldsymbol K_m = \boldsymbol I,\boldsymbol K_f = k_f\boldsymbol I\ \text{where} \ k_f\to+\infty, \lambda = \lambda_0$，不允许滑移边界让流体速度为0，所以参数选择上同solid-phase
- Free-slip-boundary material：



