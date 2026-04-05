## 底层算子框架

![workflow](./images/workflow.png)

## AlphaGen

![alphagen](./images/alphagen.png)


## 相比于 AlphaGen 项目的变化

- `.expression.py` 中给出了**截面排序变换** `CSRANK` 的算法, `alphagen`, `alphaforge` 均对 `tensor` 直接使用 `argsort().argsort()`, 这样会导致值相同的并列值被强行打散, 在本项目中进行了平均秩处理.

- `.expression.py` 新增子类 `FORWARDRET`, 明确与 `RollingOperator` 区分, 添加限制条件 `window>=1`, 显式避免前瞻误差.

- `.calculator.py` 增加了 `winsorize` 去极值模块, 将极端值压缩至3个方差内

- `wrapper.py` 中执行 `AlphaEnvWrapper.step()` 时会将当前 `action` 录入 `state`, `alphagen` 中直接写入 `action`, 而 `action==0` 对应 `SEP`, 但是 `policy.py` 中的 Embedding 规则又规定 `padding_idx=0`, 会导致将 `SEP` 和填充空位在编码时视作同一种, 应该是typo, 因为后续 Embedding 规则也是 $K+2$ 维, 已修改为写入 `action + 1`

- 调整了 `alphagen` 中 `core.py` 的检测逻辑

- `alphagen` 里的 AlphaPool 是随着采样过程不断更新的，这样会导致 critic 部分的结果不平稳，尽管在 `alphaqcm` 里试图解决这个问题，这里显式提供了选项.
