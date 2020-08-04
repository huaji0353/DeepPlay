# DeepPlay
rapid prototyping without code and interactive building ,visual training, interpretable deep network, with UIs distribution

### 我提出可视化方案（具体实现还是很远的，哪位大佬看到有兴趣，能搞出来就好了）
用图去可视化操作过程（历史）
比如ps的历史记录是一维，不是一维而是一张图
而且每一步都能用调用算子去可视化当前张量（namedtensor，按照字典去解析可视化张量实际含义）
你也能回退到某个节点去建新分支，再者融合分支（残差）
这种可视化可比敲那堆代码方便多了
，再者计算时直接运行，backend交给autograd，让整个（做的行为）能优化
有人说这是git branch，很像，从输入看就只是history而已，但包括进req_grad的叶节点，实际上表示的应该是计算图

在实际构建网络（神经网络架构）时，重复的堆积木不太好，所以你也可以用简易代码去操作视觉历史节点
（简易代码是只允许很简单的逻辑操作，没有任何触及底层的操作）

(korean blogs)[https://github.com/subinium/subinium.github.io/blob/0a3342750d1ffa0e78f6f1d7c56d1d9c6729018c/_posts/LR/2019-06-13-LR014.md]

### survey collection
(feifeife/All-about-XAI)[https://github.com/feifeife/All-about-XAI]
(rehmanzafar/xai-iml-sota)[https://github.com/rehmanzafar/xai-iml-sota]
